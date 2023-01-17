# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
WEIGHT_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
TRAIN_FILE = '../input/train.csv'
TEST_FILE = '../input/test.csv'
import re
import unicodedata
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import torch
class Dataset:

    def __init__(self,
                 train_df,
                 test_df,
                 input_field='question_text',
                 target_filed='target',
                 validation_size=0.1,
                 clean_function=None,
                 verbose=False):

        self.input_field = input_field
        self.target_filed = target_filed
        self.validation_size = validation_size
        self.verbose = verbose

        self.train = None
        self.validation = None
        self.test = None

        self.test_qid = []

        self.clean_function = clean_function if clean_function is not None else lambda x: x
        self.words = set()

        self.init_data(train_df, test_df)

    def init_data(self, train_df, test_df, validation_size=None):

        validation_size = validation_size if validation_size is not None else self.validation_size

        train_df, validation_df = train_test_split(train_df,
                                                   test_size=validation_size,
                                                   stratify=train_df[self.target_filed])

        self.train = self._form_data_field(df=train_df, title='Collect train')
        self.validation = self._form_data_field(df=validation_df, title='Collect validation')
        self.test = self._form_data_field(df=test_df, title='Collect test')
        self.test_qid = test_df.qid

    def _form_data_field(self, df, title=''):

        data = []

        indexes = tqdm(df.index, desc=title) if self.verbose else df.index

        for index in indexes:

            if len(df.loc[index, self.input_field]) <= 3:
                continue

            text = wordpunct_tokenize(self.clean_function(df.loc[index, self.input_field]))

            for word in text:
                self.words.add(word)

            target = df.loc[index, self.target_filed] if self.target_filed in df else False

            if not text:
                continue

            sample = {
                self.input_field: text,
                self.target_filed: target
            }

            data.append(sample)

        return data

    def batch_generator(self, data_type, batch_size=32, sequence_max_length=None):

        data = self.__dict__[data_type]

        for n_batch in range(len(data) // batch_size):

            batch = data[n_batch * batch_size:(n_batch + 1) * batch_size]

            sequence_max_length = sequence_max_length if sequence_max_length is not None else -1

            x = [sample[self.input_field][:sequence_max_length] for sample in batch]

            y = [sample[self.target_filed] for sample in batch]

            yield x, y
class Cleaner:

    def __init__(self):

        pass

    @staticmethod
    def unicode_to_ascii(x):

        return ''.join(
            c for c in unicodedata.normalize('NFD', x)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalize_string(x):

        x = re.sub(r"([.!?])", r" .", x)
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        s = re.sub(r"[^a-zA-Z.#]+", r" ", x)

        return s

    def clean(self, sentence):

        x = sentence.strip().lower()
        x = self.unicode_to_ascii(x)
        x = self.normalize_string(x)

        return x
class Wrapper:

    def __init__(self, dataset, model, model_name, criterion, optimizer, sequence_max_length=32):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = dataset

        self.model = model.to(self.device)
        self.model_name = model_name
        self.criterion = criterion
        self.optimizer = optimizer

        self.sequence_max_length = sequence_max_length

        self.losses = []
        self.batch_mean_losses = []

        self.f1 = []
        self.best_f1 = 0
        self.best_threshold = 0
        self.best_epoch = 0

        self.batch_size = 0
        self.epochs = 0

    @staticmethod
    def search_best_threshold_for_f1_score(y_prediction, y_true):

        y_prediction = y_prediction.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy().astype(int)

        best_f1, best_thresh = 0, 0

        for thresh in np.arange(0.1, 0.501, 0.01):

            thresh = np.round(thresh, 2)

            pred = (y_prediction > thresh).astype(int)

            if len(pd.Series(pred).value_counts()) > 1:

                f1 = f1_score(y_true, pred)

                if f1 > best_f1:
                    best_thresh = thresh
                    best_f1 = f1

        return best_f1, best_thresh

    def train(self, epochs=5, batch_size=32, verbose=False, save=False):

        self.epochs = epochs
        self.batch_size = batch_size

        self.losses = []
        self.batch_mean_losses = []

        for n_epoch in range(1, self.epochs+1):

            if verbose:
                pbar = tqdm(total=len(self.dataset.train) // self.batch_size, desc='Train Epoch {}'.format(n_epoch))

            batch_losses = []

            for x, y in self.dataset.batch_generator(data_type='train',
                                                     batch_size=self.batch_size):

                y_prediction, y = self.model(x, y)

                loss = self.criterion(y_prediction, y)

                self.losses.append(loss.item())
                batch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if verbose:
                    pbar.update(1)

            batch_mean_loss = np.mean(batch_losses)

            self.batch_mean_losses.append(batch_mean_loss)

            if verbose:
                pbar.close()

            with torch.no_grad():

                y_validation_prediction = torch.Tensor().to(self.device)
                y_validation = torch.Tensor().to(self.device)

                if verbose:
                    pbar = tqdm(total=len(self.dataset.validation) // self.batch_size,
                                desc='Validation Epoch {}'.format(n_epoch))

                for x, y in self.dataset.batch_generator(data_type='validation', batch_size=self.batch_size):

                    y_prediction, y = self.model(x, y)

                    y_validation_prediction = torch.cat((y_validation_prediction, y_prediction))
                    y_validation = torch.cat((y_validation, y))

                    if verbose:
                        pbar.update(1)

            if verbose:
                pbar.close()

            f1, threshold = self.search_best_threshold_for_f1_score(y_validation_prediction, y_validation)

            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_threshold = threshold
                self.best_epoch = n_epoch
                
                if save:
                    self.save_model()

            self.f1.append(f1)

            message = 'Epoch: [{}/{}] | Loss: {:.5f} | F1 Score: {:3f} | Threshold: {}'.format(
                n_epoch,
                self.epochs,
                batch_mean_loss,
                f1,
                threshold,
            )

            if verbose:
                print(message)

    def save_model(self):

        state_dict = {
            'epoch': self.best_epoch,
            'f1': self.best_f1,
            'threshold': self.best_threshold,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.batch_mean_losses,
        }

        directory = 'models_checkpoints/{}/'.format(self.model_name)

        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        torch.save(state_dict, directory + 'best.pt')

    def load_model(self, file=None):

        file = file if file is not None else 'models_checkpoints/{}/best.pt'.format(self.model_name)

        checkpoint = torch.load(file)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.best_epoch = checkpoint['epoch']
        self.best_f1 = checkpoint['f1']
        self.best_threshold = checkpoint['threshold']

        self.losses = checkpoint['losses']

    def plot_losses(self, losses_type='batch_mean_losses', figsize=(16, 14), xlabel='Epoch'):

        losses = self.__dict__[losses_type]

        plt.figure(figsize=figsize)

        plt.plot([0] + losses)

        plt.title('Losses')
        plt.xlabel(xlabel)
        plt.ylabel('Loss')

        plt.grid()

        plt.ylim(0, np.max(losses) * 1.2)
        plt.xlim(1, len(losses))

    def submission(self, verbose=False):

        with torch.no_grad():

            y_test_prediction = torch.Tensor().to(self.device)

            if verbose:
                pbar = tqdm(total=len(self.dataset.test),
                            desc='Test')

            for x, y in self.dataset.batch_generator(data_type='test', batch_size=1):

                y_prediction, _ = self.model(x, y)

                y_test_prediction = torch.cat((y_test_prediction, y_prediction))

                if verbose:
                    pbar.update(1)

        if verbose:
            pbar.close()

        y_test_prediction = y_test_prediction.cpu().detach().numpy()

        submission = pd.DataFrame(data={
            'qid': self.dataset.test_qid,
            'prediction': (y_test_prediction > self.best_threshold).astype(int)
        })

        return submission
class EmbeddingFromPretrained(nn.Module):

    def __init__(self,
                 weight_file,
                 vector_size,
                 sequence_max_length=64,
                 pad_token='PAD',
                 pad_after=True,
                 existing_words=None,
                 verbose=False):

        super(EmbeddingFromPretrained, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.weight_file = weight_file
        self.vector_size = vector_size
        self.sequence_max_length = sequence_max_length

        self.pad_token = pad_token
        self.pad_index = 0

        self.pad_after = pad_after

        self.existing_words = existing_words if existing_words is not None else []

        self.word2index = {
            self.pad_token: self.pad_index
        }

        self.index2word = {
            self.pad_index: self.pad_token
        }

        self.embedding_layer = self.__collect_embeddings__(verbose=verbose)

    def __collect_embeddings__(self, verbose=False):

        embedding_matrix = [np.zeros(shape=(self.vector_size, ))]

        with open(file=self.weight_file, mode='r', encoding='utf-8', errors='ignore') as file:

            index = len(self.word2index)

            lines = tqdm(file.readlines(), desc='Collect embeddings') if verbose else file.readlines()

            for line in lines:

                line = line.split()

                word = ' '.join(line[:-self.vector_size])
                embeddings = np.asarray(line[-self.vector_size:], dtype='float32')

                if not word or embeddings.shape[0] != self.vector_size or word not in self.existing_words:
                    continue

                self.word2index[word] = index
                self.index2word[index] = word

                embedding_matrix.append(embeddings)

                index += 1

        return torch.nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix)).to(self.device)

    def forward(self, input_batch, targets_batch):

        sequence_max_length = self.sequence_max_length if self.sequence_max_length is not None \
            else max([len(sample) for sample in input_batch])

        sequence_lengths = []

        embedded_batch = torch.Tensor(size=(len(input_batch), sequence_max_length, self.vector_size)).to(self.device)

        for n_sample in range(len(input_batch)):

            tokens = [self.word2index[token] for token in input_batch[n_sample] if token in self.word2index]
            tokens = tokens[:sequence_max_length]

            if not tokens:
                targets_batch.pop(n_sample)
                continue

            sequence_lengths.append(len(tokens))

            if len(tokens) < sequence_max_length:

                pads = [self.pad_index] * (sequence_max_length - len(tokens))

                if self.pad_after:
                    tokens = tokens + pads
                else:
                    tokens = pads + tokens

            tokens = torch.LongTensor(tokens).to(self.device)

            embedded_batch[n_sample] = self.embedding_layer(tokens).to(self.device)

        targets_batch = torch.Tensor(targets_batch).to(self.device)

        if embedded_batch.sum() == 0:
            return None, None, None

        sequence_lengths = torch.Tensor(sequence_lengths)

        sequence_lengths, permutation_idx = sequence_lengths.sort(descending=True)

        embedded_batch = embedded_batch[permutation_idx]
        sequence_lengths = sequence_lengths.to(self.device)
        targets_batch = targets_batch[permutation_idx]

        return embedded_batch, sequence_lengths, targets_batch
class NeuralNetwork(nn.Module):

    def __init__(self,
                 sizes,
                 activation_function=F.relu,
                 sigmoid_output=False):

        super(NeuralNetwork, self).__init__()

        self.sizes = list(sizes)
        self.activation_function = activation_function
        self.sigmoid_output = sigmoid_output

        if self.sizes[-1] != 1 and self.sigmoid_output:
            self.sizes.append(1)

        self.input_size = self.sizes[0]
        self.output_size = self.sizes[-1]

        self.linear_1 = nn.Linear(in_features=self.sizes[0], out_features=self.sizes[1])

        if len(self.sizes) > 3:
            self.linear_2 = nn.Linear(in_features=self.sizes[1], out_features=self.sizes[2])

        if len(self.sizes) > 4:
            self.linear_3 = nn.Linear(in_features=self.sizes[2], out_features=self.sizes[3])

        if len(self.sizes) > 5:
            self.linear_4 = nn.Linear(in_features=self.sizes[3], out_features=self.sizes[4])

        self.linear_last = nn.Linear(in_features=self.sizes[-2], out_features=self.sizes[-1])

        # its not work
        #
        # self.layers = []
        # for n in range(len(self.sizes[:-1])):
        #     self.__dict__['linear_layer_{}'.format(n)] = nn.Linear(in_features=self.sizes[n],
        #                                                            out_features=self.sizes[n+1])
        #     self.layers.append(self.__dict__['linear_layer_{}'.format(n)])

    def forward(self, x, x_lengths=None):

        # for n, layer in enumerate(self.layers):
        #
        #     x = layer.forward(x)
        #
        #     if n == len(self.sizes) and self.sigmoid_output:
        #         x = torch.sigmoid(x)
        #         return x[:, 0]
        #
        #     else:
        #         x = self.activation_function(x)

        x = self.linear_1(x)
        x = self.activation_function(x)

        if len(self.sizes) > 3:
            x = self.linear_2(x)
            x = self.activation_function(x)

        if len(self.sizes) > 4:
            x = self.linear_3(x)
            x = self.activation_function(x)

        if len(self.sizes) > 5:
            x = self.linear_4(x)
            x = self.activation_function(x)

        x = self.linear_last(x)
        x = torch.sigmoid(x)

        return x
class DAN(nn.Module):

    def __init__(self,
                 embedding_layer=None,
                 weight_file=None,
                 embedding_size=300,
                 sizes=(300, 128, 64),
                 activation_function=F.relu,
                 sigmoid_output=True):

        super(DAN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if embedding_layer is not None:
            self.embedding_layer = embedding_layer
        elif weight_file is not None:
            self.embedding_layer = EmbeddingFromPretrained(weight_file=weight_file, vector_size=embedding_size)
        else:
            raise ValueError('Need embedding layer or weight file')

        self.embedding_layer = self.embedding_layer.to(self.device)

        self.neural_network = NeuralNetwork(sizes=sizes,
                                            activation_function=activation_function,
                                            sigmoid_output=sigmoid_output).to(self.device)

    def forward(self, tokens, target):

        x, _, y = self.embedding_layer(tokens, target)

        x = x.mean(dim=1)

        x = self.neural_network(x)

        return x[:, 0], y
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
cleaner = Cleaner()
dataset = Dataset(train_df=train_df, test_df=test_df, verbose=True, validation_size=0.1, clean_function=cleaner.clean)
embedding_layer = EmbeddingFromPretrained(weight_file=WEIGHT_FILE, vector_size=300, existing_words=dataset.words, verbose=True)
dan = DAN(
    sizes=[embedding_layer.vector_size, 256, 128, 64, 32],
    embedding_layer=embedding_layer
)
criterion = torch.nn.modules.loss.BCELoss()
optimizer = torch.optim.Adam(dan.parameters())
EPOCHS = 7
BATCH_SIZE = 32
dan_wrapper = Wrapper(dataset=dataset,
                      model=dan, 
                      model_name='DAN',
                      criterion=criterion, 
                      optimizer=optimizer)
dan_wrapper.model
dan_wrapper.train(epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True, save=False)
dan_wrapper.plot_losses()
submission = dan_wrapper.submission(verbose=True)
submission.prediction.value_counts()
submission.to_csv('submission.csv', index=False)
