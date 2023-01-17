import pandas as pd

from collections import Counter

import numpy as np

import torchvision

import torch.nn as nn

import torch

import torch.nn.functional as F

from torch.autograd import Variable

import time

import copy

from sklearn.metrics import f1_score, classification_report, confusion_matrix

from sklearn.utils import shuffle
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



# Fill missing values

df_train.fillna("_##_", inplace=True)  

df_test.fillna("_##_", inplace=True)
def split(df):

    split_questions = []

    for data in df.itertuples():

        split_q_tmp = []

        for w in data[2].split(' '):

            split_q_tmp.append(w)

        split_questions.append(split_q_tmp)

    return split_questions

        

X_train = split(df_train)

y_train = df_train['target'].values.tolist()

qid_test = df_test['qid'].values.tolist()

X_test = split(df_test)



print(len(X_train))

print(len(y_train))

print(len(X_test))

print(len(qid_test))
def get_stat(X,y):

    num_sincere = 0

    num_insincere = 0

    for i in range(len(X)):

        if y[i] == 0:

            num_sincere += 1

        elif y[i] == 1:

            num_insincere += 1

        else:

            print("We should not end up here! i={}".format(i))

    total = num_sincere + num_insincere

    print("Sincere: {}, insincere: {}".format(num_sincere, num_insincere))

    print("ratio_sincere, ratio_insincere: {:.4f}, {:.4f}".format(num_sincere / total, num_insincere / total))

    return num_sincere, num_insincere



num_sincere, num_insincere = get_stat(X_train, y_train)
def balance(X, y, n):

    newX = []

    newY = []

    sincere_taken = 0

    insincere_taken = 0

    for i in range(len(X)):

        if y[i] == 0 and sincere_taken <= n:

            newX.append(X[i])

            newY.append(y[i])

            sincere_taken += 1

        elif y[i] == 1 and insincere_taken <= n:

            newX.append(X[i])

            newY.append(y[i])

            insincere_taken += 1

    print("Sincere taken: {}, insincere taken: {}".format(sincere_taken, insincere_taken))

    return newX, newY



print("Extracting {} questions from each class!".format(num_insincere))



X_train, y_train = balance(X_train, y_train, num_insincere)



# Random shuffle

X_train, y_train = shuffle(X_train, y_train)



print(len(X_train))

print(len(y_train))
def get_all_words(questions):

    all_words = []

    for question in questions:

        for word in question:

            all_words.append(word)

    return all_words



def build_vocab(all_words):

    count = Counter(all_words)

    return sorted(count, key=count.get, reverse=True)



def vocab_to_integer(vocab):

    ''' Map each vocab words to an integer.

        Starts at 1 since 0 will be used for padding.'''

    return {word: ii for ii, word in enumerate(vocab, 1)}



all_words = get_all_words(X_train + X_test)

vocab = build_vocab(all_words)

vocab_to_int = vocab_to_integer(vocab)



print("Vocab size: {}".format(len(vocab_to_int)))
def embed_word_to_int(X, vocab_to_int):

    embedded_X = []

    for q in X:

        tmp_X = []

        for w in q:

            tmp_X.append(vocab_to_int[w])

        embedded_X.append(tmp_X)

    return embedded_X

        

X_train = embed_word_to_int(X_train, vocab_to_int)

X_test = embed_word_to_int(X_test, vocab_to_int)
def create_validation_set(X, y, factor=0.8):

    num_train = int(len(X) * factor)

    X_train = X[:num_train]

    y_train = y[:num_train]

    X_val = X[num_train:]

    y_val = y[num_train:]

    return X_train, y_train, X_val, y_val



X_train_split, y_train_split, X_val_split, y_val_split = create_validation_set(X_train, y_train, factor=0.8)



print(len(X_train_split))

print(len(y_train_split))

print(len(X_val_split))

print(len(y_val_split))
def pad_features(questions, sequence_length=50):

    ''' Pad each question with zeros to the same length.

        Padding is done in the beginning of the sentence.

        If question is truncated the truncation starts at the end of the question. '''

    features = np.zeros((len(questions), sequence_length), dtype=int)

    for i, row in enumerate(questions):

        features[i, -len(row):] = np.array(row)[:sequence_length]

    return features



def format_labels(labels):

    ''' What we actually do is one-hot encode the labels so that a sincere question

        gets the label [0, 1] and a insincere questions gets the label [0, 1] '''

    y = np.zeros((len(labels), 2), dtype=int)

    for i in range(len(labels)):

        if labels[i] == 0:

            y[i] = [1,0]

        else:

            y[i] = [0,1]

    return y

        

# Calculate the max length and pad all questions to this length

max_train_len = max(Counter([len(x) for x in X_train]))

max_test_len = max(Counter([len(x) for x in X_test]))

max_len = max(max_train_len, max_test_len)



pad_length = max_len



# Pad and format

X_train_pad = pad_features(X_train_split, pad_length)

y_train_pad = format_labels(y_train_split)

X_val_pad = pad_features(X_val_split, pad_length)

y_val_pad = format_labels(y_val_split)

X_test_pad = pad_features(X_test, pad_length)
def numpy_to_tensor(X, y=None):

    X_tensor = Variable(torch.from_numpy(X).long(),

                        requires_grad=False)

    y_tensor = None

    if y is not None:

        y_tensor = Variable(torch.from_numpy(y).float(),

                            requires_grad=False)

    return X_tensor, y_tensor



X_train_tensor, y_train_tensor = numpy_to_tensor(X_train_pad, y_train_pad)

X_val_tensor, y_val_tensor = numpy_to_tensor(X_val_pad, y_val_pad)

X_test_tensor, _ = numpy_to_tensor(X_test_pad)



print(X_train_tensor.shape)

print(y_train_tensor.shape)

print(X_val_tensor.shape)

print(y_val_tensor.shape)

print(X_test_tensor.shape)
# Define hyperparams

MINIBATCH_SIZE = 16

LEARNING_RATE = 1e-3

EPOCHS = 75

SGD_MOMENTUM = 0.9

LSTM_HIDDEN_SIZE = 100

LSTM_EMBEDDING_SIZE = pad_length
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

test_dataset = torch.utils.data.TensorDataset(X_test_tensor)



train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                                batch_size=MINIBATCH_SIZE,

                                                shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,

                                                batch_size=MINIBATCH_SIZE,

                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                                batch_size=MINIBATCH_SIZE,

                                                shuffle=False)



classes = {0:"sincere", 1:"insincere"} # These are the classes we have in the dataset (labels)



# Save loaders in single dict

dataloaders_dict = {"train": train_loader, "val": val_loader, "test": test_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



print("Running on device: {}".format(device))

if torch.cuda.is_available():

    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))

    print("torch.cuda.device(0): {}".format(torch.cuda.device(0)))

    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))

    print("torch.cuda.get_device_name(0): {}".format(torch.cuda.get_device_name(0)))
class LSTM01(nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, device):

        super(LSTM01, self).__init__()

        """

        Arguments

        ---------

        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator

        output_size : 2 = (sincere, insincere)

        hidden_sie : Size of the hidden_state of the LSTM

        vocab_size : Size of the vocabulary containing unique words

        embedding_length : Embeddding dimension of our word embeddings (glove dimension if using glove)

        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 



        """

        self.batch_size = batch_size

        self.output_size = output_size

        self.hidden_size = hidden_size

        self.vocab_size = vocab_size

        self.embedding_length = embedding_length



        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.

        # self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.

        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=False)

        self.label = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid() # Should use sigmoid for binary classification



    def forward(self, input_sentence, batch_size=None):

        if batch_size is not None and batch_size is not self.batch_size:

            print("Got batch size {} in LSTM".format(batch_size))

        """ 

        Parameters

        ----------

        input_sentence: input_sentence of shape = (batch_size, num_sequences)

        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)



        Returns

        -------

        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM

        final_output.shape = (batch_size, output_size)

        

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''

        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)

        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

        if batch_size is None:

            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device)) # Initial hidden state of the LSTM

            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device)) # Initial cell state of the LSTM

        else:

            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))

            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

        final_output = self.sigmoid(output)

        return final_output

    

    def predict(self, inp, batch_size):

        return self.forward(inp, batch_size=batch_size)



    

# Create model

model = LSTM01(batch_size=MINIBATCH_SIZE, output_size=2,

                           hidden_size=LSTM_HIDDEN_SIZE, vocab_size=len(vocab_to_int)+1,

                            embedding_length=LSTM_EMBEDDING_SIZE, device=device)



# Print the model we just instantiated

print(model)



# Send model to deivce (GPU hopefully!)

model = model.to(device)



# Binary classification -> we use binary cross entropy loss func

loss_function = nn.BCELoss()



# We use the SGD optimizer algorithm

# https://pytorch.org/docs/stable/optim.html

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)
def train(model, train_loader, criterion, optimizer, minibatch_size, epoch):

    ''' Standard pytorch procedure for training a model '''

    print("Training on device: {}".format(device))

    

    # declare variables

    t0 = time.time()

    total_epoch_loss = 0.0

    total_epoch_acc = 0

    steps = 0

    model.train() # Set model to training mode

    

    for idx, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device)

        labels = labels.to(device)

            

        # zero the parameter gradients

        optimizer.zero_grad()

        

        with torch.set_grad_enabled(True):

            # forward

            outputs = model.predict(inputs, inputs.size(0))

            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            

            # calc accuracy

            num_corrects = torch.sum(preds == torch.argmax(labels,1))

            acc = 100.0 * num_corrects/inputs.size(0)

            

            # backward

            loss.backward()

            optimizer.step()

        

        steps += 1

        

        if steps % 1000 == 0:

            time_elapsed = time.time() - t0

            print (f"Epoch: {epoch+1} [{time_elapsed//60:.0f}m, {time_elapsed%60:.0f}s], Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%")

        

        total_epoch_loss += loss.item()

        total_epoch_acc += acc.item()

             

    return total_epoch_loss/len(train_loader), total_epoch_acc/len(train_loader)



def validate(model, val_loader, criterion, minibatch_size):

    ''' In the validation procedure we keep track of the weights that gives the best val acc

    In the end, we load the weights that gave us the best accuracy '''

    print("Validating on device: {}".format(device))



    # declare variables

    t0 = time.time()

    total_epoch_loss = 0.0

    total_epoch_acc = 0

    model.eval() # Set model to evaluate mode



    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(val_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)



            outputs = model.predict(inputs, inputs.size(0))

            loss = criterion(outputs, labels) 

            _, preds = torch.max(outputs, 1)



            # calc accuracy

            num_corrects = torch.sum(preds == torch.argmax(labels,1))

            acc = 100.0 * num_corrects/inputs.size(0)   



            total_epoch_loss += loss.item()

            total_epoch_acc += acc.item()

        

    return total_epoch_loss/len(val_loader), total_epoch_acc/len(val_loader)



def test(model, test_loader, minibatch_size, device):

    ''' Test the model on test set and save results '''

    print("Testing on device: {}".format(device))

    t0 = time.time()

    results = {'qid':[], 'prediction':[]}

    with torch.no_grad():

        for idx, data in enumerate(test_loader):

            inputs = data[0]            

            inputs = inputs.to(device)

            # get prediction

            outputs = model.predict(inputs, inputs.size(0))

            _, preds = torch.max(outputs, 1)

            # save each prediction

            for pred in preds:

                results['prediction'].append(pred.item())               

    time_elapsed = time.time() - t0

    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return results
best_wts = copy.deepcopy(model.state_dict())

best_acc = 0

best_epoch = 0



for epoch in range(EPOCHS):

    # train

    train_loss, train_acc = train(model, 

                                  dataloaders_dict['train'], 

                                  loss_function, optimizer, 

                                  MINIBATCH_SIZE, 

                                  epoch)

    # validate

    val_loss, val_acc = validate(model, 

                                 dataloaders_dict['val'],

                                 loss_function,

                                 MINIBATCH_SIZE)

    # save weights that give highest val acc

    if val_acc > best_acc:

        best_wts = copy.deepcopy(model.state_dict())

        best_acc = val_acc

        best_epoch = epoch

        

    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    print("Best acc: {:.4f}%, at epoch: {}".format(best_acc, best_epoch))

    

# load best weights

model.load_state_dict(best_wts)
# Run test set and get results

results = test(model, dataloaders_dict['test'], MINIBATCH_SIZE, device)

results['qid'] = qid_test



# Print some stats

print("Number of qids: {}, number of predictions: {}".format(len(results['qid']), format(len(results['prediction']))))



# Save results

df = pd.DataFrame(data=results)

df.to_csv('submission.csv', index=False)

print("Saved csv to disk!")