import io
import warnings
from gluonnlp.embedding import TokenEmbedding
from itertools import takewhile, repeat
from types import SimpleNamespace
class CustomEmbedding(TokenEmbedding):
    """This embedding has an optimized version of loading embedding from a text file"""
    UNK_IDX = 0

    def __init__(self, text_file_path, dims, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)

        self._dims = dims
        embeddings_count = self._rawincount(text_file_path)
        self._load_embedding_txt_custom(text_file_path, ' ', embeddings_count, encoding='utf8')

    def _rawincount(self, filename):
        f = open(filename, 'rb')
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in bufgen)

    def _load_embedding_txt_custom(self, pretrained_file_path, elem_delim,
                            embeddings_count, encoding='utf8'):
        """Load embedding vectors from a pre-trained token embedding file.

        For every unknown token, if its representation `self.unknown_token` is encountered in the
        pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained token
        embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to the
        text embedding vector initialized by `self._init_unknown_vec`.

        If a token is encountered multiple times in the pre-trained text embedding file, only the
        first-encountered token embedding vector will be loaded and the rest will be skipped.
        """

        index = 0
        vec_len = None
        all_elems = nd.zeros(shape=(embeddings_count, self._dims))
        loaded_unknown_vec = None

        for line_num, line in CustomEmbedding._get_lines(pretrained_file_path, encoding):

            token, vector = CustomEmbedding._parse_embedding_line(line, elem_delim, self._dims)

            assert token and len(vector) == self._dims, \
                'line {} in {}: unexpected data format.'.format(line_num, pretrained_file_path)

            if token == self.unknown_token and loaded_unknown_vec is None:
                loaded_unknown_vec = vector

            elif token in self._token_to_idx:
                warnings.warn('line {} in {}: duplicate embedding found for '
                              'token "{}". Skipped.'.format(line_num, pretrained_file_path,
                                                            token))

            elif len(vector) == 1 and line_num == 0:
                warnings.warn('line {} in {}: skipped likely header line.'
                              .format(line_num, pretrained_file_path))
            else:
                if not vec_len:
                    vec_len = len(vector)

                    if self.unknown_token:
                        # Reserve a vector slot for the unknown token at the very beggining
                        # because the unknown token index is 0.
                        all_elems[index] = [0] * vec_len
                        index = index + 1
                else:
                    assert len(vector) == vec_len, \
                        'line {} in {}: found vector of inconsistent dimension for token ' \
                        '"{}". expected dim: {}, found: {}'.format(line_num,
                                                                   pretrained_file_path,
                                                                   token, vec_len, len(vector))
                all_elems[index] = vector
                index = index + 1

                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

        self._idx_to_vec = all_elems

        if self.unknown_token:
            if loaded_unknown_vec is None:
                self._idx_to_vec[CustomEmbedding.UNK_IDX] = self._init_unknown_vec(shape=vec_len)
            else:
                self._idx_to_vec[CustomEmbedding.UNK_IDX] = nd.array(loaded_unknown_vec)

    @staticmethod
    def _get_lines(pretrained_file_path, encoding):
        with io.open(pretrained_file_path, 'rb') as f:
            for line_num, line in enumerate(f):
                try:
                    line = line.decode(encoding)
                except ValueError:
                    warnings.warn('line {} in {}: failed to decode. Skipping.'
                                  .format(line_num, pretrained_file_path))
                    continue

                yield line_num, line

    @staticmethod
    def _parse_embedding_line(s, elem_delim, dims):
        i = 0

        token = ''
        vector = [None] * dims
        index = 0

        while True:
            j = s.find(elem_delim, i)
            if j < 0:
                if i < len(s):
                    vector[index - 1] = float(s[i:])
                break

            if index == 0:
                token = s[i:j]
            else:
                vector[index - 1] = float(s[i:j])

            index = index + 1
            i = j + 1
        return token, vector
import csv
import multiprocessing as mp
from gluonnlp import Vocab, data
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from nltk import word_tokenize


def get_sub_segment_from_list(dataset, indices):
    return ArrayDataset([dataset[i] for i in indices])
class QuoraDataset(ArrayDataset):
    """This dataset provides access to Quora insincere data competition"""

    def __init__(self, segment, root_dir="../input/"):
        self._root_dir = root_dir
        self._segment = segment
        self._segments = {
            'train': 'train.csv',
            'test': 'test.csv'
        }

        super(QuoraDataset, self).__init__(self._read_data())

    def _read_data(self):
        file_path = os.path.join(self._root_dir, self._segments[self._segment])
        with open(file_path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            # ignore 1st line - which is header
            data = [(i,) + tuple(row) for i, row in enumerate(reader) if i > 0]

        return data

class VocabProvider:
    """Provide word-level vocab based on datasets"""

    def __init__(self, datasets):
        self._datasets = datasets

    def get_vocab(self):
        all_words = [word for dataset in self._datasets for item in dataset for word in item[2]]
        vocab = Vocab(data.count_tokens(all_words), min_freq=3)
        return vocab

class DataTokenizer:
    """Run tokenizer on all cores"""

    def __init__(self, run_async=True):
        self._run_async = run_async

    def _tokenize(self, item):
        tokenized_item = [word.lower() for word in word_tokenize(item[2])]

        if len(item) == 4:
            return item[0], item[1], tokenized_item, float(item[3])
        else:
            return item[0], item[1], tokenized_item

    def __call__(self, dataset):
        if self._run_async:
            with mp.Pool() as pool:
                tokenized_dataset = SimpleDataset(pool.map(self._tokenize, dataset))

            return tokenized_dataset
        else:
            return SimpleDataset([self._tokenize(item) for item in dataset])
class DataTransformer:
    """Data transformer cuts max length of string, but does not pad it.
    GluonNLP can effectively work with variable sequence length"""

    def __init__(self, word_vocab, max_length=70):
        self._word_vocab = word_vocab
        self._max_length = max_length

    def __call__(self, *items):
        indices = self._word_vocab[items[2][:self._max_length]]

        if len(items) == 4:
            return items[0], indices, items[3]
        else:
            return items[0], indices
from gluonnlp.model import ConvolutionalEncoder
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import Dense, Dropout, HybridSequential, Embedding, Conv2D

class QuoraModel(HybridBlock):
    def __init__(self, word_vocab_size, dropout=0.1, embedding_size=300, prefix=None, params=None):
        super(QuoraModel, self).__init__(prefix=prefix, params=params)
        self._embedding_size = embedding_size

        with self.name_scope():
            self.embedding = Embedding(input_dim=word_vocab_size, output_dim=self._embedding_size)
            self.encoder = ConvolutionalEncoder(embed_size=self._embedding_size,
                                                num_filters=(128, 128, 128, 128,),
                                                ngram_filter_sizes=(1, 2, 3, 5,),
                                                conv_layer_activation='tanh',
                                                num_highway=None,
                                                output_size=None)
            self.output = HybridSequential()

            with self.output.name_scope():
                self.output.add(Dropout(dropout))
                self.output.add(Dense(units=1))

    def hybrid_forward(self, F, data):
        embedded = self.embedding(data)
        embedded = embedded.transpose(axes=(1, 0, 2))
        encoded = self.encoder(embedded)
        result = self.output(encoded)
        return result

import os
import time
import numpy as np
import pandas as pd
import mxnet as mx
from gluonnlp.data import FixedBucketSampler
from gluonnlp.data.batchify import Tuple, Pad, Stack
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet import nd, autograd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def get_args():
    args = SimpleNamespace()
    args.gpu = None
    args.learning_rate = 0.001
    args.epochs = 3
    args.batch_size = 512

    return args


def find_best_f1(outputs, labels):
    tmp = [0, 0, 0]  # idx, cur, max
    threshold = 0

    for tmp[0] in np.arange(0.1, 0.501, 0.01):
        tmp[1] = f1_score(labels.asnumpy(), outputs.asnumpy() > tmp[0])
        if tmp[1] > tmp[2]:
            threshold = tmp[0]
            tmp[2] = tmp[1]

    return tmp[2], threshold


def run_training(net, trainer, train_dataloader, val_dataloader, epochs, context):
    loss_fn = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    print("Start training for {} epochs: {}".format(epochs, time.ctime()))

    for e in range(epochs):
        train_loss = 0.
        total_items = 0
        val_outputs = []
        val_l = []

        for i, (q_idx, (word_data, word_valid_lengths), label) in enumerate(train_dataloader):
            items_per_iteration = word_data.shape[0]
            total_items += items_per_iteration

            word_data = word_data.as_in_context(context)
            label = label.astype('float32').as_in_context(context)

            with autograd.record():
                out = net(word_data)
                loss = loss_fn(out, label)

            loss.backward()
            trainer.step(1)
            train_loss += loss.mean().asscalar()

        for i, (q_idx, (word_data, word_valid_lengths), label) in enumerate(val_dataloader):
            word_data = word_data.as_in_context(context)
            label = label.astype('float32').as_in_context(context)

            out = net(word_data)
            val_outputs.append(out.sigmoid())
            val_l.append(label.reshape(shape=(label.shape[0], 1)))

        val_outputs = mx.nd.concat(*val_outputs, dim=0)
        val_l = mx.nd.concat(*val_l, dim=0)

        val_f1, threshold = find_best_f1(val_outputs, val_l)
        print("Epoch {}. Current Loss: {:.5f}. Val F1: {:.3f}, Val Threshold: {:.3f}, {}"
              .format(e, train_loss / total_items, val_f1, threshold, time.ctime()))

    return net, threshold


def run_evaluation(net, dataloader, threshold, context):
    print("Start predicting")
    outputs = []
    result = []

    for i, (q_idx, (word_data, word_valid_lengths)) in enumerate(dataloader):
        word_data = word_data.as_in_context(context)

        out = net(word_data)
        outputs.append((q_idx, out.sigmoid() > threshold))

    for batch in outputs:
        result.extend([(int(q_idx.asscalar()), int(pred.asscalar()))
                       for q_idx, pred in zip(batch[0], batch[1])])

    return result


def load_and_process_dataset(dataset, word_vocab, path=None, async=True):
    tokenizer = DataTokenizer(async)
    transformer = DataTransformer(word_vocab)
    lazy_trasform = True if not path else False
    processed_dataset = tokenizer(dataset).transform(transformer, lazy=lazy_trasform)

    return processed_dataset


def load_vocab(dataset):
    tokenizer = DataTokenizer()
    tokenized_dataset = tokenizer(dataset)
    vocab_provider = VocabProvider([tokenized_dataset])
    word_vocab = vocab_provider.get_vocab()

    return word_vocab

args = get_args()
context = mx.cpu(0) if args.gpu is None else mx.gpu(args.gpu)

print('Script started. {}'.format(time.ctime()))
dataset = QuoraDataset('train')
word_vocab = load_vocab(dataset)
glove = CustomEmbedding('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', 300)
word_vocab.set_embedding(glove)

model = QuoraModel(len(word_vocab))
model.initialize(mx.init.Xavier(magnitude=2.24), ctx=context)
model.embedding.weight.set_data(word_vocab.embedding.idx_to_vec)
model.embedding.collect_params().setattr('grad_req', 'null')
model.hybridize(static_alloc=True)

train_indices, dev_indices = train_test_split(range(len(dataset)), train_size=0.9)

train_dataset = load_and_process_dataset(get_sub_segment_from_list(dataset, train_indices),
                                         word_vocab)

dev_dataset = load_and_process_dataset(get_sub_segment_from_list(dataset, dev_indices),
                                       word_vocab)

batchify_fn = Tuple(Stack(),
                    Pad(axis=0, pad_val=word_vocab[word_vocab.padding_token], ret_length=True),
                    Stack())

train_sampler = FixedBucketSampler(lengths=[len(item[1]) for item in train_dataset],
                                   batch_size=args.batch_size,
                                   shuffle=True)

dev_sampler = FixedBucketSampler(lengths=[len(item[1]) for item in dev_dataset],
                                 batch_size=args.batch_size,
                                 shuffle=False)

train_dataloader = DataLoader(train_dataset,
                              batchify_fn=batchify_fn,
                              batch_sampler=train_sampler,
                              num_workers=10)

dev_dataloader = DataLoader(dev_dataset,
                            batchify_fn=batchify_fn,
                            batch_sampler=dev_sampler,
                            num_workers=10)

trainer = Trainer(model.collect_params(), 'adam', {'learning_rate': args.learning_rate})
best_model, best_val_threshold = run_training(model, trainer, train_dataloader,
                                              dev_dataloader, args.epochs, context)


test_dataset = QuoraDataset('test')
processed_test_dataset = load_and_process_dataset(test_dataset, word_vocab)
test_sampler = FixedBucketSampler(lengths=[len(item[1]) for item in processed_test_dataset],
                                  batch_size=args.batch_size)

batchify_test_fn = Tuple(Stack(),
                         Pad(axis=0, pad_val=word_vocab[word_vocab.padding_token],
                             ret_length=True))

test_dataloader = DataLoader(processed_test_dataset,
                             batchify_fn=batchify_test_fn,
                             batch_sampler=test_sampler,
                             num_workers=10,
                             shuffle=False)
predictions = run_evaluation(model, test_dataloader, best_val_threshold, context)

submission = pd.DataFrame()
mapping = {item[0]: item[1] for item in test_dataset}
submission['qid'] = [mapping[p[0]] for p in predictions]
submission['prediction'] = [p[1] for p in predictions]

# some magic to restore the order, in case Kaggle is sensetive to order of items
sample = pd.read_csv('../input/sample_submission.csv')
joined = sample.join(submission.set_index('qid'), on='qid', lsuffix='_sample', rsuffix='_real')

joined = joined.drop(['prediction_sample'], axis=1)
joined = joined.rename(index=str, columns={'prediction_real': 'prediction'})
joined.to_csv("submission.csv", index=False)