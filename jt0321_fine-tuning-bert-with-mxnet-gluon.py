import warnings

warnings.filterwarnings(action='once')



import os

from pathlib import Path

import random

import multiprocessing



import numpy as np

import pandas as pd



from tqdm import tqdm_notebook, tqdm



import mxnet as mx

from mxnet import gluon

from mxnet.gluon import Block

from mxnet.gluon import nn

import gluonnlp as nlp

from gluonnlp.model import get_bert_model

from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
path = Path('../input/jigsaw-unintended-bias-in-toxicity-classification')

os.listdir(path)
train_csv = path/'train.csv'

test_csv = path/'test.csv'

sample_csv = path/'sample_submission.csv'
test_size=90000
train_df = pd.read_csv(train_csv)

train_df = train_df[:test_size]
test_df = pd.read_csv(test_csv)

test_df.head()
sample_df = pd.read_csv(sample_csv)

sample_df.head()
class BERTClassifier(Block):

    """Model for sentence (pair) classification task with BERT.



    The model feeds token ids and token type ids into BERT to get the

    pooled BERT sequence representation, then apply a Dense layer for

    classification.



    Parameters

    ----------

    bert: BERTModel

        Bidirectional encoder with transformer.

    num_classes : int, default is 2

        The number of target classes.

    dropout : float or None, default 0.0.

        Dropout probability for the bert output.

    prefix : str or None

        See document of `mx.gluon.Block`.

    params : ParameterDict or None

        See document of `mx.gluon.Block`.

    """



    def __init__(self,

                 bert,

                 num_classes=2,

                 dropout=0.0,

                 prefix=None,

                 params=None):

        super(BERTClassifier, self).__init__(prefix=prefix, params=params)

        self.bert = bert

        with self.name_scope():

            self.classifier = nn.HybridSequential(prefix=prefix)

            if dropout:

                self.classifier.add(nn.Dropout(rate=dropout))

            self.classifier.add(nn.Dense(units=num_classes))



    def forward(self, inputs, token_types, valid_length=None):  # pylint: disable=arguments-differ

        """Generate the unnormalized score for the given the input sequences.



        Parameters

        ----------

        inputs : NDArray, shape (batch_size, seq_length)

            Input words for the sequences.

        token_types : NDArray, shape (batch_size, seq_length)

            Token types for the sequences, used to indicate whether the word belongs to the

            first sentence or the second one.

        valid_length : NDArray or None, shape (batch_size)

            Valid length of the sequence. This is used to mask the padded tokens.



        Returns

        -------

        outputs : NDArray

            Shape (batch_size, num_classes)

        """

        _, pooler_out = self.bert(inputs, token_types, valid_length)

        return self.classifier(pooler_out)
class BERTDatasetTransform(object):

    """Dataset Transformation for BERT-style Sentence Classification or Regression.



    Parameters

    ----------

    tokenizer : BERTTokenizer.

        Tokenizer for the sentences.

    max_seq_length : int.

        Maximum sequence length of the sentences.

    labels : list of int , float or None. defaults None

        List of all label ids for the classification task and regressing task.

        If labels is None, the default task is regression

    pad : bool, default True

        Whether to pad the sentences to maximum length.

    pair : bool, default True

        Whether to transform sentences or sentence pairs.

    label_dtype: int32 or float32, default float32

        label_dtype = int32 for classification task

        label_dtype = float32 for regression task

    """



    def __init__(self,

                 tokenizer,

                 max_seq_length,

                 class_labels=None,

                 pad=True,

                 pair=True,

                 has_label=True):

        self.class_labels = class_labels

        self.has_label = has_label

        self._label_dtype = 'int32' if class_labels else 'float32'

        if has_label and class_labels:

            self._label_map = {}

            for (i, label) in enumerate(class_labels):

                self._label_map[label] = i

        self._bert_xform = BERTSentenceTransform(

            tokenizer, max_seq_length, pad=pad, pair=pair)



    def __call__(self, line):

        """Perform transformation for sequence pairs or single sequences.



        The transformation is processed in the following steps:

        - tokenize the input sequences

        - insert [CLS], [SEP] as necessary

        - generate type ids to indicate whether a token belongs to the first

          sequence or the second sequence.

        - generate valid length



        For sequence pairs, the input is a tuple of 3 strings:

        text_a, text_b and label.



        Inputs:

            text_a: 'is this jacksonville ?'

            text_b: 'no it is not'

            label: '0'

        Tokenization:

            text_a: 'is this jack ##son ##ville ?'

            text_b: 'no it is not .'

        Processed:

            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'

            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1

            valid_length: 14

            label: 0



        For single sequences, the input is a tuple of 2 strings: text_a and label.

        Inputs:

            text_a: 'the dog is hairy .'

            label: '1'

        Tokenization:

            text_a: 'the dog is hairy .'

        Processed:

            text_a:  '[CLS] the dog is hairy . [SEP]'

            type_ids: 0     0   0   0  0     0 0

            valid_length: 7

            label: 1



        Parameters

        ----------

        line: tuple of str

            Input strings. For sequence pairs, the input is a tuple of 3 strings:

            (text_a, text_b, label). For single sequences, the input is a tuple

            of 2 strings: (text_a, label).



        Returns

        -------

        np.array: input token ids in 'int32', shape (batch_size, seq_length)

        np.array: valid length in 'int32', shape (batch_size,)

        np.array: input token type ids in 'int32', shape (batch_size, seq_length)

        np.array: classification task: label id in 'int32', shape (batch_size, 1),

            regression task: label in 'float32', shape (batch_size, 1)

        """

        if self.has_label:

            input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])

            label = line[-1]

            # map to int if class labels are available

            if self.class_labels:

                label = self._label_map[label]

            label = np.array([label], dtype=self._label_dtype)

            return input_ids, valid_length, segment_ids, label

        else:

            return self._bert_xform(line)

ctx = mx.gpu(0)
max_len = 128

pad = True

pair = False



epochs = 1

batch_size = 32

#optimizer='bertadam'

class_labels=[0,1]

lr = 1e-6

epsilon = 1e-06

warmup_ratio = 0.1

log_interval = 1000

accumulate = None
bert_base, vocabulary = nlp.model.get_model('bert_12_768_12', 

                                             dataset_name='book_corpus_wiki_en_uncased',

                                             pretrained=True, ctx=ctx, use_pooler=True,

                                             use_decoder=False, use_classifier=False,

                                             root='../input/bertmx/bert-mx')
model = BERTClassifier(bert_base, num_classes=2, dropout=0.1)

# only need to initialize the classifier layer.

model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)

model.hybridize(static_alloc=True)



# softmax cross entropy loss for classification

loss_function = gluon.loss.SoftmaxCELoss()

loss_function.hybridize(static_alloc=True)



metric = mx.metric.Accuracy()
tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
train_df['comment_text'] = train_df['comment_text'].astype(str)

train_df['target']=(train_df['target']>=0.5).astype(int)
train_data_raw = train_df[['comment_text', 'target']].values
pool = multiprocessing.Pool()



# transformation for data train and dev

#label_dtype = 'float32' if not task.class_labels else 'int32'

label_dtype='int32'

trans = BERTDatasetTransform(tokenizer, max_len,

                             class_labels=class_labels,

                             pad=pad, pair=pair,

                             has_label=True)



# data train

# task.dataset_train returns (segment_name, dataset)

#train_tsv = task.dataset_train()[1]

#data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_data_raw))

data_train = mx.gluon.data.SimpleDataset(train_data_raw)

data_train = data_train.transform(trans)

data_train_len = data_train.transform(

    lambda input_id, length, segment_id, label_id: length, lazy=False)

# bucket sampler for training

batchify_fn = nlp.data.batchify.Tuple(

    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),

    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(label_dtype))

batch_sampler = nlp.data.sampler.FixedBucketSampler(

    data_train_len,

    batch_size=batch_size,

    #num_buckets=10,

    ratio=0,

    shuffle=True)

# data loader for training

train_data = gluon.data.DataLoader(

    dataset=data_train,

    num_workers=4,

    batch_sampler=batch_sampler,

    batchify_fn=batchify_fn)

num_train_examples = len(data_train)
print('vocabulary used for tokenization = \n%s'%vocabulary)

print('[PAD] token id = %s'%(vocabulary['[PAD]']))

print('[CLS] token id = %s'%(vocabulary['[CLS]']))

print('[SEP] token id = %s'%(vocabulary['[SEP]']))

print('token ids = \n%s'%data_train[4][0])

print('valid length = \n%s'%data_train[4][1])

print('segment ids = \n%s'%data_train[4][2])

print('label = \n%s'%data_train[4][3])
optimizer = mx.optimizer.create('adam', multi_precision=True, learning_rate=lr, epsilon=epsilon)
all_model_params = model.collect_params()

optimizer_params = {'learning_rate': lr, 'epsilon': epsilon, 'wd': 0.01}



try:

    trainer = gluon.Trainer(all_model_params, optimizer,

                            update_on_kvstore=False)

except ValueError as e:

    print(e)

    warnings.warn(

        'AdamW optimizer is not found. Please consider upgrading to '

        'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')

    trainer = gluon.Trainer(all_model_params, 'adam',

                            optimizer_params, update_on_kvstore=False)
model.load_parameters('../input/mxbert06151/j-20190616k', ctx=ctx)
step_size = batch_size * accumulate if accumulate else batch_size

num_train_steps = int(num_train_examples / step_size * epochs)

num_warmup_steps = int(num_train_steps * warmup_ratio)

step_num = 0



# Do not apply weight decay on LayerNorm and bias terms

for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():

    v.wd_mult = 0.0

# Collect differentiable parameters

params = [p for p in all_model_params.values() if p.grad_req != 'null']



# Set grad_req if gradient accumulation is required

if accumulate:

    for p in params:

        p.grad_req = 'add'



#tic = time.time()

for epoch_id in range(epochs):

    metric.reset()

    step_loss = 0

    #tic = time.time()

    all_model_params.zero_grad()



    for batch_id, seqs in enumerate(train_data):

        # learning rate schedule

        if step_num < num_warmup_steps:

            new_lr = lr * step_num / num_warmup_steps

        else:

            non_warmup_steps = step_num - num_warmup_steps

            offset = non_warmup_steps / (num_train_steps - num_warmup_steps)

            new_lr = lr - offset * lr

        optimizer.set_learning_rate(new_lr)



        # forward and backward

        with mx.autograd.record():

            input_ids, valid_length, type_ids, label = seqs

            out = model(

                input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),

                valid_length.astype('float32').as_in_context(ctx))

            ls = loss_function(out, label.as_in_context(ctx)).mean()

        ls.backward()



        # update

        if not accumulate or (batch_id + 1) % accumulate == 0:

            trainer.allreduce_grads()

            nlp.utils.clip_grad_global_norm(params, 1)

            trainer.update(accumulate if accumulate else 1)

            # set grad to zero for gradient accumulation

            all_model_params.zero_grad()

            step_num += 1



        step_loss += ls.asscalar()

        metric.update([label], [out])

        if (batch_id + 1) % (log_interval) == 0:

            #log_train(batch_id, len(train_data), metric, step_loss, log_interval,

                      #epoch_id, trainer.learning_rate)

            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'

                  .format(epoch_id, batch_id + 1, len(train_data),

                 step_loss / log_interval,

                 optimizer.learning_rate, metric.get()[1]))

            step_loss = 0

    mx.nd.waitall()
model.save_parameters('j-20190616k')
test_df['comment_text'] = test_df['comment_text'].astype(str)

test_data_raw = test_df[['comment_text']].values
# batchify for data test

test_batchify_fn = nlp.data.batchify.Tuple(

    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),

    nlp.data.batchify.Pad(axis=0))



# transform for data test

test_trans = BERTDatasetTransform(tokenizer, max_len,

                                  class_labels=None,

                                  pad=pad, pair=pair,

                                  has_label=False)



data_test = mx.gluon.data.SimpleDataset(pool.map(test_trans, test_data_raw))

loader_test = mx.gluon.data.DataLoader(

    data_test,

    batch_size=batch_size,

    num_workers=4,

    shuffle=False,

    batchify_fn=test_batchify_fn)
data_test[0]
#value_list = []

#index_list = []

results = []

for _, seqs in enumerate(loader_test):

    input_ids, valid_length, type_ids = seqs

    out = model(input_ids.as_in_context(ctx),

                type_ids.as_in_context(ctx),

                valid_length.astype('float32').as_in_context(ctx))

    results.extend([o for o in out.asnumpy()])

    #values, indices = mx.nd.topk(out, k=1, ret_typ='both')

    #value_list.extend(values.asnumpy().reshape(-1).tolist())

    #index_list.extend(indices.asnumpy().reshape(-1).tolist())



mx.nd.waitall()



results[24]
predictions = [mx.nd.array(result).softmax().asnumpy()[1] for result in results]
submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': predictions

})
submission.to_csv('submission.csv', index=False)