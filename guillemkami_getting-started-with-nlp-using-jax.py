!nvcc --version

!python --version
!pip install --upgrade "https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.39-cp36-none-linux_x86_64.whl"

!pip install --upgrade jax 
import jax

import jax.numpy as np



key = jax.random.PRNGKey(0)



print('JAX is running on', jax.lib.xla_bridge.get_backend().platform)
!ls ../input/nlp-getting-started
import pandas as pd



df = pd.read_csv('../input/nlp-getting-started/train.csv', encoding='utf-8')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv', encoding='utf-8')



df.head()
import re

import string



def clean_tweet(tweet: str) -> str:

    url = re.compile(r'https?://\S+|www\.\S+')

    tweet = url.sub(r'',tweet)

    

    html = re.compile(r'<.*?>')

    tweet = html.sub(r'', tweet)

    

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    tweet = emoji_pattern.sub(r'', tweet)

    

    tweet = re.sub('([.,!?()#])', r' \1 ', tweet)

    tweet = re.sub('\s{2,}', ' ', tweet)



    return tweet.lower()



df['text'] = df['text'].apply(clean_tweet)

test_df['text'] = test_df['text'].apply(clean_tweet)



df.head()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



MAX_SEQUENCE_LENGTH = 64



tokenizer = Tokenizer(num_words=None, filters='')

tokenizer.fit_on_texts(df.text.tolist() + test_df.text.tolist())



train_sequences = tokenizer.texts_to_sequences(df.text.tolist())

train_sequences = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)



test_sequences = tokenizer.texts_to_sequences(test_df.text.tolist())

test_sequences = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)



word2idx = tokenizer.word_index
list(word2idx.items())[:10]
from typing import Any, Callable, Container, NamedTuple



Parameter = Container[np.ndarray]

ForwardFn = Callable[[Parameter, np.ndarray, Any], np.ndarray]



class JaxModule(NamedTuple):

    # Dict or list contining the layer parameters

    parameters: Parameter

    

    # How we operate with parameters to generate an output

    forward_fn: ForwardFn

    

    def update(self, new_parameters) -> 'JaxModule':

        # As tuples are immutable, we create a new jax module keeping the

        # forward_fn but with new parameters

        return JaxModule(new_parameters, self.forward_fn)

    

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:

        # Automatically injects parameters

        return self.forward_fn(self.parameters, x, **kwargs)
def linear(key: np.ndarray, 

           in_features: int, 

           out_features: int, 

           activation = lambda x: x) -> 'JaxModule':

    x_init = jax.nn.initializers.xavier_normal()

    u_init = jax.nn.initializers.uniform()



    key, W_key, b_key = jax.random.split(key, 3)

    W = x_init(W_key, shape=(in_features, out_features))

    b = u_init(b_key, shape=(out_features,))

    params = dict(W=W, bias=b)

    

    def forward_fn(params: Parameter, x: np.ndarray, **kwargs):

        return np.dot(x, params['W']) + params['bias']

    

    return JaxModule(parameters=params, forward_fn=forward_fn)



def flatten(key: np.ndarray) -> 'JaxModule':

    # Reshapes the data to have 2 dims [BATCH, FEATURES]

    def forward_fn(params, x, **kwargs):

        bs = x.shape[0]

        return x.reshape(bs, -1)

    return JaxModule({}, forward_fn)





key, subkey, linear_key = jax.random.split(key, 3)



# Input vector of shape [BATCH, FEATURES]

mock_in = jax.random.uniform(subkey, shape=(10, 32))

linear_layer = linear(linear_key, 32, 512)



print('Input shape:', mock_in.shape, 

      'Output shape after linear:', linear_layer(mock_in).shape)
def conv_1d(key: np.ndarray, 

            in_features: int, 

            out_features: int,

            kernel_size: int,

            strides: int = 1,

            padding: str = 'SAME',

            activation = lambda x: x) -> 'JaxModule':

    

    # [KERNEL_WIDTH, IN_FEATURES, OUT_FEATURES]

    kernel_shape = (kernel_size, in_features, out_features)

    # [BATCH, WIDTH, IN_FEATURES]

    seq_shape = (None, None, in_features)

    

    # Declare convolutional specs

    dn = jax.lax.conv_dimension_numbers(

        seq_shape, kernel_shape, ('NWC', 'WIO', 'NWC'))

    

    key, k_key, b_key = jax.random.split(key, 3)

    

    kernel = jax.nn.initializers.glorot_normal()(k_key, shape=kernel_shape)

    b = jax.nn.initializers.uniform()(b_key, shape=(out_features,))

    params = dict(kernel=kernel, bias=b)

    

    def forward_fn(params: Parameter, x: np.ndarray, **kwargs):

        return activation(jax.lax.conv_general_dilated(

            x, params['kernel'], 

            (strides,), padding, 

            (1,), (1,), dimension_numbers=dn) + params['bias'])



    return JaxModule(params, forward_fn)





key, subkey, conv_key = jax.random.split(key, 3)



# Input vector of shape [BATCH, FEATURES]

mock_in = jax.random.uniform(subkey, shape=(10, 128, 32))

conv = conv_1d(conv_key, 32, 512, kernel_size=3, strides=2)



print('Input shape:', mock_in.shape, 

      'Output shape after 1D Convolution:', conv(mock_in).shape)
!ls ../input
import numpy as onp # Use onp to avoid overheat of moving each embedding to GPU



embeddings_index = {}

embedding_matrix = onp.zeros((len(word2idx) + 1, 50))



f = open('../input/glove-twitter-27b-50d/glove.twitter.27B.50d.txt')



for line in f:

    values = line.split()

    word = values[0]

    coefs = onp.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



for word, i in word2idx.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector



embeddings = jax.device_put(embedding_matrix)

print('Embedding for "hello" is:', embeddings[word2idx['hello']])
def embed_sequence(sequence: np.ndarray) -> np.ndarray:

    return embeddings[sequence.astype('int32')]



key, subkey, embedding_key = jax.random.split(key, 3)



# Input vector of shape [BATCH, FEATURES]

mock_in = jax.random.randint(subkey, 

                             minval=0,

                             maxval=512,

                             shape=(10, 128))



print('Input shape:', mock_in.shape, 

      'Output shape after Embeddings:', embed_sequence(mock_in).shape)
from typing import Sequence

from functools import partial



# Partially evaluated layers without the random key

PartialLayer = Callable[[np.ndarray], JaxModule]



def sequential(key: np.ndarray, *modules: Sequence[PartialLayer]) -> JaxModule:

    key, *subkeys = jax.random.split(key, len(modules) + 1)

    model = [m(k) for k, m in zip(subkeys, modules)]

    

    def forward_fn(params, x, **kwargs):

        for m, p in zip(model, params):

            x = m.forward_fn(p, x, **kwargs)

        return x

    

    return JaxModule([m.parameters for m in model], forward_fn)



mock_model = sequential(

    key,

    partial(conv_1d, 

            in_features=50, out_features=256, 

            kernel_size=5, strides=2, activation=jax.nn.relu),

    flatten,

    partial(linear, 

            in_features=16384, out_features=128, 

            activation=jax.nn.relu),

    partial(linear, 

            in_features=128, out_features=1, 

            activation=jax.nn.sigmoid))



embedded_in = embed_sequence(mock_in)



print('Input shape:', mock_in.shape, 

      'Output shape after model:', mock_model(embedded_in).shape)
def bce(y_hat: np.ndarray, y: np.ndarray) -> float:

    y_hat = y_hat.reshape(-1)

    y = y.reshape(-1)

    y_hat = np.clip(y_hat, 1e-6, 1 - 1e-6)

    pt = np.where(y == 1, y_hat, 1 - y_hat)

    loss = -np.log(pt)

    return np.mean(loss)





def create_backward(model: JaxModule):

    # Backward is just a function that receives the model parameters and combines them

    # using the forward_fn. This will allow as to compute the partial derivate of the 

    # weights with respect to the loss

    def backward(params: Sequence[Parameter], 

                 x: np.ndarray, y: np.ndarray) -> float:

        y_hat = model.forward_fn(params, x)

        return bce(y_hat, y)

    return backward



# Compile and differentiate the function

backward_fn = jax.jit(jax.value_and_grad(create_backward(mock_model)))
LEARNING_RATE = 1e-4



def optimizer_step(params: Sequence[Parameter], 

                   gradients: Sequence[Parameter]) -> Sequence[Parameter]:

    def optim_single(param, grad):

        for p in param:

            param[p] = param[p] - LEARNING_RATE * grad[p]

        return param



    return [optim_single(p, g) for p, g in zip(params, gradients)]
# Mock labels

y_trues = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0])



loss, grads = backward_fn(mock_model.parameters, embedded_in, y_trues)

print('Loss:', loss)



# Update the parameters with the obtained gradients

new_params = optimizer_step(mock_model.parameters, grads)

model = mock_model.update(new_params)



loss, grads = backward_fn(mock_model.parameters, embedded_in, y_trues)

print('Loss after one step:', loss)
from sklearn.model_selection import train_test_split



x = train_sequences

y = df.target.values



x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=.9)



x_train = jax.device_put(x_train)

x_val = jax.device_put(x_val)

y_train = jax.device_put(y_train)

y_val = jax.device_put(y_val)
model = sequential(

    key,

    partial(conv_1d, in_features=50, out_features=128,  kernel_size=7),

    partial(conv_1d, in_features=128, out_features=128, kernel_size=7, strides=2, activation=jax.nn.relu),

    

    partial(conv_1d, in_features=128, out_features=256, kernel_size=5),

    partial(conv_1d, in_features=256, out_features=256, kernel_size=5, strides=2, activation=jax.nn.relu),

    

    flatten,

    partial(linear, in_features=4096, out_features=128, activation=jax.nn.relu),

    partial(linear, in_features=128, out_features=1, activation=jax.nn.sigmoid))



backward_fn = jax.jit(jax.value_and_grad(create_backward(model)))
from sklearn.metrics import accuracy_score, f1_score, recall_score



EPOCHS = 20

BATCH_SIZE = 32

train_steps = x_train.shape[0] // BATCH_SIZE



for epoch in range(EPOCHS):

    

    running_loss = 0.0

    

    for step in range(train_steps):        

        key, subkey = jax.random.split(key)

        batch_idx = jax.random.randint(subkey, 

                                       minval=0, 

                                       maxval=x_train.shape[0],

                                       shape=(BATCH_SIZE,))

        

        x_batch = embed_sequence(x_train[batch_idx])

        y_batch = y_train[batch_idx]

        

        loss, grads = backward_fn(model.parameters, x_batch, y_batch)

        running_loss += loss

        model = model.update(optimizer_step(model.parameters, grads))

        

    loss_mean = running_loss / float(step)

    print(f'Epoch[{epoch}] loss: {loss_mean:.4f}')

        

    predictions = model(embed_sequence(x_val))

    loss = bce(predictions, y_val)

    predictions = predictions > .5

    

    print('-- Validation --')

    print('Loss: {}'.format(loss))

    print('Accuracy: {}'.format(accuracy_score(y_val, predictions)))

    print('Recall: {}'.format(recall_score(y_val, predictions)))

    print('F1-Score: {}'.format(f1_score(y_val, predictions)))
ids = test_df.id

targets = (model(embed_sequence(test_sequences)) > .4).reshape(-1)

pd.DataFrame(dict(id=ids, target=targets.astype('int32'))).to_csv('submission.csv', index=False)