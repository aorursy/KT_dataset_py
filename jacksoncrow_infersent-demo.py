%matplotlib inline



from random import randint

import numpy as np

import torch

import shutil

import string

import nltk.data

import matplotlib



matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
# here we need to restructure working directory, so that script imports working properly

shutil.copytree("/kaggle/input/infersent/", "/kaggle/working/infersent")

! mv /kaggle/working/infersent/* /kaggle/working/
%%time



# TODO: add encoder to dataset as well

# If this cell freezes, probably you haven't enabled Internet access for the notebook

! mkdir encoder

! curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
model_version = 1

MODEL_PATH = "encoder/infersent%s.pkl" % model_version

W2V_PATH = '/kaggle/input/glove-840b-300d/glove.840B.300d.txt'

VOCAB_SIZE = 1e5  # Load embeddings of VOCAB_SIZE most frequent words

USE_CUDA = False  # Keep it on CPU if False, otherwise will put it on GPU
from models import InferSent

params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,

                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}

model = InferSent(params_model)

model.load_state_dict(torch.load(MODEL_PATH))
%%time

model = model.cuda() if USE_CUDA else model



model.set_w2v_path(W2V_PATH)



model.build_vocab_k_words(K=VOCAB_SIZE)
sentences = ['Everyone really likes the newest benefits',

 'The Government Executive articles housed on the website are not able to be searched .',

 'I like him for the most part , but would still enjoy seeing someone beat him .',

 'My favorite restaurants are always at least a hundred miles away from my house .',

 'What a day !',

 'What color is it ?',

 'I know exactly .']
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



def format_text(text):

    global tokenizer

    padded_text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))

    return tokenizer.tokenize(padded_text)



text = 'Everyone really likes the newest benefits. The Government Executive articles housed on the website are not able to be searched.'\

'I like him for the most part, but would still enjoy seeing someone beat him. My favorite restaurants are always at least a hundred '\

'miles away from my house. What a day! What color is it? I know exactly.'



sentences = format_text(text)

sentences
embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)

print('nb sentences encoded : {0}'.format(len(embeddings)))
np.linalg.norm(model.encode(['the cat eats.']))
def cosine(u, v):

    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0])
idx = randint(0, len(sentences) - 1)

_, _ = model.visualize(sentences[idx])
_, _ = model.visualize('The cat is drinking milk.')
%%time

model.build_vocab_k_words(5e5) # getting 500K words vocab

_, _ = model.visualize("barack-obama is the former president of the United-States.")