# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!git clone https://github.com/kamalkraj/minGPT-TF.git
!cd ./minGPT-TF
!pwd 
# ./
import os
os.chdir('./minGPT-TF')
! pip install fastprogress==0.2.3
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
import math
import numpy as np
import tensorflow as tf
from mingpt.model import GPT, GPTConfig
class CharDataset:

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __iter__(self):
        # we're actually going to "cheat" and pick a spot in the dataset at 
        for _ in range(self.__len__()):
            i = np.random.randint(0, len(self.data) - (self.block_size + 1))
            chunk = self.data[i:i+self.block_size+1]
            dix = [self.stoi[s] for s in chunk]
            x = tf.convert_to_tensor(dix[:-1], dtype=tf.int32)
            y = tf.convert_to_tensor(dix[1:], dtype=tf.int32)
            yield x, y
    
    __call__ = __iter__
block_size = 128 
# text = open('input.txt', 'r').read()
# train_dataset_gen = CharDataset(text, block_size) 
# text
# train_dataset = tf.data.Dataset.from_generator(train_dataset_gen,(tf.int32,tf.int32))
# from mingpt.model import GPT, GPTConfig
# mconf = GPTConfig(train_dataset_gen.vocab_size, train_dataset_gen.block_size,
#                   n_layer=8, n_head=8, n_embd=512)
# from mingpt.trainer import Trainer, TrainerConfig

# # initialize a trainer instance and kick off training
# tconf = TrainerConfig(max_epochs=10, batch_size=128, learning_rate=6e-4,
#                       lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(train_dataset_gen)*block_size,
#                       num_workers=4)
# trainer = Trainer(GPT, mconf, train_dataset, len(train_dataset_gen), None, None, tconf)
# trainer.train()
# # alright, let's sample some character-level
# from mingpt.utils import sample

# context = "O God, O God!"
# x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
# y = sample(trainer.model, x, 2000, temperature=0.9, sample=True, top_k=5)[0]
# completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
# print(completion)
# os.chdir('/kaggle/working')
import pandas as pd
txt = pd.read_csv('../input/short-jokes/shortjokes.csv')
txt.head()
txt1= []

for i in txt['Joke']:
    txt1.append(i)
txt2 = str(txt1)

type(txt2)
import os
os.chdir('./minGPT-TF')
train_dataset_gen = CharDataset(txt2, block_size) 
train_dataset = tf.data.Dataset.from_generator(train_dataset_gen,(tf.int32,tf.int32))
from mingpt.model import GPT, GPTConfig
mconf = GPTConfig(train_dataset_gen.vocab_size, train_dataset_gen.block_size,
                  n_layer=8, n_head=8, n_embd=512)
from mingpt.trainer import Trainer, TrainerConfig


tconf = TrainerConfig(max_epochs=10, batch_size=128, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(train_dataset_gen)*block_size,
                      num_workers=4)
trainer = Trainer(GPT, mconf, train_dataset, len(train_dataset_gen), None, None, tconf)
trainer.train()

from mingpt.utils import sample

context = "love"
x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
y = sample(trainer.model, x, 500, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
print(completion)

from mingpt.utils import sample

context = "Tanul is crazy"
x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
y = sample(trainer.model, x, 100, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
print(completion)

from mingpt.utils import sample

context = "There was no one"
x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
y = sample(trainer.model, x, 90, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
print(completion)

from mingpt.utils import sample

context = "How do you feel when you lie to me"
x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
y = sample(trainer.model, x, 64, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
print(completion)

from mingpt.utils import sample

context = "You are a Dick "
x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
y = sample(trainer.model, x, 64, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
print(completion)

from mingpt.utils import sample

context = "That cat "
x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
y = sample(trainer.model, x, 64, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
print(completion)

from mingpt.utils import sample

context = "This is fucking insane"
x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
y = sample(trainer.model, x, 64, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
print(completion)

from mingpt.utils import sample

context = "I can crack jokes do you want "
x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None,...]
y = sample(trainer.model, x, 100, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
print(completion)
