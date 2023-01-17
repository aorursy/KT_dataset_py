!pip install -q -U trax
import trax # Our Main Library

from trax import layers as tl

import os # For os dependent functionalities

import numpy as np # For scientific computing

import pandas as pd # For basic data analysis

import random as rnd # For using random functions
data = pd.read_csv("/kaggle/input/entity-annotated-corpus/ner_dataset.csv",encoding = 'ISO-8859-1')

data = data.fillna(method = 'ffill')

data.head()
## Extract the 'Word' column from the dataframe

words = data.loc[:, "Word"]



## Convert into a text file using the .savetxt() function

np.savetxt(r'words.txt', words.values, fmt="%s")
vocab = {}

with open('words.txt') as f:

  for i, l in enumerate(f.read().splitlines()):

    vocab[l] = i

  print("Number of words:", len(vocab))

  vocab['<PAD>'] = len(vocab)
class Get_sentence(object):

    def __init__(self,data):

        self.n_sent=1

        self.data = data

        agg_func = lambda s:[(w,p,t) for w,p,t in zip(s["Word"].values.tolist(),

                                                     s["POS"].values.tolist(),

                                                     s["Tag"].values.tolist())]

        self.grouped = self.data.groupby("Sentence #").apply(agg_func)

        self.sentences = [s for s in self.grouped]
getter = Get_sentence(data)

sentence = getter.sentences
words = list(set(data["Word"].values))

words_tag = list(set(data["Tag"].values))



word_idx = {w : i+1 for i ,w in enumerate(words)}

tag_idx =  {t : i for i ,t in enumerate(words_tag)}
X = [[word_idx[w[0]] for w in s] for s in sentence]

y = [[tag_idx[w[2]] for w in s] for s in sentence]
def data_generator(batch_size, x, y,pad, shuffle=False, verbose=False):



    num_lines = len(x)

    lines_index = [*range(num_lines)]

    if shuffle:

        rnd.shuffle(lines_index)

    

    index = 0 

    while True:

        buffer_x = [0] * batch_size 

        buffer_y = [0] * batch_size 



        max_len = 0

        for i in range(batch_size):

            if index >= num_lines:

                index = 0

                if shuffle:

                    rnd.shuffle(lines_index)

            

            buffer_x[i] = x[lines_index[index]]

            buffer_y[i] = y[lines_index[index]]

            

            lenx = len(x[lines_index[index]])    

            if lenx > max_len:

                max_len = lenx                  

            

            index += 1



        X = np.full((batch_size, max_len), pad)

        Y = np.full((batch_size, max_len), pad)





        for i in range(batch_size):

            x_i = buffer_x[i]

            y_i = buffer_y[i]



            for j in range(len(x_i)):



                X[i, j] = x_i[j]

                Y[i, j] = y_i[j]



        if verbose: print("index=", index)

        yield((X,Y))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state=1)
def NERmodel(tags, vocab_size=35181, d_model = 50):



  model = tl.Serial(

    # tl.Embedding(vocab_size, d_model),

    trax.models.reformer.Reformer(vocab_size, d_model, ff_activation=tl.LogSoftmax),

    tl.Dense(tags),

    tl.LogSoftmax()

  )



  return model
model = NERmodel(tags = 17)



print(model)
from trax.supervised import training



rnd.seed(33)



batch_size = 64



train_generator = trax.data.inputs.add_loss_weights(

    data_generator(batch_size, x_train, y_train,vocab['<PAD>'], True),

    id_to_mask=vocab['<PAD>'])



eval_generator = trax.data.inputs.add_loss_weights(

    data_generator(batch_size, x_test, y_test,vocab['<PAD>'] ,True),

    id_to_mask=vocab['<PAD>'])
def train_model(model, train_generator, eval_generator, train_steps=1, output_dir='model'):

    train_task = training.TrainTask(

      train_generator,  

      loss_layer = tl.CrossEntropyLoss(), 

      optimizer = trax.optimizers.Adam(0.01), 

      n_steps_per_checkpoint=10

    )



    eval_task = training.EvalTask(

      labeled_data = eval_generator, 

      metrics = [tl.CrossEntropyLoss(), tl.Accuracy()], 

      n_eval_batches = 10 

    )



    training_loop = training.Loop(

        model, 

        train_task, 

        eval_tasks = eval_task, 

        output_dir = output_dir) 



    training_loop.run(n_steps = train_steps)

    return training_loop
train_steps = 100

training_loop = train_model(model, train_generator, eval_generator, train_steps)