!pip install trax
import pandas as pd 

import os

import trax

import trax.fastmath.numpy as np

import random as rnd

from trax import fastmath

from trax import layers as tl
directories = os.listdir('/kaggle/input/')

lines = []

for directory in directories:

    for filename in os.listdir(os.path.join('/kaggle/input',directory)):

        if filename.endswith(".txt"):

            with open(os.path.join(os.path.join('/kaggle/input',directory), filename)) as files:

                for line in files: 

                    processed_line = line.strip()

                    if processed_line:

                        lines.append(processed_line)
for i, line in enumerate(lines):

    lines[i] = line.lower()
def line_to_tensor(line, EOS_int=1):

    

    tensor = []

    for c in line:

        c_int = ord(c)

        tensor.append(c_int)

    

    tensor.append(EOS_int)



    return tensor
def data_generator(batch_size, max_length, data_lines, line_to_tensor=line_to_tensor, shuffle=True):

    

    index = 0                         

    cur_batch = []                    

    num_lines = len(data_lines)       

    lines_index = [*range(num_lines)] 



    if shuffle:

        rnd.shuffle(lines_index)

    

    while True:

        

        if index >= num_lines:

            index = 0

            if shuffle:

                rnd.shuffle(lines_index)

            

        line = data_lines[lines_index[index]] 

        

        if len(line) < max_length:

            cur_batch.append(line)

            

        index += 1

        

        if len(cur_batch) == batch_size:

            

            batch = []

            mask = []

            

            for li in cur_batch:



                tensor = line_to_tensor(li)



                pad = [0] * (max_length - len(tensor))

                tensor_pad = tensor + pad

                batch.append(tensor_pad)



                example_mask = [0 if t == 0 else 1 for t in tensor_pad]

                mask.append(example_mask)

               

            batch_np_arr = np.array(batch)

            mask_np_arr = np.array(mask)

            

            

            yield batch_np_arr, batch_np_arr, mask_np_arr

            

            cur_batch = []

            
def GRULM(vocab_size=256, d_model=512, n_layers=2, mode='train'):

    model = tl.Serial(

      tl.ShiftRight(mode=mode),                                 

      tl.Embedding( vocab_size = vocab_size, d_feature = d_model), 

      [tl.GRU(n_units=d_model) for _ in range(n_layers)], 

      tl.Dense(n_units = vocab_size), 

      tl.LogSoftmax() 

    )

    return model
def LSTMLM(vocab_size=256, d_model=512, n_layers=2, mode='train'):

    model = tl.Serial(

      tl.ShiftRight(mode=mode),                                 

      tl.Embedding( vocab_size = vocab_size, d_feature = d_model), 

      [tl.LSTM(n_units=d_model) for _ in range(n_layers)], 

      tl.Dense(n_units = vocab_size), 

      tl.LogSoftmax() 

    )

    return model
def SRULM(vocab_size=256, d_model=512, n_layers=2, mode='train'):

    model = tl.Serial(

      tl.ShiftRight(mode=mode),                                 

      tl.Embedding( vocab_size = vocab_size, d_feature = d_model), 

      [tl.SRU(n_units=d_model) for _ in range(n_layers)], 

      tl.Dense(n_units = vocab_size), 

      tl.LogSoftmax() 

    )

    return model
GRUmodel = GRULM(n_layers = 5)

LSTMmodel = LSTMLM(n_layers = 5)

SRUmodel = SRULM(n_layers = 5)

print(GRUmodel)

print(LSTMmodel)

print(SRUmodel)
batch_size = 32

max_length = 64
eval_lines = lines[-1000:] # Create a holdout validation set

lines = lines[:-1000] # Leave the rest for training
from trax.supervised import training

import itertools



def train_model(model, data_generator, batch_size=32, max_length=64, lines=lines, eval_lines=eval_lines, n_steps=10, output_dir = 'model/'): 



    

    bare_train_generator = data_generator(batch_size, max_length, data_lines=lines)

    infinite_train_generator = itertools.cycle(bare_train_generator)

    

    bare_eval_generator = data_generator(batch_size, max_length, data_lines=eval_lines)

    infinite_eval_generator = itertools.cycle(bare_eval_generator)

   

    train_task = training.TrainTask(

        labeled_data=infinite_train_generator, 

        loss_layer=tl.CrossEntropyLoss(),   

        optimizer=trax.optimizers.Adam(0.0005)  

    )



    eval_task = training.EvalTask(

        labeled_data=infinite_eval_generator,    

        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],

        n_eval_batches=3    

    )

    

    training_loop = training.Loop(model,

                                  train_task,

                                  eval_tasks=[eval_task],

                                  output_dir = output_dir

                                  )



    training_loop.run(n_steps=n_steps)

    

    return training_loop

GRU_training_loop = train_model(GRUmodel, data_generator,n_steps=10, output_dir = 'model/GRU')
LSTM_training_loop = train_model(LSTMmodel, data_generator, n_steps = 10, output_dir = 'model/LSTM')
SRU_training_loop = train_model(SRUmodel, data_generator, n_steps = 10, output_dir = 'model/SRU')