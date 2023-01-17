import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter



import tensorflow as tf

from tensorflow.contrib import seq2seq

import time #helper libraries

print('TensorFlow Version: {}'.format(tf.__version__))
recipes =  pd.read_json('../input/train.json')['ingredients']
recipes.head()
#average recipe length

print('average recipe length',sum( map(len, recipes) ) / len(recipes))



#number of unique ingredients

raw_ingredients = list()



for recipe in recipes:

    for ingredient in recipe:

        raw_ingredients.append(ingredient.strip())



counts_ingr = Counter(raw_ingredients)

vocab = sorted(counts_ingr, key=counts_ingr.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab)}

int_to_vocab = {ii: word for ii, word in enumerate(vocab)}            

number_of_unique_ingredients=len(counts_ingr)



print('number of unique ingredients:',number_of_unique_ingredients)
def create_dataset(recipes, look_around=2,batch_size=2):

    

    batches = []

    for recipe in recipes:

        dataX, dataY = [],[]        

        recipe_np = recipe

        for idx,ingredient in enumerate(recipe_np):

            init = []

            finish = []

            

            if(idx>0):                

                init=list(range(max(0,idx-look_around),idx))

                          

            finish=list(range(idx+1,min(idx+look_around+1,len(recipe_np))))            

            

            aux=[]            

            for xidx in list(init+finish):                

                aux.append(vocab_to_int[recipe_np[xidx]])

            

            if(len(aux)== look_around*2):

                

                dataX.append(aux)

                y = [vocab_to_int[recipe[idx]]]

                dataY.append(y)

                

    

            if(len(dataX)==batch_size):  

            

                batches.append([np.reshape(dataX, (batch_size, -1)),np.array(np.reshape(dataY, (batch_size, -1)))])

                

                dataX = []

                dataY = []

    

    return batches
training_batches = create_dataset(recipes,2,2)

print(training_batches[0])
# Number of Epochs

num_epochs =1

# Batch Size

batch_size = 2

# RNN Size

rnn_size = 256

# Sequence Length

seq_length = 2

# Learning Rate

lrate = 0.1

# Show stats for every n number of batches

show_every_n_batches = 1

#let's set the look_around variable to 5

look_around = 5
#Step 2 Build Model

tf.reset_default_graph()



train_graph = tf.Graph()

with train_graph.as_default():

    

    #placeholders

    input_text=tf.placeholder(tf.int32, [None, None], name='input')

    targets=tf.placeholder(tf.int32, [None, None], name='targets')

    learning_rate=tf.placeholder(tf.float32, name='learning_rate')

    

    input_data_shape = tf.shape(input_text)       

    

    # Your basic LSTM cell

    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

            

    # Stack up multiple LSTM layers, for deep learning

    cell = tf.contrib.rnn.MultiRNNCell([lstm])

    

    initial_state=tf.identity(cell.zero_state(batch_size, tf.float32),name="initial_state")

   

    embedding = tf.Variable(tf.random_uniform((number_of_unique_ingredients, rnn_size), -1, 1))

    embed = tf.nn.embedding_lookup(embedding, input_text)

    

    print('embed',embed.get_shape())

    

    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)

    print('outputs',outputs.get_shape())

    logits = tf.contrib.layers.fully_connected(outputs[:, -1], number_of_unique_ingredients, activation_fn=None)

    

    # Probabilities for generating words

    probs = tf.nn.softmax(logits, name='probs')



    # Loss function

    print(input_data_shape.get_shape())

    print(targets.get_shape())

    print(logits.get_shape())    

    

    cost = tf.losses.mean_squared_error(targets, probs)    

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
training_batches = create_dataset(recipes,batch_size,seq_length)

training_batches[0]
with tf.Session(graph=train_graph) as sess:

        sess.run(tf.global_variables_initializer())

        

        for epoch_i in range(num_epochs):

            

            state = sess.run(initial_state, {input_text: training_batches[0][0]})            

                        

            for batch_i, (x, y) in enumerate(training_batches):

                print(x)

                feed = {

                    input_text: x,

                    targets: y,

                    initial_state: state,

                    learning_rate: lrate}

                train_loss, state = sess.run([cost, final_state], feed)



                # Show every <show_every_n_batches> batches

                if (epoch_i * len(training_batches) + batch_i) % show_every_n_batches == 0:

                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(

                        epoch_i,

                        batch_i,

                        len(training_batches),

                        train_loss))

                    