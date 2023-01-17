from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import History, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, Input, Bidirectional, concatenate
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

import os
import csv
import random
import pandas
from pandas import Series
from collections import deque
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 

%matplotlib inline



GAMMA = 0.95
LEARNING_RATE = 0.001
MAXLEN = 30
MEMORY_SIZE = 1000000

MAX_TICKS=30
MEMORY_BATCH_SIZE = 20
GEN_BATCH_SIZE = MAX_TICKS

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
train_df = pd.read_csv("../input/train.csv")
#train_df, val_df = train_test_split(train_df, test_size=0.1)
# embdedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:MAXLEN]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (MAXLEN - len(embeds))
    return np.array(embeds)

# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
#val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:10000])])
#val_y = np.array(val_df["target"][:10000])

# Data providers



def batch_gen(train_df, batch_size):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])

# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        #print ('input_shape', input_shape)
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
class Env:
    """
    Environment for agent to play with 
    """

    def __init__(self, gen, max_ticks):
        self.action_space = 2

        # episode over 
        self.episode_over = False
        self.info = {} 

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.gen = gen
        self.max_ticks = max_ticks

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                state of system
               
            reward (float) :
               
            episode_over (bool) :
                
            info (dict) :
                optional info
                 
        """
        # ground_truth = self.gen_state[1]
        self.reward =  1 if  self.gen_state[1][self.ticks] == action else -1
        
        self.ticks += 1
        print('Tick', self.ticks, 'reward', self.reward) 

        # check if end of input batch 
        if self.ticks == self.max_ticks:
            self.episode_over = True
        else:
            self.state = self.gen_state[0][self.ticks]
            
        
        return self.state, self.reward, self.episode_over, self.info 
          

    def get_reward(self):
        return self.reward


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_episode += 1
        self.ticks = 0
        self.action_episode_memory.append([])
        
        self.episode_over = False
        self.gen_state = next(self.gen)
        self.state = self.gen_state[0][0]
        return self.state

    def get_state(self):
        """Get the observation. It is the gen_state[0]"""
        return self.state
    
    def cleanup(self):
        pass


# text_input = Input(shape=(MAXLEN ,300))
# x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(text_input)
# x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
# x = Attention(MAXLEN)(x)
# x = Dense(64, activation="relu")(x)
# x = Dropout(0.1)(x)
# x = Dense(64, activation="relu")(x)
# x = Dropout(0.1)(x)
# x = Dense(24, activation="relu")(x)
# model_output = Dense(2, activation="linear")(x)
# model = Model(inputs=text_input, outputs=model_output)
# model.compile(loss='mse',
#               optimizer=Adam(lr=1e-3),
#               metrics=['accuracy'])


# gen = batch_gen(train_df)
# state = next(gen)
# print (state[0].shape)
# print (state[0][0].shape)
# print (state[1].shape)
# text_input = np.reshape(state[0], [state[0].shape[0], 30, 300])

# output = model.predict(text_input)
# print (output.shape)
# action = np.argmax(output, axis = -1)
# sum(action == state[1]) + sum (action != state[1])

class DQNSolver:

    def __init__(self, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        text_input = Input(shape=(MAXLEN ,300))
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(text_input)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
        x = Attention(MAXLEN)(x)
        x = Dense(64, activation="relu")(x)
        #x = Dropout(0.1)(x)
        x = Dense(64, activation="relu")(x)
        #x = Dropout(0.1)(x)
        x = Dense(24, activation="relu")(x)
        model_output = Dense(self.action_space, activation="linear")(x)
        self.model = Model(inputs=text_input, outputs=model_output)
        self.model.compile(loss='mse',
                      optimizer=Adam(lr=1e-3),
                      metrics=['accuracy'])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            #print ("Taking random action", action)
            return action
    
        q_values = self.predict(state)
        #print ('q_values', q_values)
        action = np.argmax(q_values[0])
        #print ("Taking predicted  action", action)
        return action


    def predict(self, state):
        state = np.reshape(state, [1, MAXLEN, 300])
        return self.model.predict(state)
    
    def fit(self, state, q_values, verbose=0):
        state = np.reshape(state, [1, MAXLEN, 300])
        self.model.fit(state, q_values, verbose=0)
        
    def experience_replay(self):
        if len(self.memory) < MEMORY_BATCH_SIZE:
            return
        batch = random.sample(self.memory, MEMORY_BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:                
                q_update = (reward + GAMMA * np.amax(self.predict(state_next)[0]))
            q_values = self.predict(state)
            q_values[0][action] = q_update
            self.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save_model(self, model_name='model.h5'):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_name)
        print("Saved model to disk")
    random.seed(100)
    env = Env(batch_gen(train_df, GEN_BATCH_SIZE), MAX_TICKS)
    dqn_solver = DQNSolver(env.action_space)
    
    run = 0
    MAX_RUN = 50
    score_card = []
    while run < MAX_RUN:
        run += 1
        state = env.reset()
        step = 0
        score = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            score += reward
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(score))
                score_card.append((run, score, step))
                break
            dqn_solver.experience_replay()

    with open('dqn_stat_score_card_{0}.csv'.format(MAX_RUN), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(score_card)

    dqn_solver.save_model('dqn_stat_model_{0}_run.h5'.format(MAX_RUN))


    env.cleanup()

# random.seed(100)
# env = Env(batch_gen(train_df, GEN_BATCH_SIZE), MAX_TICKS)
# dqn_solver = DQNSolver(env.action_space)
# score_card = []
# state = env.reset()
# step = 0
# while True:
#     step += 1
#     #env.render()
#     action = dqn_solver.act(state)
#     state_next, reward, terminal, info = env.step(action)        
#     print ("Step: " + str(step) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(reward))
#     score_card.append((step, reward))
#     dqn_solver.remember(state, action, reward, state_next, terminal)
#     state = state_next
#     if terminal:
#         print ('Training ended after {} steps'.format(step))
#         break
#     dqn_solver.experience_replay()

# with open('dqn_nlp_score_card_{0}.csv'.format(MAX_TICKS), 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(score_card)

# dqn_solver.save_model('dqn_nlp_model_{0}_run.h5'.format(MAX_TICKS))

# env.cleanup()
def draw_cumulative_score(filename,title):
    df = pandas.read_csv(filename, header=None)
    data = df.values
    # x axis values 
    x =  [float(x[0]) for x in data]
    #print (x)
    # corresponding y axis values 
    y = [float(x[1]) for x in data]
    #print (y)
    series = Series(y)
    print ('Min', series.min(), 'Max', series.max(), 'mean', series.mean())
    # plotting the points  
    plt.plot(x, y) 

    # naming the x axis 
    plt.xlabel('Steps') 
    # naming the y axis 
    plt.ylabel('Score per step') 

    # giving a title to my graph 
    plt.title(title) 

    # show a legend on the plot 
    # plt.legend() 

    # function to show the plot 
    plt.show() 

draw_cumulative_score('dqn_stat_score_card_50.csv', 'DQN Agent')
# import matplotlib.pyplot as plt
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
model = dqn_solver.model
model.load_weights('dqn_stat_model_50_run.h5')
# prediction part
batch_size = 128
def batch_gen_test(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("../input/test.csv")

all_preds = []
for x in tqdm(batch_gen_test(test_df)):
    all_preds.extend(model.predict(x))
# Do Not Submit - predicts all zeroes **************************************
# action = np.argmax(all_preds, axis = -1).astype(np.int)
# submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": action})
# submit_df.to_csv("submission.csv", index=False)
