import pandas as pd

import numpy as np

import os

import string



import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.figure_factory as ff

from wordcloud import WordCloud



import tensorflow as tf

print(tf.__version__)
os.listdir("/kaggle/input")
file_name = "/kaggle/input/Names.txt"



with open(file_name,'r') as f:

    names = f.read().split("\n")[:-1]
print("Number of Names: ",len(names))

print("\nMax Length of a Name: ",max(map(len,names))-1)
fig = go.Figure(data=[go.Table(

                header = dict(values = ["Names"]),

                cells = dict(values = [[name for name in np.random.choice(names,size=15)]]))])

    

fig.update_layout(title = "Randomly Chosen Names")



fig.show()
fig = ff.create_distplot([list(map(len,names))],

                        group_labels=["Length"])

    

fig.update_layout(title="Name-Length Distribution")



fig.show()
#Selecting Names with length not more than 10



MAX_LENGTH = 10

names = [name for name in names if len(name)<=MAX_LENGTH]

print("Number of Names: ",len(names))



assert max(map(len,names))<=MAX_LENGTH, f"Names length more than {MAX_LENGTH}"
start_token = " " # so that the network knows that we're generating a first token



# this is the token for padding, we will add fake pad token at the end of names 

# to make them of equal size for further batching

pad_token = "#"



#Adding start token in front of all Names

names = [start_token+name for name in names]

MAX_LENGTH += 1



# set of tokens

tokens = sorted(set("".join(names + [pad_token])))



tokens = list(tokens)

n_tokens = len(tokens)

print("Tokens: ",tokens)

print ('n_tokens:', n_tokens)
token_to_id = dict(zip(tokens,range(len(tokens))))

print(token_to_id)



def to_matrix(names, max_len=None, pad=token_to_id[pad_token], dtype=np.int32):

    """Casts a list of names into rnn-digestable padded matrix"""



    max_len = max_len or max(map(len, names))

    names_ix = np.zeros([len(names), max_len], dtype) + pad



    for i in range(len(names)):

        name_ix = list(map(token_to_id.get, names[i]))

        names_ix[i, :len(name_ix)] = name_ix



    return names_ix
print('\n'.join(names[::5000]))

print(to_matrix(names[::5000]))
X = to_matrix(names)

X_train = np.zeros((X.shape[0],X.shape[1],n_tokens),np.int32)

y_train = np.zeros((X.shape[0],X.shape[1],n_tokens),np.int32)



for i, name in enumerate(X):

    for j in range(MAX_LENGTH-1):

        X_train[i,j,name[j]] = 1

        y_train[i,j,name[j+1]] = 1

    X_train[i,MAX_LENGTH-1,name[MAX_LENGTH-1]] = 1

    y_train[i,MAX_LENGTH-1,token_to_id[pad_token]] = 1
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print("TPU Detected")

    

except ValueError:

    print("TPU not Detected")

    tpu = None



# TPUStrategy for distributed training

if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)



else: # default strategy that works on CPU and single GPU

    strategy = tf.distribute.get_strategy()
name_count = X.shape[0]

print("Names in training set: ",name_count)



if tpu:

    BATCH_SIZE = 128 * strategy.num_replicas_in_sync

else:

    BATCH_SIZE = 64



print("Setting Batch size to: ",BATCH_SIZE)

    

STEPS_PER_EPOCH = np.ceil(name_count/BATCH_SIZE)

print("Steps per epoch: ",STEPS_PER_EPOCH)



# GCS_PATH = KaggleDatasets().get_gcs_path()

# print("GCS Path: ",GCS_PATH)



AUTO = tf.data.experimental.AUTOTUNE

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False
train_dataset = (

    tf.data.Dataset.from_tensor_slices((X,y_train))

    .shuffle(5000)

    .cache()

    .repeat()

    .batch(BATCH_SIZE)

    .prefetch(AUTO))
num_rnn_units = 256

embedding_size = 16



def make_model():

    model = tf.keras.models.Sequential()



    model.add(tf.keras.layers.Embedding(n_tokens,embedding_size,input_length=MAX_LENGTH))

#     model.add(tf.keras.layers.LSTM(num_rnn_units,return_sequences=True,activation='elu',input_shape=(X_train.shape[1],X_train.shape[2])))

#     model.add(tf.keras.layers.LSTM(num_rnn_units,return_sequences=True,activation='elu'))

#     model.add(tf.keras.layers.Dropout(0.2))

#     model.add(tf.keras.layers.LSTM(num_rnn_units,return_sequences=True,activation='elu'))

    model.add(tf.keras.layers.SimpleRNN(num_rnn_units,return_sequences=True,activation='elu'))

    model.add(tf.keras.layers.SimpleRNN(num_rnn_units,return_sequences=True,activation='elu'))

    model.add(tf.keras.layers.Dense(n_tokens,activation='softmax'))



    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(0.001))



    return model
with strategy.scope():

    

    model = make_model()

    

    model.summary()
class CyclicLR(tf.keras.callbacks.Callback):

    

    def __init__(self,base_lr=1e-5,max_lr=1e-3,stepsize=10):

        super().__init__()

        

        self.base_lr = base_lr

        self.max_lr = max_lr

        self.stepsize = stepsize

        self.iterations = 0

        self.history = {}

        

    def clr(self):

        cycle = np.floor((1+self.iterations)/(2*self.stepsize))

        x = np.abs(self.iterations/self.stepsize - 2*cycle + 1)

        

        return self.base_lr + (self.max_lr - self.base_lr)*(np.maximum(0,1-x))

    

    def on_train_begin(self,logs={}):

        tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)

    

    def on_batch_end(self,batch,logs=None):

        logs = logs or {}

        

        self.iterations += 1

        

        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

        
def generateName(model=model,seed_phrase=start_token,max_length=MAX_LENGTH):

    

    assert len(seed_phrase)<max_length, f"Length of the Seed-phrase is more than Max-Length: {max_length}"

    

    name = [seed_phrase]

    x = np.zeros((1,max_length),np.int32)



    x[0,0:len(seed_phrase)] = [token_to_id[token] for token in seed_phrase]

    

#     x = np.zeros((1,max_length,n_tokens),np.int32)

    

#     for i in range(len(seed_phrase)):

#         x[0,i,token_to_id[seed_phrase[i]]] = 1

    

    for i in range(len(seed_phrase),max_length):

        

#         x_seq = (tf.data.Dataset.from_tensor_slices(x).batch(1))        

        

        probs = list(model.predict(x)[0,i-1])

        

        probs = probs/np.sum(probs)

        

        index = np.random.choice(range(n_tokens),p=probs)

        

        if index == token_to_id[pad_token]:

            break

            

#         x[0,i,index] = 1

        x[0,i] = index

        

        name.append(tokens[index])

    

    return "".join(name)





# def generateNamesLoop(epoch,logs):

#     if epoch%10==0:

#         print("\n--------------------------------------")

#         print(f"Names generated after epoch-{epoch}:")

        

#         for i in range(5):

#             print(generateName())

        

#         print("--------------------------------------")
%%time



# printNames = tf.keras.callbacks.LambdaCallback(on_epoch_end=generateNamesLoop)



cyclicLR = CyclicLR(base_lr=1e-4,max_lr=1e-3,stepsize=6000)



EPOCHS = 1000



history = model.fit(train_dataset,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,callbacks=[cyclicLR])
fig = go.Figure()



fig.add_trace(go.Scatter(x=np.arange(1,len(history.history['loss'])+1),

                        y=history.history['loss'],

                        mode='lines+markers',

                        name='Training loss'))



fig.update_layout(title_text="Training loss")



fig.show()
weights = '/kaggle/working/IndianNames(2SimpleRNN1000).h5'



model.save_weights(weights)
predictor = make_model()



predictor.load_weights(weights)
# When the Seed Phrase is start-token



seed_phrase = " "

for _ in range(20):

    name = generateName(predictor,seed_phrase=seed_phrase)

    if name not in names:

        print(f"{name.lstrip()} (New Name)")

    else:

        print(name.lstrip())
# When seed-phrase is a single Alphabet



seed_phrase = f" {np.random.choice(list(string.ascii_uppercase))}"

for _ in range(20):

    name = generateName(predictor,seed_phrase=seed_phrase)

    if name not in names:

        print(f"{name.lstrip()} (New Name)")

    else:

        print(name.lstrip())
# When seed-phrase is some combination of Alphabets



seed_phrase = f" {np.random.choice(list(string.ascii_uppercase))}{np.random.choice(list(string.ascii_lowercase))}"

for _ in range(20):

    name = generateName(predictor,seed_phrase=seed_phrase)

    if name not in names:

        print(f"{name.lstrip()} (New Name)")

    else:

        print(name.lstrip())
new_names = []



while len(new_names) is not 200:

    name = generateName(predictor)

    if name not in names:

        new_names.append(name.lstrip())



wordcloud = WordCloud(background_color="white",height=400,width=1900).generate(" ".join(new_names))



fig, ax = plt.subplots(figsize=(20, 10))

ax.imshow(wordcloud, interpolation='bilinear',aspect='auto')

ax.axis("off")

plt.show()