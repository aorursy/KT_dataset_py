from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, GRU, BatchNormalization

from keras.callbacks import TensorBoard, ModelCheckpoint

import pandas as pd

import numpy as np

import re
#Data retrieved from https://www.kaggle.com/mousehead/songlyrics

song_df = pd.read_csv("../input/songlyrics/songdata.csv")

song_df.head()
song_df.at[0, "text"]
#Use the previous 64 characters to predict the 65th

SEQ_LEN = 64
def clean_lyrics(lyrics):

    lyrics = lyrics.replace("\n", ".").lower() #Newlines generally indicate pauses

    lyrics = re.sub(r"\(.*\)", "", lyrics) #Get rid of lines inside parentheses (chorus)

    lyrics = re.sub(r"\[.*\]", "", lyrics) #Get rid of lines inside brackets [chorus]

    lyrics = re.sub(r"[\(\)\[\]]", "", lyrics) #Some parentheses were unbalanced...

    lyrics = re.sub(r"(\s+\.)+", ". ", lyrics) #Some brackets were unbalanced...

    lyrics = re.sub(r"([\?\.\!\;\,])\.+", r"\1", lyrics)  #Drop periods appearing after other punctuation

    lyrics = re.sub(r"\s+", " ", lyrics)  #Replace 1 or more whitespace characters with a single space

    return " " * (SEQ_LEN) + lyrics + "E" #Pad the beginning with whitespace so we can predict without feeding in lyrics
#Check out random songs to see if we should add anything to clean_lyrics

random_index = np.random.choice(len(song_df))

clean_lyrics(song_df.at[random_index, "text"])
#Vectorize clean_lyrics over the entire song text column

song_df["clean"] = song_df.text.apply(clean_lyrics)
song_df.head()
data = song_df.clean.values

data[0]
from itertools import chain



#Chain takes a bunch of iterables and connects them together

#The * unpacks an iterable so you can use it as positional arguments

#For example:  print(*[1,2,3]) is the same as calling print(1,2,3)

char_set = set(chain(*data))

len(char_set) #46 characters to predict
print(sorted(char_set))
len(list(chain(*data))) #56 million characters in data
N = len(data) #Number of songs

K = len(char_set) #Number of unique characters
#Mappings back and forth between character and integer index

#It is imperative to sort the char_set, otherwise the enumeration will

#return different indices in future sessions, which will ruin our model

letter2idx = dict((c, i) for i, c in enumerate(sorted(char_set)))

idx2letter = dict((i, c) for i, c in enumerate(sorted(char_set)))
def create_batch(data, n=128):

    #Create a batch of n samples, each row in X representing SEQ_LEN letters from a song

    #with each row in y representing the one-hot encoding of the next letter (or the STOP character "S")

    #p_start determines the probability of starting at the beginning of the song vice a random point

    X = np.zeros((n, SEQ_LEN, K))

    y = np.zeros((n, K))

    

    for i in range(n):

        #random.choice(N) would make sequences ending in "E" SEQ_LEN times as likely

        #I still wanted them to be more common than uniform probability; here they are about 6x as likely

        song_idx = np.random.choice(N - int(SEQ_LEN * .9))

        song_len = len(data[song_idx])

        

        #We don't want to run out of song!  Clip the random choice to be within valid range

        start_idx = min(np.random.choice(song_len), song_len - SEQ_LEN - 1)

        

        #Iterate over letters in the song and one-hot encode them into the array

        for j, letter in enumerate(data[song_idx][start_idx:start_idx + SEQ_LEN]):

            letter_idx = letter2idx[letter]

            X[i, j, letter_idx] = 1

        

        #One-hot encode the next letter

        next_letter_idx = letter2idx[data[song_idx][start_idx + SEQ_LEN]]

        y[i, next_letter_idx] = 1

    

    return X, y
X, y = create_batch(data)
index_iter = iter(range(len(X)))
#Test to see if create_batch worked properly

i = next(index_iter)

"".join([idx2letter[idx] for idx in X[i].argmax(axis = 1)]), idx2letter[y[i].argmax()]
#Check what proportion of the next letters are the end of the song

np.mean(np.array([idx2letter[idx] for idx in y.argmax(axis = 1)]) == "E")
X.shape, y.shape
#Loading in the model I previously trained

model = load_model("../input/50000-songs-gru/songlyrics_gru_v1.hdf5")
"""

model = Sequential()

#return_sequences = True is required if plugging into another recurrent layer

model.add(GRU(128, dropout = .1, recurrent_dropout = .1, input_shape = (SEQ_LEN, K), return_sequences = True))

model.add(BatchNormalization())

model.add(GRU(128, dropout = .2, recurrent_dropout = .2))

model.add(BatchNormalization())

model.add(Dense(512, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(.3))

model.add(Dense(256, activation = "relu"))

model.add(Dropout(.5))

model.add(Dense(K, activation = "softmax"))

""";
#Save model weights at the end of each epoch

#chk_callback = ModelCheckpoint("models/songlyrics_gru_v1.hdf5", save_best_only = True)

#Save logs to check out TensorBoard

#tb_callback = TensorBoard()
#model.compile("adam", "categorical_crossentropy", ["accuracy"])
#model.train_on_batch(X, y)
#X, y = create_batch(data, n = 100000)
#model.fit(X, y, batch_size = 128, epochs = 20, callbacks = [chk_callback, tb_callback])
def make_song(model, start = " " * SEQ_LEN):

    #model (fitted Keras model) 

    #start (str) : 

    #  Beginning of the song to continue filling in with predictions

    

    #Pad with whitespace regardless of what comes in, send to lower case

    #make_song will break if user inputs something not in the alphabet

    start = list((" " * SEQ_LEN + start).lower())

    

    #Get the index number for the final SEQ_LEN letters

    X_digits = [letter2idx[letter] for letter in start[-SEQ_LEN:]]

    X = []

    

    for digit in X_digits:

        #Create a one-hot encoding for each letter

        row = np.zeros(K)

        row[digit] = 1

        X.append(row)



    #While we haven't predicted the end character

    while start[-1] != "E":

        #predict the next character, grabbing the last SEQ_LEN rows from X

        #prediction returns an array of just one element, so we index to retrieve it

        pred = model.predict(np.array(X[-SEQ_LEN:]).reshape((1, SEQ_LEN, K)))[0]

        

        #Force it to make songs of a decent length by setting P("E") = 0 and normalizing

        if len(start) < 300:

            pred[letter2idx["E"]] = 0

            pred = pred / pred.sum()

            

        #Also returns an array

        prediction = np.random.choice(K, 1, p = pred)[0]

        row = np.zeros(K)

        row[prediction] = 1

        X.append(row)

        start.append(idx2letter[prediction])

    

    #Return the letters, strip out the whitespace padding and drop the "E" character

    #Possible TODO: Enforce proper capitalization on the resulting song

    return "".join(start).strip()[:-1]
#Left this running for a while and periodically checked the songs it was making

#I believe there's lots of room left for training, but it's so slow and I'm moving on!

for i in range(1):

    stats = model.train_on_batch(*create_batch(data))

    if i % 100 == 0:

        print("Iteration {}, {}".format(i, stats))

        print(make_song(model))

        #model.save("models/songlyrics_gru_v2.hdf5")
#Some starting lyrics:

song_index = np.random.choice(len(song_df))

print(song_df.text[song_index][:SEQ_LEN*3])

print()



#Grab up to SEQ_LEN*2 because the first SEQ_LEN chars are whitespace

print(make_song(model, start = song_df.clean[song_index][:SEQ_LEN*2]))
song_index = np.random.choice(len(song_df))

print(song_df.text[song_index][:SEQ_LEN*3])

print()



print(make_song(model, start = song_df.clean[song_index][:SEQ_LEN*2]))
#As we can see, it really comes up with some random stuff sometimes

#The model could use more work (additional training, more expressive),

#but I'd be curious to see how it'd perform given just one genre of music