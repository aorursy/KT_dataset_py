from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from keras.preprocessing.text import Tokenizer

data = pd.read_csv("../input/training_set.csv", index_col="essay_id")

data

xdata = data['essay']

ydata = data['domain1_score']

# from sklearn.preprocessing import StandardScaler

# ydata = np.array(ydata)

# ydata.shape = (-1, 1)

# # print(ydata)

# scaler = StandardScaler().fit(ydata)

# ydata = scaler.transform(ydata)

ydata = ydata / 60

tokenizer = Tokenizer(num_words=None, 

                     filters='$%&()*+-<=>@[\\]^_`{|}~\t\n',

                     lower = False, split = ' ')



tokenizer.fit_on_texts(xdata)



xdata = tokenizer.texts_to_sequences(xdata)

ydata.describe()
print(ydata)
xdata = np.array(xdata)

ln = max([len(i) for i in xdata])

xnd = np.zeros((len(xdata), ln), np.uint32)

for i, seq in enumerate(xdata):

    for j in range(len(seq)):

        xnd[i, j+ln-len(seq)] = seq[j]

xdata = xnd
X_train, X_test, y_train, y_test = train_test_split(xdata,ydata)
print(max([len(i) for i in xdata]))
idx_words = tokenizer.index_word

from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()



# Embedding layer

model.add(

    Embedding(input_dim=len(idx_words)+1,

              input_length=max([len(i) for i in xdata]),

              output_dim=50))



# Masking layer for pre-trained embeddings



# Recurrent layer

model.add(LSTM(64, return_sequences=False, 

               dropout=0.1, recurrent_dropout=0.1))

# model.add(LSTM(64, return_sequences=False, 

#                dropout=0.1, recurrent_dropout=0.1))



# Fully connected layer

model.add(Dense(64, activation='relu'))



# Dropout for regularization

model.add(Dropout(0.5))



# Output layer

model.add(Dense(1, activation='sigmoid'))



# Compile the model

model.compile(

    optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint



# Create callbacks

callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint('../models/model.h5', save_best_only=True, save_weights_only=False)]
y_train
history = model.fit(X_train,y_train, 

                    batch_size=2048, epochs=256,

                    callbacks=callbacks, verbose=1)
y_pred = model.predict(X_test)
y_pred *= 60

y_test *= 60
y_pred = [round(i[0]) for i in y_pred]
print(np.array([y_pred[:50], list(y_test[:50])]))
mine = ['''It’s going to be a catastrophe, but it doesn’t matter, it’s required. they’re going to be hundreds of deaths, but it doesn’t matter. I’m going to do die, but it doesn’t matter, the homeland is more important. 3… 2… 1…



It began when Wilhelm saw this advertisement:

    

He understood that in order to win against Britain in a war, which is probably going to come soon, Germany must overcome Britain’s enormous fleet. So he contacted me, Wenzel Kroemer, the head of German intelligence. I had to construct a plan to reveal the British ship secrets. And so I did. I was able to purchase a first-class ticket for the first voyage of the cruise ship. It was expensive but with the German budget, it was nothing. I took with me some clothes, a secured radio communicator and a… bomb. Yes. This was the plan. during the voyage, I would get any possible data about the ship and its building techniques, and when we’ll be close to the coast, I’ll bomb the ship, so the data gathered and what happened there will stay a secret, known only to the German leaders.



10th April:

We set sail in the first voyage of the Titanic ever. Apart from the important service that I serve to Germany, I’m also pretty excited to be on this voyage, as an individual. we are sailing toward Cherbourg, and we should arrive today.



We arrived at Cherbourg for a short stop, and from here we are going to sail towards Queenstown, our final stop.



11th April:

We arrived at Queenstown, our second and final stop, and we’re going to do the last sail in a few days. after I’ve been a while at the Titanic and saw this wonderful ship and those amazing people, I’m feeling a little bad about destroying it. but for the mission, everything should be done.



We set sail toward New York, the final destination. Only me, out of those thousands of people, know the truth - we will never get there.



12th April:

I started gathering data. I found out that British ships use a lot of elastic metal plates to overcome the hard conditions of the ocean, instead of one big plate.



13th April:

Tomorrow is the crucial day. I feel terrible about all of the people the I’m going to ruin their life, but after I saw this ship, I can’t understand why Britain hasn’t attacked already. Their ships are better than ours in a few orders of magnitude, one of their ships can take out about 10 of ours… That makes me understand how important this move is, to give us a chance against the British fleet.



14th April:

Today is the day. I am going to plug the bomb into the engines and then leave to the deck. I’ll take the remote and wait until I see an iceberg, so this disaster wouldn’t be suspicious and will make unwanted focus on Germany.



I see an iceberg. Now it’s the time. I’m going to do this. it’s gonna be a catastrophe, but it doesn’t matter, it’s required. they’re going to be hundreds of deaths, but it doesn’t matter. I’m going to die, but it doesn’t matter, the homeland is more important. 3… 2… 1…

''']

mine = tokenizer.texts_to_sequences(mine)

mine = np.array(mine)

mn = np.zeros((len(mine), ln), np.uint32)

for i, seq in enumerate(mine):

    for j in range(len(seq)):

        mn[i, j+ln-len(seq)] = seq[j]

mine = mn

print(mine)

output = model.predict(mine)

print(output*60)