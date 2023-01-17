import pandas_profiling #for EDA

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#For text proccessing

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import text_to_word_sequence

#For network building

from keras.layers.embeddings import Embedding

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

#For model evaluation

from sklearn.model_selection import train_test_split
df=pd.read_csv("../input/winemag-data-130k-v2.csv")
print(df.shape)

print(df.columns)
pandas_profiling.ProfileReport(df)
del df["Unnamed: 0"]

df=df.loc[np.logical_and(df.taster_name.notna(),df.description.notna()),:]
df.taster_name.value_counts()
df.taster_name.value_counts().plot(kind="bar")
df["taster_name"]=[

    x if x in ["Roger Voss","Michael Schachner","Kerin O’Keefe","Virginie Boone","Paul Gregutt","Matt Kettmann","Joe Czerwinski"]

    else "Other Reviewer" 

    for x in df.taster_name

]
df.taster_name.value_counts()
for i in range(5):

    print(df.description[i])

    print("---------------------")
t=Tokenizer()

t.fit_on_texts(df.description.tolist())
#Had to break this up into chunks for memory purposes

X=(t.texts_to_sequences(df.description[0:50000]))

X.extend(t.texts_to_sequences(df.description[50000:]))
X=np.array(pad_sequences(X))
Y=pd.get_dummies(df.taster_name)

target_lables=Y.columns

Y=Y.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
max_words=len(t.word_counts)

embed_len=100

mem_units=100
model = Sequential()

model.add(Embedding(max_words, embed_len, input_length=128))

model.add(LSTM(mem_units))

model.add(Dense(8, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=64)
preds=model.predict_classes(X_test)

preds=[target_lables[x] for x in preds]

actual=[target_lables[x] for x in Y_test.argmax(axis=1)]
pd.crosstab(pd.Series(preds),pd.Series(actual))