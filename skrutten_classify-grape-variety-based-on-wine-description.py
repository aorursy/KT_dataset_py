import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout

from keras.callbacks import EarlyStopping



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split



from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import nltk

nltk.download("stopwords")
plt.rcParams['figure.figsize'] = (16,8)
df = pd.read_csv("../input/winemag-data_first150k.csv")

df.head()
print(df.shape)

df = df.drop(columns=["Unnamed: 0", "region_1", "region_2"])

df = df.dropna()

df.head()

print(df.shape)
df.describe()
sns.countplot(x="points", data=df)
# Hard to see so will limit the x axis

sns.distplot(df.price, bins=500).set(xlim=(0, 250))
g = sns.countplot(x="province", data=df,  order = df['province'].value_counts().iloc[:20].index)

plt.setp(g.get_xticklabels(), rotation=45)

plt.show()
g = sns.countplot(x="variety", data=df,  order = df['variety'].value_counts().iloc[:20].index)

plt.setp(g.get_xticklabels(), rotation=45)

plt.show()
dfWinary = df.groupby('winery').mean()

dfWinary.sort_values(by='points', ascending=False)['points'].iloc[:20].plot(kind='bar').set_title('Winery average point')

plt.show()

dfWinary.sort_values(by='price', ascending=False)['price'].iloc[:20].plot(kind='bar').set_title('Winery average price')

plt.show()

dfWinary['P/P'] = dfWinary.points / dfWinary.price

dfWinary.sort_values(by='P/P', ascending=False)['P/P'].iloc[:20].plot(kind='bar').set_title('Most bang for the buck')

plt.show()
df = pd.read_csv("../input/winemag-data_first150k.csv")

print(df.shape)

df = df[["description","variety"]]

df = df.dropna()

df = df.drop_duplicates(subset="description")

print(df.shape)

df.head()
df.description = df.description.str.replace('[^A-Za-z\s]+', '')

df.description = df.description.str.lower()
ps = PorterStemmer()

english_words = stopwords.words('english')



def stem_words(sentence):

    sentence = sentence.split()

    sentence = [word for word in sentence if not word in english_words]

    sentence = [ps.stem(word) for word in sentence]

    return ' '.join(sentence)



print(df['description'].head())

df['description'] = df['description'].apply(stem_words)

print(df['description'].head())
vocab_size = 10000

tokenize = Tokenizer(num_words=vocab_size)



tokenize.fit_on_texts(df.description)

print(len(tokenize.word_index) + 1)



X = tokenize.texts_to_matrix(df.description, mode='tfidf')
encoder = LabelBinarizer()

encoder.fit(df.variety)

Y = encoder.transform(df.variety)



del df
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)



del X

del Y
model = Sequential()



model.add(Dense(128, input_dim=x_train.shape[1]))

model.add(Activation('tanh'))

model.add(Dropout(0.2))



model.add(Dense(256))

model.add(Activation('sigmoid'))

model.add(Dropout(0.2))



model.add(Dense(y_train.shape[1]))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', metrics=['accuracy'],

              optimizer='adam')



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)



model_output = model.fit(x_train, y_train,

             batch_size=32,

             epochs=100,

             verbose=2,

             callbacks=[es],

             validation_split=0.2)
print('Training Accuracy : ' , np.mean(model_output.history["acc"]))

print('Validation Accuracy : ' , np.mean(model_output.history["val_acc"]))



plt.plot(model_output.history['acc'])

plt.plot(model_output.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(model_output.history['loss'])

plt.plot(model_output.history['val_loss'])

plt.title('model_output loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()