# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding='latin-1')
data.info()
data.head()
data.shape
data.describe()
data = pd.concat([data.gender,data.description],axis=1)

print(data.info())
data.dropna(axis = 0,inplace = True)

data.gender = [1 if each == "female" else 0 for each in data.gender]

print(data.head())

print(data.info())
import re

import nltk # natural language tool kit

nltk.download("stopwords")      # corpus diye bir kalsore indiriliyor

from nltk.corpus import stopwords  # sonra ben corpus klasorunden import ediyorum



import nltk as nlp

lemma = nlp.WordNetLemmatizer()



from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
description_list = []

for description in data.description:

    # regular expression RE mesela "[^a-zA-Z]"

    description = re.sub("[^a-zA-Z]"," ",description)

    description = description.lower()   # buyuk harftan kucuk harfe cevirme

    # description = description.split()

    # split yerine tokenizer kullanabiliriz

    # split kullanırsak "shouldn't " gibi kelimeler "should" ve "not" diye ikiye ayrılmaz ama word_tokenize() kullanirsak ayrilir

    description = nltk.word_tokenize(description)

    # stopwords (irrelavent words) gereksiz kelimeler

    description = [ word for word in description if not word in set(stopwords.words("english"))]

    # lemmatazation loved => love   gitmeyecegim = > git

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description] #[ ps.stem(word) for word in description]

    description = " ".join(description)  #vektor için list değil string hali gerekli

    description_list.append(description)

description_list[0:5]
from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak icin kullandigim metot

max_features = 5000 #kelımelerden en cok kullanılan kac tane alınsın



count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x bağımsız değişken

print("en sik kullanilan {} kelimeden bazıları: {}".format(max_features,count_vectorizer.get_feature_names()[0:5]))



y = data.iloc[:,0].values   # male or female classes

x = sparce_matrix
import seaborn as sns

import matplotlib.pyplot as plt

# visualize number of digits classes

plt.figure(figsize=(10,6))

sns.countplot(y)

plt.title("male or female classes")
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 2)



# %% naive bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)



#%% prediction

y_pred = nb.predict(x_test)



print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)



# %% cm visualization

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression()

classifier.fit(x_train, y_train)

score = classifier.score(x_test, y_test)



#%% prediction

#y_pred = nb.predict(x_test)



print("Accuracy:", score)



from keras.models import Sequential

from keras import layers



input_dim = x_train.shape[1]  # Number of features



model = Sequential()

model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 

               optimizer='adam', 

               metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train,

                     epochs=50,

                     verbose=False,

                     validation_data=(x_test, y_test),

                     batch_size=5)
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))
import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
plot_history(history)