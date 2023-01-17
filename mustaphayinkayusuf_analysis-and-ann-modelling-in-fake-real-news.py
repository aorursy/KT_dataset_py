# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

true['status'] = 1

fake['status'] = 0
df = pd.concat([true, fake])

df = df.sample(frac = 1).reset_index(drop = True)

df.head()
df.shape
df.isna().sum()
df['len'] = df.text.apply(len)
df.head()
df.shape
df.status.value_counts()
#Pie chart showing the percentage of fake news and true news.

df.status.value_counts().plot(kind = 'pie', autopct = '%0.1f%%', explode = [0,0.2])

plt.title('Percentage of fake and real news')

plt.legend(['Fake', 'True'])

plt.show()
df.subject.value_counts()
#Bar chart showing the subject of news from highest to lowest.

df.subject.value_counts().plot(kind = 'bar', grid = 'True')

plt.title('Ranking of subject of news')

plt.legend()

plt.show()
df.groupby('subject')['status'].value_counts()
df.loc[df.status == 0, 'subject'].value_counts()
df.loc[df.status == 1, 'subject'].value_counts()
features = 500

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = features)

X = cv.fit_transform(df['text'])

y = df['status']



# Splitting the dataset into the Training set, Validation set and Test set?

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

x_test1, x_test2, y_test1, y_test2 = train_test_split(x_test, y_test, test_size = 0.50, random_state = 0)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences
###Build ANN

model = Sequential()

model.add(Dense(features,activation = 'relu', input_dim = features))

   

model.add(Dense(1,kernel_initializer='uniform',activation = 'sigmoid'))



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



model.summary()

history = model.fit(x_train.toarray(), y_train, batch_size = 500, epochs = 10, validation_data=(x_test1.toarray(), y_test1))
history.history
plt.subplot(1,2,1)

plt.title('Model Accuracy')

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Val'])

plt.subplot(1,2,2)

plt.title('Model Loss')

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train', 'Val'])

plt.tight_layout()

plt.show()
