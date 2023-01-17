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
from sklearn.model_selection import train_test_split

import tensorflow as tf

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report, confusion_matrix

tf.__version__
df_alexa = pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv', sep = '\t')

df_alexa.head()
df_alexa.keys()
df_alexa['verified_reviews']
positive = df_alexa[df_alexa['feedback'] == 1]

negative = df_alexa[df_alexa['feedback'] == 0]
sns.countplot(df_alexa['feedback'], label = 'COUNT');



#The classes is unbalanced
#most ratings are between 4 and 5

sns.countplot(x = 'rating', data = df_alexa,palette='rocket')
df_alexa['rating'].hist(bins = 5)
plt.figure(figsize = (40,10))

sns.barplot(x = 'variation', y = 'rating', data = df_alexa, palette = 'rocket')
df_alexa = df_alexa.drop(['date', 'rating'], axis = 1)

df_alexa.head()
variation_dummies = pd.get_dummies(df_alexa['variation'])

variation_dummies
df_alexa.drop(['variation'], axis = 1, inplace=True)

df_alexa.head()
df_alexa = pd.concat([df_alexa, variation_dummies], axis = 1)

df_alexa.head()
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

alexa_countvectorizer = vectorizer.fit_transform(df_alexa['verified_reviews'])
vectorizer.get_feature_names()
df_alexa.drop(['verified_reviews'], axis = 1, inplace=True)
df_alexa.head()


reviews = pd.DataFrame(alexa_countvectorizer.toarray())

reviews
df_alexa = pd.concat([df_alexa, reviews], axis = 1)

df_alexa.head()
X = df_alexa.drop(['feedback'], axis = 1)
y = df_alexa['feedback']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
X_test.shape
classifier = tf.keras.models.Sequential()

classifier.add(tf.keras.layers.Dense(units = 400, activation='relu', input_shape=(4060,)))

classifier.add(tf.keras.layers.Dense(units = 400, activation='relu'))

classifier.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

classifier.summary()
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
epochs_hist = classifier.fit(X_train, y_train, epochs=15)
plt.plot(epochs_hist.history['loss'])

plt.title('Model loss progress during training')

plt.xlabel('Epoch')

plt.ylabel('Training loss')

plt.legend(['Training loss'])
plt.plot(epochs_hist.history['accuracy'])

plt.title('Model accuracy progress during training')

plt.xlabel('Epoch')

plt.ylabel('Training accuracy')

plt.legend(['Training accuracy'])
y_pred_test = classifier.predict(X_test)

y_pred_test = (y_pred_test > 0.5)

cm = confusion_matrix(y_test, y_pred_test)

cm
sns.heatmap(cm, annot=True);