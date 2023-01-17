# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

np.random.seed(123)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import tensorflow as tf

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, classification_report, confusion_matrix,accuracy_score



fig_size = (15,15)
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        df = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
model_scores={}

model_index = 0



def create_confusion_matrix(predictions, labels):

    

    confusion_matrix = {"True Positive": 0, "True Negative": 0, "False Positive": 0, "False Negative": 0}

    label_list = list(labels.values.flatten())



    label_index = 0

    

    for row in predictions:



        if row == label_list[label_index]:

            

            if row == 1:

                confusion_matrix["True Positive"] += 1

            else:

                confusion_matrix["True Negative"] += 1

        else:

            if row == 1:

                confusion_matrix["False Positive"] += 1

            else:

                confusion_matrix["False Negative"] += 1

        label_index += 1



    print(confusion_matrix)

    print("n: {}".format(label_index))

    return confusion_matrix





def analyze_performance(classifier, label=None):

    try:

        preds = classifier.predict_classes(X_test)

    except:

        preds = classifier.predict(X_test)

        

    print(classification_report(y_test, preds))

    

    f1 = f1_score(y_test, preds)

    if label:

        model_scores[label] = round(f1, 2)

    else:

        model_scores[model_index] = round(f1, 2)

        

    display("F1: {}".format(round(f1, 2)))

    create_confusion_matrix(preds, y_test)

    

    accuracy_score(preds, y_test)

df.head()
df.hist(bins=75, figsize=fig_size)
df.describe()
scatter_matrix(df, figsize=fig_size)
%matplotlib inline

correllations = df.corr()

fig = plt.figure(figsize=fig_size)

ax = fig.add_subplot(111)

cax = ax.matshow(correllations, vmin=-1, vmax=1)

fig.colorbar(cax)

names = df.columns

ticks = np.arange(0,len(names),1)

display(names)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names, rotation='vertical')

ax.set_yticklabels(names)

plt.show()
#

corrs = df.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

# Remove features correllated with itself

corrs = corrs[corrs['level_0'] != corrs['level_1']]

corrs.tail(10)
df.info()
y = df.target_class

X = df.drop('target_class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
from sklearn.tree import DecisionTreeClassifier

decision_clf = DecisionTreeClassifier()

decision_clf.fit(X_train, y_train)
analyze_performance(decision_clf, 'Decision Tree')
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(X_train, y_train)

analyze_performance(clf, "Gaussian NB")
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

analyze_performance(logreg, "Logistic Regression")
from keras.models import Sequential

from keras.layers import Dense, LeakyReLU

from keras.optimizers import SGD

from keras.layers import Dropout



# from keras.layers import LeakyReLU
#Convert test to proper format

poor_nn = Sequential()

poor_nn.add(Dense(8, input_dim=8, activation='relu'))

poor_nn.add(Dense(4, activation='relu'))

poor_nn.add(Dense(1, activation='softmax'))

#poor_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

poor_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

poor_nn.summary()
poor_nn.fit(X_train, y_train, batch_size=64, epochs=80, verbose=1, validation_data=(X_test, y_test))
analyze_performance(poor_nn, "Poor relu NN")
small_nn = Sequential()

small_nn.add(Dense(8, input_dim=8, activation='relu'))

small_nn.add(Dense(4))

small_nn.add(LeakyReLU(alpha=0.05))

small_nn.add(Dense(1, activation='sigmoid'))

#small_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

small_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mape'])

small_nn.summary()
small_nn.fit(X_train, y_train, batch_size=64, epochs=80, verbose=1, validation_data=(X_test, y_test))
analyze_performance(small_nn, "small leaky nn")
med_nn = Sequential()

med_nn.add(Dense(12, input_shape=(8,)))

med_nn.add(LeakyReLU(alpha=0.05))

med_nn.add(Dense(8))

med_nn.add(LeakyReLU(alpha=0.05))

med_nn.add(Dense(4))

med_nn.add(LeakyReLU(alpha=0.05))

med_nn.add(Dense(1, activation='sigmoid'))



med_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mape'])

med_nn.summary()
med_nn.fit(X_train, y_train, batch_size=64, epochs=80, verbose=1, validation_data=(X_test, y_test))
analyze_performance(med_nn, "medium leaky nn")
smallrelu_nn = Sequential()

smallrelu_nn.add(Dense(12, input_dim=8, activation='relu'))

smallrelu_nn.add(Dense(4, activation='relu'))

smallrelu_nn.add(Dense(1, activation='sigmoid'))

smallrelu_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mape'])

smallrelu_nn.summary()
smallrelu_nn.fit(X_train, y_train, batch_size=64, epochs=90, verbose=1, validation_data=(X_test, y_test))
analyze_performance(smallrelu_nn, "small relu nn")
from keras.layers import LeakyReLU



large_nn = Sequential()

large_nn.add(Dense(16, input_dim=8))

large_nn.add(LeakyReLU(alpha=0.05))

large_nn.add(Dense(8))

large_nn.add(LeakyReLU(alpha=0.05))

large_nn.add(Dense(4))

large_nn.add(LeakyReLU(alpha=0.05))

large_nn.add(Dense(2))

large_nn.add(LeakyReLU(alpha=0.05))

large_nn.add(Dense(1))

large_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mape'])

large_nn.summary()
large_nn.fit(X_train, y_train, batch_size=64, epochs=90, verbose=1, validation_data=(X_test, y_test))
analyze_performance(large_nn, 'Large Leaky')
fig, ax = plt.subplots(figsize=(11, 9))

x = model_scores.keys()

y = [model_scores[i] for i in x]

display(y)

plt.bar(range(len(y)),y, color='#333333')

plt.plot(x, y)
