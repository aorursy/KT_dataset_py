# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

np.random.seed(444)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix





from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))





figure_size = (15,15)



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')
def create_confusion_matrix(predictions, labels):

    

    confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    

    label_list = list(labels.values.flatten())



    label_index = 0

    

    for row in predictions:



        if row == label_list[label_index]:

            

            if row == 1:

                confusion_matrix["TP"] += 1

            else:

                confusion_matrix["TN"] += 1

        else:

            if row == 1:

                confusion_matrix["FP"] += 1

            else:

                confusion_matrix["FN"] += 1

        label_index += 1



    print(confusion_matrix)

    print("n:{}".format(label_index))

    return confusion_matrix



def perf(model):

    try:

        preds = model.predict_classes(X_test)

    except:

        preds = model.predict(X_test)

        

    print(classification_report(y_test, preds))

    create_confusion_matrix(preds, y_test)

    accuracy_score(preds, y_test)





    
df.head()
df.hist('cp', figsize=(7,7))
df.hist("age", bins=75)
unique_ages = pd.unique(df.age)

print("Number of unique ages: {}".format(len(unique_ages)))

unique_ages.sort(kind='mergesort')

display(unique_ages)
df.target.hist()
display(df.describe())

display(df.isna().sum())
df.hist(bins=75, figsize=figure_size)
scatter_matrix(df, figsize=figure_size)
%matplotlib inline

correllations = df.corr()

fig = plt.figure(figsize=figure_size)

ax = fig.add_subplot(111)

cax = ax.matshow(correllations, vmin=-1, vmax=1)

fig.colorbar(cax)

names = df.columns

ticks = np.arange(0,len(names),1)

display(names)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

plt.show()
df.drop(['chol', 'fbs', 'ca', 'thal'], axis=1, inplace=True)
df.info()
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

y = df.target

X  = df.drop('target', axis=1)

scaler = StandardScaler()

# Fit the scaler to the data contained in 'X'

scaler.fit(X)

# Use scaler.transform() on the X and store the results below

scaled_X = scaler.transform(X)



pca = PCA()

# call pca.fit() on the data stored in 'scaled_data'.

pca.fit(scaled_X)

x_pca = pca.transform(scaled_X)

# pca_k_means = KMeans(n_clusters=len(X.columns))

# pca_k_means.fit(x_pca)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
def comp(model):

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

    model.summary()

    return model



def fit(model):

    model.fit(X_train, list(y_train), batch_size=64, epochs=80, verbose=1, validation_data=(X_test, list(y_test)))

    return model
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
clf = DecisionTreeClassifier()

fitted = clf.fit(X_train, list(y_train))

perf(fitted)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,list(y_train))

perf(logreg)

# display(logregs.coef_)

# display(logregs.intercept_)

display(logreg.predict([X_test.iloc[5]]))

# df.to_csv('../input/saved.csv')

from sklearn import linear_model
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

clf.fit(X_train,list(y_train))

perf(clf)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(X_train, list(y_train))

perf(clf)
from sklearn.cluster import KMeans
from keras.models import Sequential

from keras.layers import Dense, LeakyReLU

from keras.utils import to_categorical
small_model = Sequential()

small_model.add(Dense(9, input_shape=(9,)))

small_model.add(Dense(4, activation='relu'))

small_model.add(Dense(1, activation='sigmoid'))

comp(small_model)

fit(small_model)
perf(small_model)
small_relu_nn = Sequential()

small_relu_nn.add(Dense(9, input_shape=(9,)))

small_relu_nn.add(LeakyReLU(alpha=0.05))

small_relu_nn.add(Dense(4))

small_relu_nn.add(LeakyReLU(alpha=0.05))

small_relu_nn.add(Dense(1, activation='sigmoid'))



comp(small_relu_nn)

fit(small_relu_nn)
perf(small_relu_nn)
med_model = Sequential()

med_model.add(Dense(9, input_shape=(9,)))

med_model.add(LeakyReLU(alpha=0.05))

med_model.add(Dense(6))

med_model.add(LeakyReLU(alpha=0.05))

med_model.add(Dense(2))

med_model.add(LeakyReLU(alpha=0.05))

med_model.add(Dense(1, activation='sigmoid'))



comp(med_model)

fit(med_model)
perf(med_model)
med_leaky_model = Sequential()

med_leaky_model.add(Dense(9, input_shape=(9, )))

med_leaky_model.add(LeakyReLU(alpha=0.05))

med_leaky_model.add(Dense(6))

med_leaky_model.add(LeakyReLU(alpha=0.05))

med_leaky_model.add(Dense(3))

med_leaky_model.add(LeakyReLU(alpha=0.05))

med_leaky_model.add(Dense(3))

med_leaky_model.add(LeakyReLU(alpha=0.05))

med_leaky_model.add(Dense(1))



comp(med_leaky_model)

fit(med_leaky_model)
perf(med_leaky_model)
from keras.layers import Dropout

tail_model = Sequential()

tail_model.add(Dense(9, activation='relu', input_shape=(9,)))

tail_model.add(Dense(18, activation='relu'))

tail_model.add(Dense(9, activation='relu'))

tail_model.add(Dense(7, activation='relu'))

tail_model.add(Dense(4, activation='relu'))

tail_model.add(Dense(1, activation='sigmoid'))



comp(tail_model)

fit(tail_model)
perf(tail_model)
from keras.layers import Dropout, LeakyReLU

leakly_large_model = Sequential()

leakly_large_model.add(Dense(16, input_shape=(9,)))

leakly_large_model.add(LeakyReLU(alpha=0.05))

leakly_large_model.add(Dense(12))

leakly_large_model.add(LeakyReLU(alpha=0.05))

leakly_large_model.add(Dense(8))

leakly_large_model.add(LeakyReLU(alpha=0.05))

leakly_large_model.add(Dropout(.5))

leakly_large_model.add(Dense(4))

leakly_large_model.add(LeakyReLU(alpha=0.05))

leakly_large_model.add(Dense(1, activation='sigmoid'))



comp(leakly_large_model)

fit(leakly_large_model)
perf(leakly_large_model)
from keras.layers import Dropout

droppy_model = Sequential()

droppy_model.add(Dense(9, activation='relu', input_shape=(9,)))

droppy_model.add(Dense(4, activation='relu'))

droppy_model.add(Dropout(0.2))

droppy_model.add(Dense(1, activation='sigmoid'))



comp(droppy_model)

fit(droppy_model)
perf(droppy_model)