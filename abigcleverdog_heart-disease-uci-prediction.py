# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

from sklearn.preprocessing import StandardScaler



from keras.models import Sequential

from keras.layers import Dense



import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

print(df.shape)

df.sample(10)
sns.countplot(df.target)
missing = df.isnull().sum()

missing_pct = missing/len(df)*100

missing_table = pd.DataFrame(data=[missing, missing_pct],

                            index=['missing','% missing'])

missing_table.T # convinient way to rotate the data
sns.distplot(df.age)
def evl_models(X_train, X_test, y_train, y_test):

    result = {}

    model_names = ['Logistric Regression', 'K Nearest Neighbors (K=5)', 'Random Forest', 'SVC', 'Linear SVC', 'Naive Bayes']

    models = [LogisticRegression(), KNeighborsClassifier(n_neighbors=5), RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1),

             SVC(), LinearSVC(), GaussianNB()]



    for name,model in zip(model_names, models):

        m = model

        m.fit(X_train, y_train)

        y_pred = m.predict(X_test)

        score = metrics.accuracy_score(y_test, y_pred)

        print(name, 'accuracy score is: ', score)

        result[name] = score

    return result
result = {}

for i in range(1,10):

    X, y = df.drop('target', axis=1), df.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    print('*'*10,' trial {}'.format(i))

    r = evl_models(X_train, X_test, y_train, y_test)

    for k in r:

        print(r[k], k)

        if k in result:

            result[k].append(r[k])

        else:

            result[k] = [r[k]]

        
plt.figure(figsize=(15,7))

for k in result:

    sns.lineplot(x=range(1,10), y=result[k], label=k)

# plt.legend
result = {}

for i in range(1,10):

    X, y = df.drop('target', axis=1), df.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)

    print('*'*10,' trial {}'.format(i))

    r = evl_models(X_train, X_test, y_train, y_test)

    for k in r:

        print(r[k], k)

        if k in result:

            result[k].append(r[k])

        else:

            result[k] = [r[k]]
plt.figure(figsize=(15,7))

for k in result:

    sns.lineplot(x=range(1,10), y=result[k], label=k)
def evl_models_10fold(X,y):

    result = []

    model_names = ['Logistric Regression', 'K Nearest Neighbors (K=5)', 'Random Forest', 'SVC', 'Linear SVC', 'Naive Bayes']

    models = [LogisticRegression(), KNeighborsClassifier(n_neighbors=5), RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1),

             SVC(), LinearSVC(), GaussianNB()]



    for name,model in zip(model_names, models):

        m = model

        scores = cross_val_score(m, X,y, cv=10,scoring='accuracy')

        print(name, 'accuracy score is: ', scores)

        result.append(scores)

    return zip(model_names,result)
X, y = df.drop('target', axis=1), df.target

res = evl_models_10fold(X,y)

plt.figure(figsize=(15,7))

for n, r in res:

    print(len(r), n, r)

    sns.lineplot(x=range(1,11), y=r, label=n)
X, y = df.drop('target', axis=1), df.target

res = evl_models_10fold(X,y)

# print(list(res))

res = list(res)

colors = sns.color_palette() # seaborn's default 10 colors list

plt.figure(figsize=(15,7))

for i, t in enumerate(res):

    n, r = t

    sns.lineplot(x=range(1,11), y=r, label=n)

    print(r.mean())

    plt.axhline(y=r.mean(), ls='--', color=colors[i])

temp = [[i,j.mean()] for i,j in res]

temp.sort(key=lambda x: -x[1])

print(temp)

temp = pd.DataFrame(temp, columns=['Model','Mean Accuracy'])

print(temp)

# plt.show()
from keras.utils.np_utils import to_categorical

X, y = df.drop('target', axis=1), df.target

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2)

rfr, nnr = [], []

for sub_train, sub_test in kfold.split(X, y):

    yy = y #to_categorical(y.copy(), num_classes = 2)

    scaler = StandardScaler().fit(X.iloc[sub_train])

    XX = scaler.transform(X)

#     print(XX[:5])

    # Random Forest for comparison

    rf = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)

    rf.fit(XX[sub_train], yy[sub_train])

    y_pred = rf.predict(XX[sub_test])

#     print(y_pred[:5])

    score = metrics.accuracy_score(yy[sub_test], y_pred)

    rfr.append(score)

    print('Random Forest accuracy score is: ', score)

    

    # TensorFlow

    model = Sequential()

    # Add an input layer 

    model.add(Dense(15, activation='relu', input_shape=(13,)))

    # Add one hidden layer 

    model.add(Dense(8, activation='relu'))

    # Add an output layer 

    model.add(Dense(1, activation='sigmoid'))

#     print(model.output_shape, model.summary(), model.get_config(), model.get_weights())

    model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

    model.fit(XX[sub_train], yy[sub_train], epochs=20, batch_size=5, verbose=0)

    

    y_pred = model.predict(XX[sub_test]).round()

#     print(y_pred[:5])

    score = metrics.accuracy_score(yy[sub_test], y_pred)

    nnr.append(score)

    print('TensorFlow accuracy score is: ', score)

    confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score = metrics.confusion_matrix(yy[sub_test], y_pred), metrics.precision_score(yy[sub_test], y_pred), metrics.recall_score(yy[sub_test], y_pred), metrics.f1_score(yy[sub_test], y_pred), metrics.cohen_kappa_score(yy[sub_test], y_pred)

    print('confusion_matrix: \n{}, \nprecision_score: {}, recall_score: {}, f1_score: {}, cohen_kappa_score: {}'.format(confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score))

    

plt.figure(figsize=(15,7))    

sns.lineplot(x=range(1,len(rfr)+1), y=rfr, label='Random Forest')

sns.lineplot(x=range(1,len(rfr)+1), y=nnr, label='Neural Network')

plt.ylim(0,1)