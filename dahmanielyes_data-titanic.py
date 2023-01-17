# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.impute import SimpleImputer
import seaborn as sn


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


test = pd.read_csv('/kaggle/input/titanic/test.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
genre = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(len(train))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def nan_padding(data, columns):
    for column in columns:
        imputer=SimpleImputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data


nan_columns = ["Age", "SibSp", "Parch"]

train = nan_padding(train, nan_columns)
test = nan_padding(test, nan_columns)


train.Name
train.loc[train['Name'].str.contains('Mrs.'), 'Name'] = 'Mrs.'
train.loc[train['Name'].str.contains('Mr.'), 'Name'] = 'Mr.'
train.loc[train['Name'].str.contains('Mme.'), 'Name'] = 'Madamme'
train.loc[train['Name'].str.contains('Rev.'), 'Name'] = 'Reverent'
train.loc[train['Name'].str.contains('Miss'), 'Name'] = 'Miss'
train.loc[train['Name'].str.contains('Master'), 'Name'] = 'Master'
train.loc[train['Name'].str.contains('Major'), 'Name'] = 'Major'
train.loc[train['Name'].str.contains('Dr.'), 'Name'] = 'Doctor'
train.loc[train['Name'].str.contains('Capt.'), 'Name'] = 'Captain'
train.loc[train['Name'].str.contains('Col.'), 'Name'] = 'Colonel'
train.loc[train['Name'].str.contains('Sagesser'), 'Name'] = 'Sagesser'
train.loc[train['Name'].str.contains('Countess'), 'Name'] = 'Countess'
train.loc[train['Name'].str.contains('Don.'), 'Name'] = 'Don.'
train.loc[train['Name'].str.contains('Ms.'), 'Name'] = 'Ms.'
train.loc[train['Name'].str.contains('Jonkheer'), 'Name'] = 'Ecuyer'
train.Name.unique()

test.loc[test['Name'].str.contains('Mrs.'), 'Name'] = 'Mrs.'
test.loc[test['Name'].str.contains('Mr.'), 'Name'] = 'Mr.'
test.loc[test['Name'].str.contains('Mme.'), 'Name'] = 'Madamme'
test.loc[test['Name'].str.contains('Rev.'), 'Name'] = 'Reverent'
test.loc[test['Name'].str.contains('Miss'), 'Name'] = 'Miss'
test.loc[test['Name'].str.contains('Master'), 'Name'] = 'Master'
test.loc[test['Name'].str.contains('Major'), 'Name'] = 'Major'
test.loc[test['Name'].str.contains('Dr.'), 'Name'] = 'Doctor'
test.loc[test['Name'].str.contains('Capt.'), 'Name'] = 'Captain'
test.loc[test['Name'].str.contains('Col.'), 'Name'] = 'Colonel'
test.loc[test['Name'].str.contains('Sagesser'), 'Name'] = 'Sagesser'
test.loc[test['Name'].str.contains('Countess'), 'Name'] = 'Countess'
test.loc[test['Name'].str.contains('Don.'), 'Name'] = 'Don.'
test.loc[test['Name'].str.contains('Ms.'), 'Name'] = 'Ms.'
test.loc[test['Name'].str.contains('Jonkheer'), 'Name'] = 'Ecuyer'

train.Name.unique()


from sklearn import preprocessing
encodage = ['Mrs.','Mr.','Madamme','Reverent','Miss','Master','Major','Doctor','Captain','Colonel','Sagesser','Countess','Don.','Ms.','Ecuyer']

le = preprocessing.LabelEncoder()
le.fit(encodage)

train['Name'] = le.transform(train['Name'].values)
test['Name'] = le.transform(test['Name'].values)

train['Sex'] = le.fit_transform(train['Sex'].values)
test.Name.value_counts()
def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

not_concerned_columns = ["PassengerId", "Ticket"]
train_data = drop_not_concerned(train, not_concerned_columns)
test_data = drop_not_concerned(test, not_concerned_columns)
corrfeatures = ["Pclass", "SibSp", "Parch","Fare","Sex","Name","Age","Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch", "Cabin"]
X = pd.get_dummies(train_data[features])
X_corr = train_data[corrfeatures]
corrmat = X_corr.corr()
top_corr_features = corrmat.index[abs(corrmat["Survived"])>=0]
plt.figure(figsize=(10,10))
g = sn.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

plt.figure(figsize=(10,5))
chart = sn.barplot(train.Sex,train.Survived)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
print()
plt.figure(figsize=(10,5))
chart = sn.barplot(train.Name,train.Survived)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
print()
train_data.isna().sum()

train_dum = pd.get_dummies(train_data[["Survived","Pclass","Age","Name","Parch","SibSp","Sex"]])
test_dum = pd.get_dummies(test_data[["Pclass","Age","Name","Parch","SibSp","Sex"]])

"""
train = train_dum.drop(["Sex_female"], axis=1).rename(columns={"Sex_male": "Sex"})
test = test_dum.drop(["Sex_female"], axis=1).rename(columns={"Sex_male": "Sex"})
"""


def normalization_age(tab):
    moy = tab["Age"].mean()
    var = tab["Age"].var()
    tab["Age"] = (tab["Age"] - moy)/var
    
normalization_age(train_dum)
normalization_age(test_dum)
train_dum.head()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

models = []
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("KNN",KNeighborsClassifier()))

X = train_dum.drop(['Survived'], axis=1)
Y = train_dum['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)


for name,model in models:
    print(name)
    model = model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    cv_result = cross_val_score(model,X,Y, cv = 10,scoring = "accuracy")
    #score = accuracy_score(Y_test, y_pred)
    print(cv_result)
    print('Average score = ',cv_result.sum()/10)
    print()
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

def NN_model(trainX,trainy):
    verbose, epochs, batch_size = 0, 250, 32
    n_row, n_features, n_outputs = trainX.shape[0], trainX.shape[1], 1
    
    model = Sequential()
    model.add(Dense(200, input_shape=((trainX.shape[1],)), kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(25, kernel_initializer='glorot_uniform', activation='relu'))
    #model.add(Dense(1, kernel_initializer='normal'))
    
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX.values, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    return model

t0 = time.time()
modelNN = NN_model(X_train,Y_train)
t1 = time.time()
total = t1-t0
print('Time to do the training :',total,' s')
ypred = modelNN.predict_classes(X_test)
print('Metrics : ', modelNN.metrics_names)
ypred = modelNN.predict_classes(X_test)
print(accuracy_score(Y_test, ypred))
plt.plot(modelNN.history.history['accuracy'])
plt.show()
train_dum.head()
test_dum.head()
ypred = modelNN.predict_classes(X_test)
exp = pd.DataFrame(data=ypred)
exp.to_csv('mycsvfile.csv',index=False)
"""X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam'
)
"""