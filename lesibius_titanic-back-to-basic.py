# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
from keras import models
from keras import layers
train.columns
train[['Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
train.describe()
not np.all(train['Sex'].isnull().apply(lambda x: not x))
sex_dummies_train = pd.get_dummies(train['Sex'])['female'].as_matrix()
sex_dummies_test = pd.get_dummies(test['Sex'])['female'].as_matrix()
not (np.all(pd.isnull(train['Age']).apply(lambda x: not x)))
mean_age = train['Age'].mean()
std_age = train['Age'].std()
def normalize_age(age):
    if pd.isnull(age):
        return 0
    else:
        return (age - mean_age)/std_age

norm_age_train = train['Age'].apply(normalize_age)
norm_age_test = test['Age'].apply(normalize_age)
not np.all(pd.isnull(train['SibSp']).apply(lambda x: not x))
mean_sibsp = train['SibSp'].mean()
std_sibsp = train['SibSp'].std()

norm_sibsp_train = train['SibSp'].apply(lambda x: (x - mean_sibsp)/std_sibsp)
norm_sibsp_test = test['SibSp'].apply(lambda x: (x - mean_sibsp)/std_sibsp)
print(not np.all(pd.isnull(train['Parch']).apply(lambda x: not x)))

mean_parch = train['Parch'].mean()
std_parch = train['Parch'].std()

norm_parch_train = train['Parch'].apply(lambda x: (x - mean_parch)/std_parch)
norm_parch_test = test['Parch'].apply(lambda x: (x - mean_parch)/std_parch)
print(not np.all(pd.isnull(train['Embarked']).apply(lambda x: not x)))
embarked_dummies_train = pd.get_dummies(train['Embarked'])[['C','Q','S']]
embarked_dummies_test = pd.get_dummies(test['Embarked'])[['C','Q','S']]
print(not np.all(pd.isnull(train['Fare']).apply(lambda x: not x)))
mean_fare = train['Fare'].mean()
std_fare = train['Fare'].std()

norm_fare_train = train['Fare'].apply(lambda x: (x - mean_fare)/std_fare)
norm_fare_test = test['Fare'].apply(lambda x: (x - mean_fare)/std_fare)
print(not np.all(pd.isnull(train['Cabin']).apply(lambda x: not x)))
train.loc[pd.isnull(train['Cabin']).apply(lambda x: not x),:]

test_re = train['Cabin'].iloc[763]
x = re.search('[A-Z]',test_re)
x = re.search('[0-9]+',test_re)

x.group(0)

def make_cabin(x):
    cabin = x
    
    if pd.isnull(cabin):
        return [0,0]
    alpha = re.search('[A-Z]',cabin)
    num = re.search('[0-9]+',cabin)
    if alpha:
        alpha = alpha.group(0)
    else:
        alpha = 0
    if num:
        num = num.group(0)
    else:
        num = 0
    return[alpha,num]

#I dropped the num for now, I do not expect it to be useful

def make_cabin_array(dataset):
    n_obs = len(dataset)
    res = np.zeros((n_obs,9))
    splitted = pd.DataFrame(np.matrix(dataset['Cabin'].apply(make_cabin).tolist()))
    splitted.columns = ['alpha','num']
    #splitted['num'] = splitted['num'].astype(int)
    #splitted['num'] = splitted['num'] / ...
    return pd.get_dummies(splitted['alpha'])['A B C D E F G'.split()]

cabin_dummies_train = make_cabin_array(train)
cabin_dummies_test = make_cabin_array(test)
print(not np.all(pd.isnull(train['Pclass']).apply(lambda x: not x)))
pclass_dummies_train = pd.get_dummies(train['Pclass'])[[1,2,3]]
pclass_dummies_test = pd.get_dummies(test['Pclass'])[[1,2,3]]
input_train = np.zeros((len(train),20))
input_test = np.zeros((len(test),20))

#Sex
input_train[:,0] = sex_dummies_train[:]
input_test[:,0] = sex_dummies_test[:]

#Age
input_train[:,1] = norm_age_train[:]
input_test[:,1] = norm_age_test[:]

#Sibling and spouses
input_train[:,2] = norm_sibsp_train[:]
input_train[:,3] = norm_parch_train[:]
input_test[:,2] = norm_sibsp_test[:]
input_test[:,3] = norm_parch_test[:]

#Embarked
input_train[:,4:7] = embarked_dummies_train
input_test[:,4:7] = embarked_dummies_test

#Fare
input_train[:,8] = norm_fare_train
input_test[:,8] = norm_fare_test

#Cabin
input_train[:,9:16] = cabin_dummies_train
input_test[:,9:16] = cabin_dummies_test

#Pclass
input_train[:,17:20] = pclass_dummies_train
input_test[:,17:20] = pclass_dummies_test
print(input_train)
train['Survived'].head()
max_epochs = 10
def k_fold_validation(model,data,folds=9):
    n_data = len(data)
    batch = int(n_data / folds)
    val_acc = []
    for i in range(folds):
        val_data = input_train[i * batch : (i + 1) * batch]
        val_labels = train['Survived'].iloc[i * batch : (i + 1) * batch]
        
        train_data = np.concatenate(
            (input_train[: i * batch], input_train[ (i+1) * batch:]), axis = 0)
        train_labels = np.concatenate(
            (train['Survived'].iloc[:i * batch],train['Survived'].iloc[(i+1) * batch:]))
        history = model.fit(train_data,train_labels,
                            epochs = max_epochs,
                            validation_data = (val_data,val_labels),
                            verbose = 0)
        val_acc.append(history.history['val_acc'])
    return [[np.mean(y), np.std(y)] for y in np.transpose(val_acc)]
                    

def build_model_v1(n_hidden):
    model = models.Sequential()
    model.add(layers.Dense(32,activation='relu',input_shape = (input_train.shape[1],)))
    for i in range(n_hidden):
        model.add(layers.Dense(32,activation='relu')) 
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model

val_acc = []
for i in range(3):
    print("Training with {} hidden layers".format(i+1))
    model = build_model_v1(i+1)
    val_acc.append(k_fold_validation(model,input_train))
    
i=1
for va in val_acc:
    x = np.array([i for i in range(max_epochs)])
    y = np.array([j[0] for j in va])
    e = np.array([j[1] for j in va])
    plt.figure(i)
    plt.subplot(i,1,i)
    plt.ylim(0.5,1)
    plt.errorbar(x,y,e)
    i += 1


model = build_model(3)
history = model.fit(input_train,train['Survived'],epochs=5)
submission = np.zeros((len(test),2))
preds = model.predict(input_test).reshape(len(test))
submission[:,1] = np.round(preds)
submission[:,0] = test['PassengerId']
submission = pd.DataFrame(submission,columns=['PassengerId','Survived'])
submission['PassengerId'] = submission['PassengerId'].apply(lambda x: int(x))
submission.fillna(0,inplace=True)
submission['Survived'] = submission['Survived'].apply(lambda x: int(x))
submission.to_csv('submission.csv',index=False)


