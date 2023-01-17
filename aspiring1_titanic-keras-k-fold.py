# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', index_col = 0)
train.head()
train.dtypes
train.info()
train.loc[pd.isnull(train['Age']), 'Age'] = train['Age'].median()
train.loc[pd.isnull(train['Cabin']), 'Cabin'] = '0'
train['Cabin'] = train['Cabin'].apply(lambda m: m[0])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train['Cabin'] = le.fit_transform(train['Cabin'])
train = pd.concat([train, pd.get_dummies(train['Cabin'])], axis = 1)
train.head()
embark_null = np.argmax(train.groupby('Embarked')['Survived'].agg('count'))# Replace the two null values with the maxima
embark_null
train.loc[train['Embarked'].isnull(), 'Embarked'] = embark_null
train.head()
train['Sex'] = train['Sex'].apply(lambda m: 1 if m[0] == 'f' else 0)
train.rename(columns= {0: 'nan', 1: 'A', 2: 'B', 3 : 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8 : 'H'}, inplace = True)
from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()
train['Embarked'] = le2.fit_transform(train['Embarked'])
train = pd.concat([train, pd.get_dummies(train['Embarked'])], axis = 1)
train.drop(columns=[ 'Name', 'Ticket', 'Cabin', 'Embarked'], inplace = True)
train.rename(columns = {0: 'P', 1 : 'Q', 2: 'S'}, inplace = True)
train.head()
X = train.iloc[:, 1:].as_matrix()
y = train.iloc[:, 0].as_matrix()
test = pd.read_csv('../input/test.csv', usecols = ['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin'], index_col = 0)
test.head()
test.info()
test.loc[test['Fare'].isnull(), 'Fare'] = test['Fare'].mean()
test.loc[pd.isnull(test['Age']), 'Age'] = test['Age'].median()
test.info()
test.head()
test = pd.concat([test, pd.get_dummies(test['Embarked'])], axis =1)
test.drop(columns = 'Embarked', inplace = True)
test['Sex'] = test['Sex'].map(lambda sex: 1 if sex[0] == 'f' else 0)
test.rename(columns = {'C': 'P'}, inplace = True)
test.head()
test.loc[pd.isnull(test['Cabin']), 'Cabin'] = '0'
test['Cabin'] = test['Cabin'].apply(lambda m : m[0])
test[1] = 0
test = pd.concat([test, pd.get_dummies(test['Cabin'])], axis = 1).drop(columns = 'Cabin')
test.rename(columns={'0': 'nan',1 : 'H'}, inplace = True)
test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','nan','A','B','C','D','E','F','G','H','P','Q','S']]
test.head()
train.head()
mean = X.mean(axis = 0)
X -= mean
test -= mean

std = X.std(axis = 0)
X /= std
test /= std
from keras import models
from keras import layers
from keras import regularizers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation = 'relu', kernel_regularizer = regularizers.l1(0.001), input_shape = (X.shape[1],)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(16, kernel_regularizer = regularizers.l1(0.001), activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer = 'rmsprop',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'])
    return model
import time
start = time.time()
k = 5
num_val_samples = len(train) // k
num_epochs = 100
all_losses = []
all_accuracies = []

for i in range(k):
    print('processing fold # ', i)
    val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate([X[: i * num_val_samples], X[ (i + 1) * num_val_samples: ]],
                                       axis = 0)
    partial_train_targets = np.concatenate([y[: i * num_val_samples], y[ ( i + 1) * num_val_samples:]],
                                          axis = 0)
    
    model = build_model()
    history = model.fit(partial_train_data,
                       partial_train_targets,
                       validation_data = (val_data, val_targets),
                       batch_size = 8,
                       verbose = 0,
                       epochs = num_epochs)
    val_losses = history.history['val_loss']
    val_accuracies = history.history['val_acc']
    all_losses.append(val_losses)
    all_accuracies.append(val_accuracies)
    
end = time.time()
print('\n', end - start)
loss_per_epoch = [np.mean([x[i] for x in all_losses]) for i in range(num_epochs)]
acc_per_epoch = [np.mean([x[i] for x in all_accuracies]) for i in range(num_epochs)]
np.argmin(loss_per_epoch)
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(1, num_epochs + 1), loss_per_epoch)
plt.title('Mean Validation loss per epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(1, num_epochs + 1), loss_per_epoch)
plt.title('Mean Validation loss per epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.plot(range(1, num_epochs + 1), acc_per_epoch)
plt.title('Mean Accuracies loss per epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.show()
np.max(acc_per_epoch)
model = build_model()
model.fit(X, y , epochs = np.argmin(loss_per_epoch), batch_size = 16, verbose = 0)
predictions = model.predict(test.values)
submission = pd.DataFrame({'Survived':(predictions > 0.5).reshape(-1, )}, test.index)
le = preprocessing.LabelEncoder()
submission['Survived'] = le.fit_transform(submission['Survived'])
submission.to_csv('test_results.csv')
!ls



