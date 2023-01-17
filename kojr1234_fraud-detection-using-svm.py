import numpy as np

import pandas as pd 

from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('../input/creditcard.csv')



#import libraries
dataset['new_amount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1, 1))

dataset.drop(['Time', 'Amount'], axis=1, inplace=True)

dataset.head()
train_set_percentage = 0.5

###################################################

# select 30% of the entire class 1 (fraudulent transactions) data in order to train the model 

fraud_series = dataset[dataset['Class'] == 1]

idx = fraud_series.index.values

np.random.shuffle(idx)

fraud_series.drop(idx[:int(idx.shape[0]*train_set_percentage)], inplace=True)

dataset.drop(fraud_series.index.values, inplace=True)

###################################################
###################################################

# normal dataset with the same size of the fraud_series (training dataset)

normal_series = dataset[dataset['Class'] == 0] 

idx = normal_series.index.values

np.random.shuffle(idx)

normal_series.drop(idx[fraud_series.shape[0]:], inplace=True)

dataset.drop(normal_series.index.values, inplace=True)

###################################################
# build the training dataset

new_dataset = pd.concat([normal_series, fraud_series])

new_dataset.reset_index(inplace=True, drop=True)

y = new_dataset['Class'].values.reshape(-1, 1)

new_dataset.drop(['Class'], axis=1, inplace=True)

X = new_dataset
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

# Attributes that will be used by the gridsearchCV algorithm 

attr={'C': [0.1, 1, 2, 5, 10, 25, 50, 100],

      'gamma': [1e-1, 1e-2, 1e-3]

     }



X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.3, random_state=10)



model = SVC()

classif = GridSearchCV(model, attr, cv=5)

classif.fit(X_train, y_train)

y_pred = classif.predict(X_test)

print('Accuracy: ',accuracy_score(y_pred, y_test))
y_all = dataset['Class'].values.reshape(-1, 1)

dataset.drop(['Class'], axis=1, inplace=True)

X_all = dataset

y_pred_all = classif.predict(X_all)

print(confusion_matrix(y_all, y_pred_all))
print(recall_score(y_all, y_pred_all))