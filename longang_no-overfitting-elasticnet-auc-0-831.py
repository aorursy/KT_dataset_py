import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet



print(os.listdir("../input"))
def read_data(file_name):

    df = pd.read_csv(file_name)    

    Y = df['target']

    X = df.drop(columns=['target','id'])

    return X, Y



# load training data

X_train, Y_train = read_data('../input/train.csv')



# load testing data

X_test = pd.read_csv('../input/test.csv').drop(columns=['id'])



print(X_train.shape, Y_train.shape, X_test.shape)
X_train.head()
sns.countplot(Y_train)
X_train.isnull().values.any()
X_test.isnull().values.any()
X_train.describe()
sc = StandardScaler()



# convert to numpy array

X_train = X_train.values

Y_train = Y_train.values

X_test = X_test.values



# transform

X_train_scaled = sc.fit_transform(X_train)

X_test_scaled = sc.fit_transform(X_test)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))



ax1.set_title('Before Scaling')

for i in range(2):

    sns.kdeplot(X_train[i], ax=ax1)



ax2.set_title('After Standard Scaler')

for i in range(2):

    sns.kdeplot(X_train_scaled[i], ax=ax2)

plt.show()
best_parameters = { 

                    'alpha': 0.198, 

                    'l1_ratio': 0.3, 

                    'precompute': True, 

                    'selection': 'random', 

                    'tol': 0.001,

                    'random_state': 19

                }



net = ElasticNet(**best_parameters)

net.fit(X_train_scaled, Y_train)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = net.predict(X_test_scaled)

submission.to_csv('submission.csv', index=False)
submission.head(10)