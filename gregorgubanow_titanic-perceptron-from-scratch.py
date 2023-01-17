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



# My imports

import tqdm

import matplotlib.pyplot as plt

%matplotlib inline 
def fit(X, y, alpha=0.005, epochs=600):

    weights = np.zeros(len(X[0]))

    errors = []

    data = list(zip(X, y))

    for _ in tqdm.tqdm(range(0, epochs)):

        np.random.shuffle(data)

        error = 0

        for inputs, target in data:

            output = predict(inputs, weights)

            

            if output >= 0.5 and target == 0:

                error += 1

            elif output < 0.5 and target == 1:

                error += 1



            weights += alpha * (target - output) * (output) * (1 - output) * inputs

        errors.append(error)

    return weights, errors
def predict(inputs, weights):

    x = inputs.dot(weights)

    return 1. / (1 + np.exp(-x))
df = pd.read_csv('../input/train.csv')

df.head()
df = df[['Pclass', 'Sex', 'Parch', 'SibSp', 'Survived']]

df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'female' else 1)

df = df.sample(frac=1.) # Shuffle data

df.head()
X = df.iloc[:,0:-1].values

X = np.insert(X,0,1,axis=1) # Bias

y = df['Survived'].values
# Split dataset

test_length = int(0.2 * len(y))

train_length = len(y) - test_length    

X_train, X_test = X[0:train_length], X[train_length:]

y_train, y_test = y[0:train_length], y[train_length:]
weights, errors = fit(X_train, y_train, alpha=0.005, epochs=650)

print(weights)
plt.plot(errors)

plt.xlabel('Epochs')

plt.ylabel('Errors')
correctly, wrong = 0, 0

for inputs, target in zip(X_test, y_test):

    output = predict(inputs, weights)



    if output >= 0.5 and target == 0:

        wrong += 1

    elif output < 0.5 and target == 1:

        wrong += 1

    else:

        correctly += 1
plt.pie([correctly, wrong], 

        labels=['correctly ({})'.format(correctly), 'wrong ({})'.format(wrong)],

        colors=['green', 'red'])
df_pred = pd.read_csv('../input/test.csv')

df_pred = df_pred[['PassengerId', 'Pclass', 'Sex', 'Parch', 'SibSp']]

df_pred['Sex'] = df_pred['Sex'].apply(lambda x: 0 if x == 'female' else 1)

df_pred['Survived'] = 1

df_pred.head()


for _, row in df_pred.iterrows():

    index = row['PassengerId']

    inputs = np.array([row[['Pclass', 'Sex', 'Parch', 'SibSp']].values])

    inputs = np.insert(inputs,0,1,axis=1) # Bias

    pred = predict(inputs, weights)

    if pred < 0.5:

        row['Survived'] = 0
df_pred.to_csv('./submission.csv', columns=['PassengerId', 'Survived'], index=False)