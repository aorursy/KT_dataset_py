import numpy as np

import pandas as pd



data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# data.head()



test['y'] = 0

data = data.append(test, ignore_index=True)

data.head()
data['X0'], ux0 = pd.factorize(data.X0)

data['X1'], ux1 = pd.factorize(data.X1)

data['X2'], ux2 = pd.factorize(data.X2)

data['X3'], ux3 = pd.factorize(data.X3)

data['X4'], ux4 = pd.factorize(data.X4)

data['X5'], ux5 = pd.factorize(data.X5)

data['X6'], ux6 = pd.factorize(data.X6)

data['X8'], ux8 = pd.factorize(data.X8)



# X = data[data.columns.difference(['y'])]

# Y = data['y']

# X = data.drop(['ID','y'], axis = 'columns').apply(lambda x: pd.factorize(x)[0])

data.head()
from sklearn.model_selection import train_test_split

from sklearn import preprocessing



Y = data.loc[data['y'] != 0].y

X = data.loc[data['y'] != 0].drop(['y'], axis = 'columns')



# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=777)
from sklearn.model_selection import cross_val_predict

from sklearn import linear_model

import matplotlib.pyplot as plt



lr = linear_model.LinearRegression()

predicted = cross_val_predict(lr, X, Y, cv=100)



fig, ax = plt.subplots()

ax.scatter(Y, predicted, edgecolors=(0, 0, 0))

ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
from sklearn.metrics import r2_score



print(r2_score(Y, predicted))
test = data.loc[data['y'] == 0]

lr.fit(X,Y)



x_test = data.loc[data['y'] == 0].drop(['y'], axis = 'columns')

test['y'] = lr.predict(x_test)

result = pd.concat([test.ID, test['y']], axis=1, keys=['ID', 'y'])

result.set_index('ID').to_csv('result.csv', sep = ',')