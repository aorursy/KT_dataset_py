import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_predict
%matplotlib inline

plt.style.use('seaborn-white')
train = pd.read_csv('../input/random-linear-regression/test.csv')

test = pd.read_csv('../input/random-linear-regression/test.csv')
plt.figure(figsize = (8,4))

plt.scatter(train['x'], train['y'])

plt.xlabel('X')

plt.ylabel('y')

plt.show()
X_train  = train[['x']]

y_train = train['y']

X_test = test[['x']]

y_test = test['y']
model = LinearRegression()

model.fit(X_train, y_train)
y_train_pred = cross_val_predict(model, X_train, y_train, cv = 10, n_jobs=-1)

y_pred = model.predict(X_test)
print(f'Accuracy of the model on Cross Validation : {(r2_score(y_train, y_train_pred) * 100):.1f} %')

print(f'Accuracy of the model on test set : {(r2_score(y_test, y_pred) * 100):.1f} %')

print('\n\nModel is Accurate')