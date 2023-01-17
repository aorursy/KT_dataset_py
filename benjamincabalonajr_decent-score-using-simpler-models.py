import os

import numpy as np 

import pandas as pd 

import seaborn as sns

import warnings

import matplotlib.pyplot as plt

from ipywidgets import interact

from scipy.stats import pearsonr



%matplotlib inline

warnings.filterwarnings('ignore')

sns.set(style='ticks')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/generalization-competition/train.csv')

test = pd.read_csv('../input/generalization-competition/test.csv')

train.head()
def plot(column):

    g = sns.jointplot(x=column, y="Revenue", data=train, kind='reg',joint_kws={'line_kws':{'color':'cyan'}}) 

    g.annotate(pearsonr)

    plt.show()



interactive = interact(plot,column=train.columns)
plot('Ad Responses')
plot('Location Participation')
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn import metrics
X = train.drop('Revenue',axis=1)

y = train['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = LinearRegression(fit_intercept=False)

model.fit(X_train,y_train)
cv = np.mean(cross_val_score(model, X_train, y_train, cv=5,scoring='neg_mean_squared_error'))

print ("Model RMSE with 5 cross validation :",np.sqrt(-cv))

y_predict_test = model.predict(X_test)

score_test = np.sqrt(metrics.mean_squared_error(y_test, y_predict_test))

print('Test RMSE',score_test)
X = train[['Ad Responses']]

y = train['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LinearRegression(fit_intercept=False)

model.fit(X_train,y_train)

cv = np.mean(cross_val_score(model, X_train, y_train, cv=5,scoring='neg_mean_squared_error'))

print ("Model RMSE with 5 cross validation :",np.sqrt(-cv))

y_predict_test = model.predict(X_test)

score_test = np.sqrt(metrics.mean_squared_error(y_test, y_predict_test))

print('Test RMSE',score_test)
X = train[['Ad Responses','Cost']]

y = train['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LinearRegression(fit_intercept=False)

model.fit(X_train,y_train)

cv = np.mean(cross_val_score(model, X_train, y_train, cv=5,scoring='neg_mean_squared_error'))

print ("Model RMSE with 5 cross validation :",np.sqrt(-cv))

y_predict_test = model.predict(X_test)

score_test = np.sqrt(metrics.mean_squared_error(y_test, y_predict_test))

print('Test RMSE',score_test)
model.fit(X[['Ad Responses']],y)

pred = model.predict(test[['Ad Responses']])
predictions= test[['index']].copy()

predictions['Revenue'] = pred

predictions.to_csv('submission.csv', index=False)
prediction = [train['Revenue'].mean()]*len(y_test)

score_test = np.sqrt(metrics.mean_squared_error(y_test, prediction))

print('RMSE:',score_test)
!jupyter nbconvert --execute --to html __notebook_source__.ipynb