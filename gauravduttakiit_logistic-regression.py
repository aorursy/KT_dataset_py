import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw_data = pd.read_csv(r'/kaggle/input/bank-data/Bank-data.csv')

raw_data.head()
data=raw_data.copy()

data= data.drop(['Unnamed: 0'],axis=1)

data['y'] = data['y'].map({'yes':1, 'no':0})

data.head()
data.describe()
data.columns
y = data['y']

x1 = data['duration']
x = sm.add_constant(x1)

reg_log = sm.Logit(y,x)

results_log = reg_log.fit()
results_log.summary()
# the odds of duration are the exponential of the log odds from the summary table

np.exp(0.0051)
# To avoid writing them out every time, we save the names of the estimators of our model in a list. 

estimators=['interest_rate','march','credit','previous','duration']



X1 = data[estimators]

y = data['y']
X = sm.add_constant(X1)

reg_logit = sm.Logit(y,X)

results_logit = reg_logit.fit()

results_logit.summary()
def confusion_matrix(data,actual_values,model):

    pred_values = model.predict(data)

    bins=np.array([0,0.5,1])

    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]

    accuracy = (cm[0,0]+cm[1,1])/cm.sum()

    return cm, accuracy
confusion_matrix(X,y,results_logit)
raw_data2 = pd.read_csv(r'/kaggle/input/bank-data-testing/Bank-data-testing.csv')

data_test = raw_data2.copy()

data_test = data_test.drop(['Unnamed: 0'], axis = 1)
data_test['y'] = data_test['y'].map({'yes':1, 'no':0})

data_test.head()
y_test = data_test['y']

X1_test = data_test[estimators]

X_test = sm.add_constant(X1_test)
confusion_matrix(X_test, y_test, results_logit)
confusion_matrix(X,y, results_logit)