# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_path = '../input/ncc_consolidated.csv'

ncc = pd.read_csv(data_path)

ncc
#drop unneccesary columns

inputs = ncc.drop(['ID','Date','Government Agency','Contract Description','Procurement Method','Currency Unit','Jamaican Equivalent','Comments','Additional Comments','Column 13'],axis='columns')

inputs.head()
from sklearn.preprocessing import LabelEncoder
le_Fund = LabelEncoder()

le_Contractor = LabelEncoder()
#transform Fund and Contractor columns to str to remove errors

inputs['Fund'] = inputs['Fund'].astype('str')

inputs['Contractor'] = inputs['Contractor'].astype('str')

inputs.dtypes
# now we can transform fund column to numbers

inputs['Fund'] = le_Fund.fit_transform(inputs['Fund'])

inputs.head()
# now we can transform Contractor column to numbers

inputs['Contractor'] = le_Fund.fit_transform(inputs['Contractor'])

inputs.head()
#show the correlation of the table

correl = inputs.corr()

correl
fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correl,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(inputs.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(inputs.columns)

ax.set_yticklabels(inputs.columns)

plt.show()

#key: Dark red is perfect positive correlation, Dark blue is perfect negative correlation
from sklearn import linear_model

import statsmodels.api as sm
#drop target variable from table and assign to its own variable

inputs_n = inputs.drop('Contractor',axis='columns')

target = inputs['Contractor']

target.head()
# Regression Analysis



regr = linear_model.LinearRegression()

regr.fit(inputs_n, target)
print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
#checks accuracy score of the developed model from 0-1

model.score(inputs_n,target)
inputs_n.head()
#just to ensure transformation is proper

inputs_n['Dollar Amount'] = le_Fund.fit_transform(inputs['Dollar Amount'])

inputs.head()
# with statsmodels

X = sm.add_constant(inputs_n) # adding a constant

 

model = sm.OLS(target, inputs_n).fit()

predictions = model.predict(inputs_n) 

 

print_model = model.summary()

print(print_model)

# low p value for Fund and Dollar amount is saying that the independent variables are statistically

#significant to impact the grades



#rsquared = 0.668
plt.scatter(inputs['Fund'],inputs['Contractor'])
plt.scatter(inputs['Dollar Amount'],inputs['Contractor'])
pd.scatter_matrix(inputs_n.loc[:,'Fund':'Dollar Amount'])
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn import linear_model
x_train, x_test, y_train, ytest = train_test_split(inputs_n,target,test_size=0.2)
import statsmodels.formula.api as sm

results = sm.ols(formula='target ~ Fund', data=inputs_n).fit()

Y_pred = results.predict(inputs_n)

residual = target.values-Y_pred

residual
plt.scatter(inputs_n['Fund'],residual)

plt.xlabel("Fund - a predictor")

plt.ylabel("residual")

plt.show()
#number of data that will be used to train randomly

len(x_train)
len(x_test)
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(x_train,y_train)
clf.predict(x_test)
clf.score(x_test,ytest)
#model predicting if project is funded by pcj, with value around 3000,

#then the contract will be awarded to Contractor numbered as 234 



model.predict([[163,3000]])
#locate index of contract number to match to original table

list(target).index(234)
# index 735 shows funded by pcj and contract was awarded to Austins Haulage Company

ncc.head()