import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dt_1 = pd.read_csv('/kaggle/input/boston-housing-dataset/train.csv')

print("Print first 5 subjects")

print(dt_1.head())

print("")

print("Basic descriptive statistics for all features")

print(dt_1.describe())

print("")

print("Feaure attributes")

print(dt_1.info())
print(plt.hist(dt_1['AGE']))
corr = dt_1.corr()

#print(corr)
corr.style.background_gradient(cmap='coolwarm')
abs_corr = np.abs(corr['MEDV'])

print(abs_corr.sort_values(ascending=False).head(7))
from sklearn import linear_model

from sklearn.model_selection import train_test_split



#define model

Llm = linear_model.Lasso(alpha = 0.1)
y = dt_1['MEDV']

X = dt_1.iloc[:, 0:13]



#Spllit the 406 data into 80% training, 20% testing

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8)
Llm.fit(train_X, train_y)

prdt = Llm.predict(val_X)
#check the coefficient

#print(Llm.coef_)

X_corr = abs_corr.iloc[0:13]

coeff = pd.DataFrame(Llm.coef_)

#pd.concat([X_corr, coeff], axis=1)

print(coeff)
#apply mean absolute error and mean squared error to evaluate performance

from sklearn.metrics import mean_absolute_error, mean_squared_error



print("Mean absolute error:")

print(mean_absolute_error(prdt, val_y))

print("Mean squared error:")

print(mean_squared_error(prdt, val_y))
from sklearn.linear_model import LassoCV

#define model

Lcvlm = LassoCV(cv=5, random_state=0)

CVreg = Lcvlm.fit(X, y)

prdt_cv = Lcvlm.predict(val_X)



print("Lasso CV Mean absolute error:")

print(mean_absolute_error(prdt_cv, val_y))

print("Lasso CV Mean squared error:")

print(mean_squared_error(prdt_cv, val_y))
exam_test = pd.read_csv('/kaggle/input/boston-housing-dataset/test.csv')

#print(exam_test.iloc[:, 1:])

result_Lcv = Lcvlm.predict(exam_test.iloc[:, 1:])

#print(result_Lcv)



result_Lcv_submit = pd.DataFrame({"ID": exam_test['ID'], "MEDV":result_Lcv})

print(result_Lcv_submit.head())



pd.DataFrame(result_Lcv_submit).to_csv("submit_SJ.csv", index=False)