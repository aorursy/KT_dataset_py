#Importing libraries



import numpy as np

import pandas as pd

from scipy import stats

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split , cross_val_score

import statsmodels.api as sm
#reading data



data=pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

data.head(5)
null =pd.isnull(data)

print(null)





#No null value
# seprating X and Y



dep = 'Salary'

X =data.drop(dep, axis = 1)

#print(X)

Y = data[dep]

#print(Y)
from sklearn import model_selection

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=5)
import seaborn as sns

sns.pairplot(data, kind='reg')
lm = sm.OLS(Y_train,X_train).fit()

print(lm.summary())
pred_train = lm.predict(X_train)

err_train = pred_train - Y_train
pred_test = lm.predict(Y_train)

err_test = pred_train - Y_train
#Actual vs predicted table

Err_table = pd.concat((Y_train, pred_train, err_train), axis = 1)

#print(Err_table.__class__)

Err_table.columns = ['Actuals','Predicted','Error']

print(Err_table)
index = np.arange(len(Err_table))

plt.plot(index,Err_table['Actuals'], color = 'b')

plt.plot(index,Err_table['Predicted'], color = 'g')

plt.xlabel('Age')

plt.ylabel('Salary')

plt.legend(loc = "upper center")

plt.show()