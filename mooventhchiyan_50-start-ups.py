# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

data= pd.read_csv('../input/50_Startups.csv')
# Understand the columns

data.head(5)
# Checking Null values

data.isnull().sum()
# Checking outlier

dd=['R&D Spend', 'Administration', 'Marketing Spend','Profit']

for i in dd:

    sns.boxplot(data[i])

    plt.show()
# Finding the distribution

plt.scatter(data["State"],data["Profit"],label="profit")

plt.xlabel("state")

plt.ylabel("profit")

plt.legend(loc='upper left')

plt.show()
plt.scatter(data["Administration"],data["Profit"],label="profit")

plt.xlabel("Administration_cost")

plt.ylabel("profit")

plt.legend(loc='upper left')

plt.show()
plt.scatter(data["R&D Spend"],data["Profit"],label="profit")

plt.xlabel("R&D Spend")

plt.ylabel("profit")

plt.legend(loc='upper left')

plt.show()
plt.scatter(data["Marketing Spend"],data["Profit"],label="profit")

plt.xlabel("Marketing Spend")

plt.ylabel("profit")

plt.legend(loc='upper left')

plt.show()
# Estimating the total counts

data["State"].value_counts()
# Creating the dummy variable,dropping the first row to avoid dummy varaible trap

dummies= pd.get_dummies(data, columns=["State"],drop_first= True)

dummies
# Seprating the target(dependent) variable and the independent varaible

X=data.drop("Profit",axis=1)

y= data["Profit"]
# Using seaborn heat map to visualize the correlation of data

import seaborn as sns

plt.figure(figsize=(6,6))

cor = data.corr()

sns.heatmap(cor,annot=True)

plt.show()
# Estimating the high correlation factors in the data

cor_tar= abs(cor["Profit"])

highcor= cor_tar[cor_tar >0.5]

highcor
# Immporting the statistical model to start with the OLS model

import statsmodels.api as sm
# Copy of data to work on statitical OLS

df= data.copy()
# Making the dummies and splitting the target

dummies= pd.get_dummies(df, columns=["State"],drop_first= True)

X=dummies.drop("Profit",axis=1)

y=dummies["Profit"]
# Adding a constant since the statistical model don't learn by default

X_constant = sm.add_constant(X)

model = sm.OLS(y, X_constant).fit()

# Prediction using the X_constant

predictions = model.predict(X_constant)

model.summary()
ax=sns.residplot(predictions,model.resid,lowess=True)

ax.set(xlabel = 'Fitted value', ylabel = 'Residuals', title = 'Residual vs Fitted Plot \n')

plt.show()

# It is observed that the line stays close to the mean so its significanty normal
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(X_test.shape)
# Importing linear regression and fit(learning) the train data

from sklearn.linear_model import LinearRegression

lr= LinearRegression()

lr.fit(X_train,y_train)
# with the help of the learning predicting the test data 

y_pred= lr.predict(X_test)

y_pred
y_test
# Checking the accuracy of the Test and train

R_score_train = lr.score(X_train,y_train)

R_score_test = lr.score(X_test,y_test)

print('R^2 score for train:',R_score_train)

print('R^2 score for train:',R_score_test)
# To find which of the feature performs the best in the estimation

# Lasso Regression(type of regularisation)- we use it to visualize the effect on the target varaible

from sklearn.linear_model import LassoCV

reg = LassoCV()

reg.fit(X, y)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# Visualising the effect

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "bar")

plt.title("Feature importance using Lasso Model")
# It can also be determined using the backward elimantion with the help of level of P-value of thr various coefficients