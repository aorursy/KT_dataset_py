# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/startup-logistic-regression/50_Startups.csv')

dataset.info()
dataset.describe()
#scatterplot

sns.set()

cols = ['Profit', 'R&D Spend', 'Marketing Spend', 'Administration']

sns.pairplot(dataset[cols], height = 2.5)

plt.show();
%matplotlib inline

plt.figure(figsize=(12,10))

cor = dataset.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()

corr_Profit=cor["Profit"].sort_values(ascending=False)

print(corr_Profit)
Q1 = dataset.quantile(0.25)

Q3 = dataset.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
print(dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))
dataset_outl = dataset[~((dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]

dataset_outl.shape
#Normality

#histogram and normal probability plot

from scipy.stats import norm

from scipy import stats

sns.distplot(dataset['Profit'], fit=norm);

fig = plt.figure()

res = stats.probplot(dataset['Profit'], plot=plt)
# splitting the dataset into  independent variable X and dependent(target) y

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 4].values
# Encoding categorical data

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer

 

ct = ColumnTransformer([('encoder', OneHotEncoder(),[3])], remainder='passthrough')

 

X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:]
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

# drop the 1 column in X

RFE_regressor = LinearRegression()

#Initializing RFE model

rfe = RFE(RFE_regressor, 2)# random number(2)

#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model

RFE_regressor.fit(X,y)

print(rfe.support_)

print(rfe.ranking_)
#-------before calc high score it says all true ----



from sklearn.model_selection  import train_test_split

from sklearn.linear_model import LinearRegression

#no of features

nof_list=np.arange(1,4)            

high_score=0

#Variable to store the optimum features

nof=0           

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

    model = LinearRegression()

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe,y_train)

    score = model.score(X_test_rfe,y_test)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

#Initializing RFE model

rfe = RFE(RFE_regressor, 3)

#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model

RFE_regressor.fit(X,y)

print(rfe.support_)

print(rfe.ranking_)

RFE_features=X[:,[0,1,2]] # using the features with only True values
# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
print("score: ",regressor.score(X_train,y_train))

print("Model slope:    ", regressor.coef_)

print("Model intercept:", regressor.intercept_)
# Predicting the Test set results

y_pred = regressor.predict(X_test)

print("score: ",regressor.score(X_test,y_test))

print("Model slope:    ", regressor.coef_)

print("Model intercept:", regressor.intercept_)