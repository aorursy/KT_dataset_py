# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 1) Import all Library that will be used

%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



import statsmodels.formula.api as smf



from scipy import stats



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import scale

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import linear_model, svm, gaussian_process

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score# 1) Data treatment and cleaning
#Let's do 03 Submitions:

#01) Simple way - no data treatment;

#02) Removing outliers and Log function;

#03) Removing outliers and Square Root function;
#df_train_original = pd.read_csv('train-House.csv')

df_train_original = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test_original = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



df_train = df_train_original

df_test = df_test_original
#01) Simple way - no data treatment:



all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],

                      df_test.loc[:,'MSSubClass':'SaleCondition']))



# Get_Dummies para transformar categoricos em Numéricos 

all_data = pd.get_dummies(all_data)



# Substitui os campos nulos pelas médias da coluna em questão

all_data = all_data.fillna(all_data.mean())

#all_data = all_data.fillna(0)



#Cria Matriz X_train utilizando a Matriz com todos os dados all_data: do inicio da matriz (:) até o fim  da matriz df_train.shape[0]

X_train = all_data[:df_train.shape[0]]



#Cria Matriz X_test utilizando a Matriz com todos os dados all_data: a partir do último registro matriz df_train.shape[0], ou seja, todos os registros que não estiverem em df_train

X_test = all_data[df_train.shape[0]:]



# Cria o y, ou seja, o que será previsto, apenas com o campo "Survived"

y = df_train.SalePrice



#Validation function

n_folds = 5
# * *  * *  * *  * *  * *  * *  * *  * *  * *  * * 

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y)



LR_yhat_train = logreg.predict(X_train)

LR_yhat_test = logreg.predict(X_test)



yhat_LR = LR_yhat_test

print (yhat_LR)
# * *  * *  * *  * *  * *  * *  * *  * *  * *  * * 

# Gerando um CSV para o resultado obtido com o Gradiente Descendente:

df_test_LR = df_test

df_test_LR['SalePrice'] = yhat_LR

df_test_LR = df_test_LR.drop(df_test_LR.columns[1:80], axis=1)

df_test_LR.to_csv('House_LR.csv', index = False)



Script01 = yhat_LR


#02) Removing outliers and Log function;



df_train = df_train_original

df_test = df_test_original

# Y or Label field: SalePrice

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'] , plot=plt)

plt.show()



df_train.plot.scatter(x='SalePrice', y='Id');



sns.distplot((df_train['SalePrice']) , fit=stats.norm);
sns.distplot((df_train['SalePrice']) , fit=stats.norm);



(mu, sigma) = stats.norm.fit(np.log(df_train['SalePrice']))

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')
print ('Before Transformation: df_train.shape', df_train.shape, 'Content: ', df_train['SalePrice'])

df_train = df_train[df_train['SalePrice'] < 350000]

print ('After Transformation: df_train.shape', df_train.shape, 'Content: ', df_train['SalePrice'])



df_train.plot.scatter(x='SalePrice', y='Id');
print (' # # # Before Log: ', df_train['SalePrice'])

df_train["SalePrice"] = np.log(df_train["SalePrice"])

print (' # # # After Log: ', df_train['SalePrice'])
sns.distplot((df_train['SalePrice']) , fit=stats.norm);



(mu, sigma) = stats.norm.fit(np.log(df_train['SalePrice']))

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')
all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],

                      df_test.loc[:,'MSSubClass':'SaleCondition']))



# Get_Dummies para transformar categoricos em Numéricos 

all_data = pd.get_dummies(all_data)



# Substitui os campos nulos pelas médias da coluna em questão

all_data = all_data.fillna(all_data.mean())

#all_data = all_data.fillna(0)



#Cria Matriz X_train utilizando a Matriz com todos os dados all_data: do inicio da matriz (:) até o fim  da matriz df_train.shape[0]

X_train = all_data[:df_train.shape[0]]



#Cria Matriz X_test utilizando a Matriz com todos os dados all_data: a partir do último registro matriz df_train.shape[0], ou seja, todos os registros que não estiverem em df_train

X_test = all_data[df_train.shape[0]:]



# Cria o y, ou seja, o que será previsto, apenas com o campo "Survived"

y = df_train.SalePrice



#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)

    rmse= np.sqrt(-cross_val_score(model, df_train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
print (' # # # Before Mult: ', y)

y = y * 10000

print (' # # # Midl: ', y)

y = y.astype(np.int)

print (' # # # After Mult: ', y)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y)



LR_yhat_train = logreg.predict(X_train)

LR_yhat_test = logreg.predict(X_test)



yhat_LR = LR_yhat_test

print (yhat_LR)
print (' # # # Before Div: ', yhat_LR)

yhat_LR = yhat_LR.astype(np.float)

yhat_LR = yhat_LR / 10000

print (' # # # After Div: ', yhat_LR)



print ('2 # # # Before Exp: ', yhat_LR)

yhat_LR = np.exp(yhat_LR)

print ('2 # # # After Exp: ', yhat_LR)
# Gerando um CSV para o resultado obtido com o Gradiente Descendente:

df_test_LR = df_test

df_test_LR['SalePrice'] = yhat_LR

df_test_LR = df_test_LR.drop(df_test_LR.columns[1:80], axis=1)

df_test_LR.to_csv('House_LR_Log.csv', index = False)



Script02 = yhat_LR
#03) Removing outliers and Square Root function;



df_train = df_train_original

df_test = df_test_original
# Y or Label field: SalePrice

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'] , plot=plt)

plt.show()



df_train.plot.scatter(x='SalePrice', y='Id');



sns.distplot((df_train['SalePrice']) , fit=stats.norm);
sns.distplot((df_train['SalePrice']) , fit=stats.norm);



(mu, sigma) = stats.norm.fit(np.log(df_train['SalePrice']))

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')
print ('Before Transformation: df_train.shape', df_train.shape, 'Content: ', df_train['SalePrice'])

df_train = df_train[df_train['SalePrice'] < 350000]

print ('After Transformation: df_train.shape', df_train.shape, 'Content: ', df_train['SalePrice'])



df_train.plot.scatter(x='SalePrice', y='Id');
print (' # # # Before Log: ', df_train['SalePrice'])

df_train["SalePrice"] = np.sqrt(df_train["SalePrice"])

print (' # # # After Log: ', df_train['SalePrice'])
sns.distplot((df_train['SalePrice']) , fit=stats.norm);



(mu, sigma) = stats.norm.fit(np.log(df_train['SalePrice']))

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')
all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],

                      df_test.loc[:,'MSSubClass':'SaleCondition']))



# Get_Dummies para transformar categoricos em Numéricos 

all_data = pd.get_dummies(all_data)



# Substitui os campos nulos pelas médias da coluna em questão

all_data = all_data.fillna(all_data.mean())

#all_data = all_data.fillna(0)



#Cria Matriz X_train utilizando a Matriz com todos os dados all_data: do inicio da matriz (:) até o fim  da matriz df_train.shape[0]

X_train = all_data[:df_train.shape[0]]



#Cria Matriz X_test utilizando a Matriz com todos os dados all_data: a partir do último registro matriz df_train.shape[0], ou seja, todos os registros que não estiverem em df_train

X_test = all_data[df_train.shape[0]:]



# Cria o y, ou seja, o que será previsto, apenas com o campo "Survived"

y = df_train.SalePrice



#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)

    rmse= np.sqrt(-cross_val_score(model, df_train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
print (' # # # Before Mult: ', y)

y = y * 100

print (' # # # Midl: ', y)

y = y.astype(np.int)

print (' # # # After Mult: ', y)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y)



LR_yhat_train = logreg.predict(X_train)

LR_yhat_test = logreg.predict(X_test)



yhat_LR = LR_yhat_test

print (yhat_LR)
print (' # # # Before Div: ', yhat_LR)

yhat_LR = yhat_LR.astype(np.float)

yhat_LR = yhat_LR / 100

print (' # # # After Div: ', yhat_LR)



print ('2 # # # Before Exp: ', yhat_LR)

yhat_LR = yhat_LR * yhat_LR

print ('2 # # # After Exp: ', yhat_LR)
# Gerando um CSV para o resultado obtido com o Gradiente Descendente:

df_test_LR = df_test

df_test_LR['SalePrice'] = yhat_LR

df_test_LR = df_test_LR.drop(df_test_LR.columns[1:80], axis=1)

df_test_LR.to_csv('House_LR_sqrt.csv', index = False)



Script03 = yhat_LR
print ('Model 01: ', Script01)

print ('Model 02: ', Script02)

print ('Model 03: ', Script03)