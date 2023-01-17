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
import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt



import statsmodels.api as sm

from statsmodels.stats import diagnostic as diag

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
data=pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

data=pd.DataFrame(data)

data.head()
data['animal']=data['animal'].replace(to_replace ="acept", 

                 value =1) 

data['animal']=data['animal'].replace(to_replace ="not acept", 

                 value =0)



data['furniture']=data['furniture'].replace(to_replace ="furnished", 

                 value =1) 

data['furniture']=data['furniture'].replace(to_replace ="not furnished", 

                 value =0)
data.isnull().sum()
data.info()
data['floor'].value_counts()
data['floor'] = data['floor'].replace(['-','301'], 0)

data['floor'] = data['floor'].astype(int)

data['floor'].value_counts()
data.describe()
sp = data.loc[data.city=='São Paulo']

sp= sp.drop(['city'], axis=1)

rj = data.loc[data.city=='Rio de Janeiro']

rj=rj.drop(['city'], axis=1)

bh = data.loc[data.city=='Belo Horizonte']

bh=bh.drop(['city'], axis=1)

poa = data.loc[data.city=='Porto Alegre']

poa=poa.drop(['city'], axis=1)

cam = data.loc[data.city=='Campinas']

cam=cam.drop(['city'], axis=1)
sns.pairplot(bh)
## Mudando nome de colunas

bh.columns=['area','rooms','bathroom','parking_spaces','floor','animal','furniture','hoa','rent_amount','property_tax','fire_insurance','total']



## Descobrindo o aluguel mais caro da cidade

bh_rent= bh.sort_values("rent_amount",ascending=False).reset_index()

bh_rent=bh_rent.drop(['index'], axis=1)



## Analisando essas propriedades

bh_rent1=bh_rent.loc[0:34]

bh_rent1
bh_rent1.hist(bins=20,figsize=(20,10))
print(bh['animal'].value_counts())

print(bh['area'].value_counts())

print(bh['rooms'].value_counts())

print(bh['bathroom'].value_counts())

print(bh['parking_spaces'].value_counts())

print(bh['floor'].value_counts())

print(bh['furniture'].value_counts())

print(bh['hoa'].value_counts())
std_dev = 3

bh_remove = bh[(np.abs(stats.zscore(bh)) < float(std_dev)).all(axis=1)]
bh_cor1=bh_remove.corr()

sns.heatmap(bh_cor1, xticklabels=bh_cor1.columns, yticklabels=bh_cor1.columns, cmap='RdBu')
# Definindo dois dataframes, um antes de eliminar variáveis, e outro depois

bh_before = bh_remove

bh_after = bh_remove.drop(['hoa','rent_amount','property_tax','fire_insurance'], axis = 1)



X1 = sm.tools.add_constant(bh_before)

X2 = sm.tools.add_constant(bh_after)



# Criando uma series para ambos

series_before = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)

series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)



# mostrando as series

print('Antes')

print('-'*100)

display(series_before)



print('Depois')

print('-'*100)

display(series_after)
## Gráfico

sns.pairplot(bh_after)
# define our input variable (X) & output variable

X = bh_after.drop('total', axis = 1)

Y = bh_after[['total']]



# Split X and y into X_

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)



# create a Linear Regression model object

regression_model = LinearRegression()



# pass through the X_train & y_train data set

regression_model.fit(X_train, y_train)
# Separando o intercepto e o coeficiente

intercept = regression_model.intercept_[0]

coefficent = regression_model.coef_[0][0]



print("The intercept for our model is {:.5}".format(intercept))

print('-'*100)



# loop through the dictionary and print the data

for coef in zip(X.columns, regression_model.coef_[0]):

    print("The Coefficient for {} is {:.4}".format(coef[0],coef[1]))
# define our intput

X2 = sm.add_constant(X)



# create a OLS model

model = sm.OLS(Y, X2)



# fit the data

est = model.fit()



# print out a summary

print(est.summary())
price=-1192.89+4.0463*200+375.3567*4+972.98*4+451.79*4+142.79*2-264.15*1+887.14*1

price