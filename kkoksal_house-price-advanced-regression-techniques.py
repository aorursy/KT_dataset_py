# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np 

from sklearn import datasets, linear_model, metrics 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

target=train.filter(["SalePrice"],axis=1)

print(train.shape)

train_temp=train.drop(["SalePrice"], axis=1)

df=pd.concat([train_temp, test])

target.shape
train.shape
test.shape
df.shape
cols = df.columns

num_cols = df._get_numeric_data().columns

num_cols

liste=list(set(cols) - set(num_cols))
for c in liste:

    print(c)

    df[c] = df[c].astype("category").cat.codes



df.head()

df=df.replace(-1,np.NAN)



df.fillna((df.mean().round()), inplace=True)
train=df.iloc[0:1460,:]



train["SalePrice"]=target["SalePrice"]

test=df[1460:]

corr = train.corr()

fig , ax = plt.subplots(figsize = (7,7))

sns.heatmap(corr, vmax = 0.8, square = True)
plt.figure(figsize=(25, 20))



k = 65    

corr = train.corr()



cols = corr.nlargest(k,'SalePrice')['SalePrice'].index



cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=0.75)

hm = sns.heatmap(cm, cmap="YlGnBu",cbar=True, annot=True, square=True, fmt='.1f',

                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
X = train[cols].drop(["SalePrice"],axis=1)

y = train["SalePrice"] 

  

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 

                                                    random_state=1) 

  

reg = linear_model.LinearRegression() 

  

reg.fit(X_train, y_train) 

  

print('Coefficients: \n', reg.coef_) 

  

# variance score: 1 means perfect prediction 

print('Variance score: {}'.format(reg.score(X_test, y_test))) 

  

plt.style.use('fivethirtyeight') 

  

plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 

            color = "green", s = 10, label = 'Train data') 

  

print(reg.predict(X_test))



plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 

            color = "blue", s = 10, label = 'Test data') 

  

plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 

  

plt.legend(loc = 'upper right') 

  

plt.title("Residual errors") 

  

plt.show() 
from sklearn.metrics import r2_score

liste=[]

for i in cols:

    if(i!="SalePrice"):

        liste.append(i) 

test = test[liste]

reg.predict(test)
