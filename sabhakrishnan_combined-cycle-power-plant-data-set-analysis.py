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
df=pd.read_csv('/kaggle/input/airpressure/Folds5x2_pp.csv')

df.head()
df.shape
# Importing Libraries. 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import model_selection

from sklearn import linear_model

import seaborn as sns
#summary statistics:

df.info()

df.describe()
fig, axes = plt.subplots(2,2)

for i,el in enumerate(list(df.columns.values)[:-1]):

    a = df.boxplot(el, ax=axes.flatten()[i])

#checking for the distribution using violin plot:

for i in range(len(df.columns)):

    sns.distplot(df.iloc[:,i])

    plt.show()
#checking linearity:

sns.pairplot(df)

print(df.corr(method="spearman"))



col=df.columns

print(col)
#normalizing the data:

df_nor=preprocessing.normalize(df)

print(df_nor)

df_nor=pd.DataFrame(df_nor)

print(df_nor)

sns.pairplot(df_nor)

df_nor.corr()

df_nor.columns = df.columns

df_nor.head()

x=df_nor.iloc[:,0:4]

y=df_nor.iloc[:,4]
#splitting train set:

X_train,X_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#fitting linear model:

lm=linear_model.LinearRegression()

model=lm.fit(X_train,y_train)

pred=lm.predict(X_train)

print(pred)

print(model.coef_)

print(model.intercept_)
#checking accuracy:

from sklearn.metrics import r2_score

print(r2_score(pred,y_train))

predd = lm.predict(X_test)

print(r2_score(predd,y_test))
sns.regplot(x=predd, y=y_test, lowess=True, line_kws={'color': 'red'})

plt.title('Observed vs. Predicted Values', fontsize=16)
