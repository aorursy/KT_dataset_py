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
dataset=pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')

dataset.head()
dataset.isnull().sum()

dataset['State / Union territory (UT)'].value_counts()

dataset['Region'].value_counts()

dataset.head()
dataset.rename(columns = {'State / Union territory (UT)':'state'}, inplace = True)

dataset.rename(columns = {'Area (km2)':'area'}, inplace = True)

dataset.rename(columns = {'National Share (%)':'share'},inplace=True)

dataset.head()

#renaming some of the columns
#dataset.isnull().sum()

import matplotlib.pyplot as plt

import seaborn as sn

plt.figure(figsize=(14,6))

dataset.groupby('state')['share'].count().sort_values(ascending=False).head(4).plot.bar(color='orange')

#Regionwise shares comparision

plt.figure(figsize=(14,6))

dataset.groupby('Region')['share'].count().sort_values(ascending=False).head(5).plot.bar(color='red')
plt.figure(figsize=(14,6))

sn.regplot(x=dataset.area,y=dataset.share,color='violet')

#Maybe a linear regression fits the accuracy
X=dataset.loc[:,['area']]

Y=dataset.loc[:,['share']]

X.head()

Y.head()

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

X=scale.fit_transform(X)

Y=scale.fit_transform(Y)

X=pd.DataFrame(X)

Y=pd.DataFrame(Y)

X.head()

Y.head()

#X=np.reshape(-1,1)

#Y=np.reshape(-1,1)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(xtrain,ytrain)

ypred=regressor.predict(xtest)

from sklearn.metrics import r2_score

print(r2_score(ypred,ytest)*100)

from sklearn.model_selection import cross_val_score

score=cross_val_score(regressor,xtrain,ytrain)

print(score.mean()*100)

#cross val score 