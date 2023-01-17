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
import pylab

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
df=pd.read_csv('/kaggle/input/crime-in-berlin-2012-2019/Berlin_crimes.csv')
sns.distplot(df['Drugs'])
sns.heatmap(df.corr())
df.columns
sns.lineplot(df.Year,df.Drugs)
years={}

for index in df.index:

    key = df['Year'][index]

    try:

        if key not in years.keys(): 

            years[df['Year'][index]]=[]

        years[df['Year'][index]].append(df['Drugs'][index])

    except:

        years[df['Year'][index]]=df['Drugs'][index]       

for y in years:

    years[y]=sum(years[y])/len(years[y])
xVals,yVals=[],[]

for e in years:

    xVals.append([e])

    yVals.append(years[e])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xVals, yVals,test_size=0.4, random_state=0)
from sklearn.linear_model import LinearRegression
lrmodel=LinearRegression()

lrmodel.fit(X_train,y_train)
predictions=lrmodel.predict(X_test)
plt.scatter(y_test,predictions)
from sklearn import metrics
print("MEA: " ,metrics.mean_absolute_error(y_test,predictions))

print("MSE: " ,metrics.mean_squared_error(y_test,predictions))

print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_test,predictions)))
def rSquared(observed, predicted):

    error = ((predicted - observed)**2).sum()

    meanError = error/len(observed)

    return 1 - (meanError/np.var(observed))



print(rSquared(y_test,predictions))