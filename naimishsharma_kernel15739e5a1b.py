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
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head()
df = df.drop(['EU_Sales','NA_Sales','JP_Sales','Other_Sales'],axis=1)
df.head()
import seaborn as sn
sn.lineplot(x=df.Year,y=df.Global_Sales)
sn.boxplot(x=df.Global_Sales)
sn.scatterplot(x=df.Year,y=df.Global_Sales,hue = df.Genre)
sn.distplot(df.Global_Sales,bins=5)
df.isnull().sum()
df.info()
df.Year.value_counts()

df.Publisher.value_counts()
df.Publisher.fillna('Electronic Arts',inplace=True)

df.Year.fillna(2009.0,inplace=True)
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

df.info()
df['Name']=label.fit_transform(df['Name'])

df['Platform']=label.fit_transform(df['Platform'])

df['Genre']=label.fit_transform(df['Genre'])

df['Publisher']=label.fit_transform(df['Publisher'])
df.Global_Sales.value_counts()
from sklearn.model_selection import train_test_split

train, test=train_test_split(df, test_size=0.1,random_state=1)
def data_split(df):

    x=df.drop(['Global_Sales'],axis=1)

    y=df['Global_Sales']

    return x,y

x_train,y_train=data_split(train)

x_test,y_test=data_split(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression



log_model=LinearRegression()

log_model.fit(x_train, y_train)

prediction=log_model.predict(x_test)

score= accuracy_score(y_test, prediction)

print(score)