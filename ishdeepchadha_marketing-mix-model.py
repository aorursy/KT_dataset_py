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
df = pd.read_csv('/kaggle/input/advertising.csv/Advertising.csv')

df.head()
df.shape
df.describe()
df.drop('Unnamed: 0' , axis=1 , inplace=True)

df.head()
import matplotlib as plt

import seaborn as sns
corr = df.corr()
sns.heatmap(corr , xticklabels=corr.columns , yticklabels=corr.columns , annot=True , cmap=sns.diverging_palette(220,20,as_cmap=True))
sns.pairplot(df)
X = df.loc[:,df.columns!='sales']

y = df['sales']
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error as mae
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=.25,random_state=0)

model=RandomForestRegressor(random_state=1)

model.fit(train_X,train_y)

pred=model.predict(test_X)
feat_importances = pd.Series(model.feature_importances_,index=X.columns)

feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
import statsmodels.formula.api as sm

model = sm.ols(formula = "sales~TV+radio+newspaper",data=df).fit()

print(model.summary())
y_pred=model.predict()

labels = df['sales']

df_temp=pd.DataFrame({'Actual': labels, 'Predicted':y_pred})

df_temp.head()
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')



y1=df_temp['Actual']

y2=df_temp['Predicted']



plt.plot(y1, label = 'Actual')

plt.plot(y2, label = 'Predicted')

plt.legend()

plt.show()