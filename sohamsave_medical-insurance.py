
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
med_df=pd.read_csv('../input/insurance.csv')
med_df.head()
med_df.info()
med_df.describe()
sns.heatmap(med_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
Male=pd.get_dummies(med_df['sex'],drop_first=True)
Smoker=pd.get_dummies(med_df['smoker'],drop_first=True)
Area=pd.get_dummies(med_df['region'],drop_first=True)
med_df=pd.concat([med_df,Male,Smoker,Area],axis=1)
med_df.head()
med_df.drop(['sex','smoker','region'],axis=1,inplace=True)
med_df.head()
sns.jointplot(data=med_df,x='age',y='charges')
sns.jointplot(x='age',y='charges',data=med_df,kind='hex')

sns.barplot(x='age',y='charges',hue='male',data=med_df)
plt.figure(figsize=(20,5))
sns.barplot(x='age',y='children', data=med_df, hue='male', palette='RdBu_d', ci=None)
from sklearn.model_selection import train_test_split
X=med_df.drop(['charges'],axis=1)
y=med_df['charges']
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)
predictions=lm.predict(X_test)
sns.scatterplot(y_test,predictions)
from sklearn import metrics


print('MAE= ', metrics.mean_absolute_error(y_test, predictions))
print('MSE= ', metrics.mean_squared_error(y_test, predictions))
print('RMS= ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


sns.distplot(y_test-predictions,bins=30)
coef_df= pd.DataFrame(lm.coef_,X.columns,columns=['coefficient'])
coef_df
from sklearn.metrics import r2_score
print(r2_score(y_test,predictions))
from sklearn.metrics import explained_variance_score
print(explained_variance_score(y_test,predictions))
