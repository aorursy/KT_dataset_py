# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/california-housing-value/housing.csv")
df.head()
df.isnull().sum()
#so the attribute total_bedrooms have 207 NAN values
df.shape
#replacing the missing data of total bedrooms as 3 where is NAN
df=df.replace(np.NAN,1000)
#lets check again
df.isnull().sum()
#now our data has no missing value and we are safe to proceed
#checking for non numerical value
df.head()
ocean_dist=pd.get_dummies(df['ocean_proximity'],drop_first=True)
df=pd.concat([df,ocean_dist],axis=1)
df.drop(['ocean_proximity'],axis=1,inplace=True)
df.head()
X=df.drop('median_house_value',axis=1)
y=df['median_house_value']
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
linear_model=LinearRegression()
def run_reg(regressor, X_train,X_test,y_train,y_test):
    regressor.fit(X_train,y_train)
    predictions=regressor.predict(X_test)
    for i in predictions:
        if i<15000:
            i=15000
        if i>500000:
            i=500000
    MAE=mean_absolute_error(y_test,predictions)
    r2=r2_score(y_test,predictions)
    plot1=plt.scatter(predictions,y_test,edgecolors='black')
    plt.title('MAE={}'.format(MAE))
    coefficients=regressor.coef_
    print(coefficients)
    a=[]
    for i in range(len(coefficients)):
        a.append(i)
    plt.show()
    plt.scatter(a,coefficients,color='red')
    plt.title('Coeffiecients')
    return MAE
MAE=run_reg(linear_model,X_train,X_test,y_train,y_test)
