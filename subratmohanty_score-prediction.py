# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

from sklearn import preprocessing

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

data.head()
clean_data=data.copy()



arr=np.array(clean_data['math score'])



q3=np.quantile(arr,0.75)

q1=np.quantile(arr,0.25)

iqr=q3-q1

print(clean_data[clean_data['math score']<(q1-1.5*iqr)])

print(clean_data[clean_data['math score']>(q3+1.5*iqr)])
clean_data.isna().sum()
standard=preprocessing.StandardScaler()

scaled=standard.fit_transform(clean_data[['math score','reading score','writing score']])

scaled=pd.DataFrame(scaled,columns=['math score','reading score','writing score'])

clean_data[['math score','reading score','writing score']]=scaled

clean_data.head()
clean_data=pd.get_dummies(clean_data)

clean_data.head()
import seaborn as sns

plt.figure(figsize=(12,12))

sns.heatmap(clean_data.corr(),center=0,cmap='inferno',annot=True)
X=clean_data[['reading score','writing score','lunch_free/reduced','lunch_standard','gender_female','gender_male']]

#X=clean_data[['reading score','writing score','lunch_free/reduced','lunch_standard','test preparation course_completed','test preparation course_none']]

Y=clean_data['math score']



from sklearn.model_selection import train_test_split

X_tr,X_te,Y_tr,Y_te=train_test_split(X,Y,test_size=0.1,random_state=1)

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(X_tr,Y_tr)

coef=reg.coef_

y_pred=X_te['reading score']*coef[0]+X_te['writing score']*coef[1]+X_te['lunch_free/reduced']*coef[2]+X_te['lunch_standard']*coef[3]+X_te['gender_female']*coef[4]+X_te['gender_male']*coef[5]

plt.scatter(X_te['reading score'],y_pred,color='k',label='predicted')

plt.scatter(X_te['reading score'],Y_te,color='b',label='actual')

plt.legend()
import statsmodels.api as sm

X_te=sm.add_constant(X_tr)

model=sm.OLS(Y_tr,X_tr).fit()

model.summary()
from sklearn.metrics import r2_score

print("R2 score without regularization - test data= ",r2_score(y_pred,Y_te))
from sklearn.linear_model import Ridge,Lasso, ElasticNet



print("After reqularization - train data")

rid=Ridge(alpha=0.05)

rid.fit(X_tr,Y_tr)

y_reg=rid.predict(X_tr)

print("R2 score with Ridge regression=",r2_score(y_reg,Y_tr))

rid=Lasso(alpha=0.05)

rid.fit(X_tr,Y_tr)

y_reg=rid.predict(X_tr)

print("R2 score with Lasso regression=",r2_score(y_reg,Y_tr))

rid=ElasticNet(alpha=0.05)

rid.fit(X_tr,Y_tr)

y_reg=rid.predict(X_tr)

print("R2 score with Elastic Net regression=",r2_score(y_reg,Y_tr))