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
df1=pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
df1

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.pairplot(data=df1)

sns.jointplot(x='GRE Score',y='Chance of Admit ',data=df1)

from  sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
x=df1[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
y=df1['Chance of Admit ']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=1)
lm=LinearRegression()
lm.fit(x_train,y_train)
g=pd.DataFrame(lm.coef_,x.columns,columns=['coeff'])
pred=lm.predict(x_test)
sns.scatterplot(y_test,pred)
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
metrics.explained_variance_score(y_test,pred)
metrics.r2_score(y_test,pred)
x=np.sqrt(metrics.mean_squared_error(y_test,pred))
x