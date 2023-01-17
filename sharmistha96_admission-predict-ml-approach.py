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
import pandas as pd

Admission_Predict = pd.read_csv("../input/admission-prediction/Admission_Predict.csv")
df=pd.read_csv("../input/admission-prediction/Admission_Predict.csv")

df.head()
df['Chance of admit class']=df['Chance of Admit '].apply(lambda x:1 if x>0.80 else 0)

df.head()
## Shape of data

print(' Shape of Data \n Rows :',df.shape[0],', Columns : ',df.shape[1])
## Checking for null values

missing_values = df.isnull().sum() * 100/len(df)

missing_values_df = pd.DataFrame({'Column_name':df.columns,'Missing_percent':missing_values})

missing_values_df
#iNFORMATION ABOUT THE DATA we can observe data doesnot have any Null value

df.info()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

import seaborn as sns

import warnings

from scipy import stats

warnings.filterwarnings('ignore')

# lets see the distribution for the target variable

print('Skewness of chance of admit : ',df['Chance of Admit '].skew())

plt.figure(figsize = (10,5))

sns.distplot(df['Chance of Admit '],kde = True,color = 'g',fit = stats.norm)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

rs = stats.probplot(df['Chance of Admit '],plot = ax)

plt.show()
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',

       'LOR ', 'CGPA', 'Research']

X = df[features]

y = df['Chance of Admit ']
X = df[features]

y = df['Chance of Admit ']

trainX ,testX , trainY, testY = train_test_split(X, y, train_size = 0.7,random_state = 5) 
lin_reg = LinearRegression()

predY = lin_reg.fit(trainX,trainY).predict(testX)

m1 = r2_score(testY,predY)

print('Accuracy/RSquared : ',r2_score(testY, predY))

print('Root Mean Squared Error : ',np.sqrt(mean_squared_error(testY,predY)))
dec_tree = DecisionTreeRegressor()

predY = dec_tree.fit(trainX,trainY).predict(testX)

m2 = dec_tree.score(testX,testY)

print('Accuracy : ',dec_tree.score(testX, testY))

print('Root Mean Squared Error : ',np.sqrt(mean_squared_error(testY,predY)))



corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import random

from sklearn.linear_model import LinearRegression 

from sklearn.ensemble import RandomForestRegressor 

from sklearn.model_selection import GridSearchCV,train_test_split 

from sklearn.metrics import mean_absolute_error

rf_model = RandomForestRegressor(n_estimators = 100,random_state = 42)

rf_model.fit(trainX,trainY)

print('Mean absolute error for RF model: %0.4f' %mean_absolute_error(testY,rf_model.predict(testX)))
feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['value','Feature'])

plt.figure(figsize=(10, 6)) 

sns.barplot(x="value", y="Feature", data=feature_importance.sort_values(by="value", ascending=False)) 
