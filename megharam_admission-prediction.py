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
data= pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")

df=pd.DataFrame(data)

df.head() # Shows 1st 5 rows of the entire dataset
df.describe() # Overall description of the data
df.corr()
#Scatter Plot between CGPA and GRE Score



import matplotlib.pyplot as plt

plt.scatter(df['Chance of Admit '],df['CGPA'])

plt.xlabel("Chance of Admit")

plt.ylabel("CGPA")

plt.title("CGPA vs Chance of Admit")
plt.scatter(df['CGPA'],df['SOP'])

plt.xlabel("CGPA")

plt.ylabel("SOP")

plt.title("CGPA vs SOP")
#we are only considering those students who's CGPA is more than 8.5

df[df['CGPA']>=8.5].plot(kind="scatter",x='GRE Score',y='TOEFL Score',color="red")



plt.xlabel("GRE Score")

plt.ylabel("TOEFL Score")

plt.title("GRE Score vs TOEFL Score for CGPA>=8.5")

plt.grid(True)

plt.show()




df['GRE Score'].plot(kind='hist',bins=100,figsize=(6,6))



plt.xlabel("GRE Score")

plt.ylabel("Frequency")

plt.title("GRE Score")



plt.show()
#Bar Chart

import numpy as np

y=np.array([df["TOEFL Score"].min(),df["TOEFL Score"].mean(),df["TOEFL Score"].max()])

x=["Worst","Average","Best"]

plt.bar(x,y)



plt.xlabel("Level")

plt.ylabel("TOEFL Score")

plt.title("TOEFL Score Comparison")



plt.show()
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot=True,linewidth=0.05,fmt='.2f',cmap="magma")

plt.show()
sns.countplot(x="University Rating",data=df)
df.columns

df.rename(columns={'Chance of Admit ':'Chance of Admit'},inplace=True)

print(df.head())

temp=df

#As per the correlation matrix most important features are 'GRE Score', 'TOEFL Score' and 'CGPA'. So lets choose that only

df2=df[['GRE Score', 'TOEFL Score','CGPA','Chance of Admit']] 

print(df2.head)
X=df2.drop(['Chance of Admit'],axis=1)

print(X.shape)



y=df2['Chance of Admit']

print(y.shape)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#Normalisation: Scaling 

from sklearn.preprocessing import MinMaxScaler

s=MinMaxScaler(feature_range=(0,1))

X_train[X_train.columns]=s.fit_transform(X_train[X_train.columns])

X_train.head()

#Only apply transform function on testing data

X_test[X_test.columns]=s.transform(X_test[X_test.columns])
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import numpy as np





linearRegression_model=LinearRegression()

linearRegression_model.fit(X_train,y_train)

y_predict= linearRegression_model.predict(X_test)

df1=pd.DataFrame({'Actual':y_test,'Predicted':y_predict})

print(df1)

score=linearRegression_model.score(X_test,y_test)

print("Score: ",score)

meanAbsoluteError=mean_absolute_error(y_test,y_predict)

print("Mean Absolute Error: ",meanAbsoluteError)

meanSquaredError=mean_squared_error(y_test,y_predict)

print("Mean Squared Error: ",meanSquaredError)

print("Root Mean Squared Error: ",np.sqrt(meanSquaredError))

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import numpy as np



randomForestRegressor_model=RandomForestRegressor(n_estimators=100,random_state=42)

randomForestRegressor_model.fit(X_train,y_train)



y_predict= randomForestRegressor_model.predict(X_test)

df1=pd.DataFrame({'Actual':y_test,'Predicted':y_predict})

print(df1)

score=randomForestRegressor_model.score(X_test,y_test)

print("Score: ",score)

error=mean_absolute_error(y_test,y_predict)

print("Error: ",error)

meanSquaredError=mean_squared_error(y_test,y_predict)

print("Mean Squared Error: ",meanSquaredError)

print("Root Mean Squared Error: ",np.sqrt(meanSquaredError))
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import numpy as np



decisionTreeRegressor_model=DecisionTreeRegressor(random_state=42)

decisionTreeRegressor_model.fit(X_train,y_train)



y_predict= decisionTreeRegressor_model.predict(X_test)

df1=pd.DataFrame({'Actual':y_test,'Predicted':y_predict})

print(df1)

score=decisionTreeRegressor_model.score(X_test,y_test)

print("Score: ",score)

error=mean_absolute_error(y_test,y_predict)

print("Error: ",error)



meanSquaredError=mean_squared_error(y_test,y_predict)

print("Mean Squared Error: ",meanSquaredError)

print("Root Mean Squared Error: ",np.sqrt(meanSquaredError))