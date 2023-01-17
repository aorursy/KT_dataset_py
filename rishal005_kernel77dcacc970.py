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

import matplotlib.pyplot as plt

admission = pd.read_csv("../input/Admission_Predict.csv")

admission.head() # to see the top five records of the data sets
admission.shape # to  see what is the shape of data_set our data set has 400 records and 9 fields
admission.columns #to see the name of the fields 
admission.describe() # to see the mathematical values of the data sets i.e mean,standar_deviation ,minimum_value,maximum_value,counts etc.
admission.info() #to see the type of values in every fields i.e int ,float etc 
admission.isnull().sum() # to see that if dataset has any null_values or not
# the dataset has no null_values so there is no need to fill null_values
# now preparing the input data_set and out_labels
X=admission.drop(['Serial No.','Chance of Admit '],axis=1) #input data_set

X.shape
y=admission['Chance of Admit '] #output labels

y.shape
admission.sample(5)
plt.scatter(admission['GRE Score'],admission['CGPA'])

plt.title('CGPA vs GRE Score')

plt.xlabel('GRE Score')

plt.ylabel('CGPA')

plt.show()
plt.scatter(admission['CGPA'],admission['SOP'])

plt.title('SOP for CGPA')

plt.xlabel('CGPA')

plt.ylabel('SOP')

plt.show()
admission[admission.CGPA >= 8.5].plot(kind='scatter', x='GRE Score', y='TOEFL Score',color="BLUE")



plt.xlabel("GRE Score")

plt.ylabel("TOEFL SCORE")

plt.title("CGPA>=8.5")

plt.grid(True)



plt.show()
admission["GRE Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))



plt.title("GRE Scores")

plt.xlabel("GRE Score")

plt.ylabel("Frequency")



plt.show()


p = np.array([admission["TOEFL Score"].min(),admission["TOEFL Score"].mean(),admission["TOEFL Score"].max()])

r = ["Worst","Average","Best"]

plt.bar(p,r)



plt.title("TOEFL Scores")

plt.xlabel("Level")

plt.ylabel("TOEFL Score")



plt.show()


g = np.array([admission["GRE Score"].min(),admission["GRE Score"].mean(),admission["GRE Score"].max()])

h = ["Worst","Average","Best"]

plt.bar(g,h)



plt.title("GRE Scores")

plt.xlabel("Level")

plt.ylabel("GRE Score")



plt.show()
import seaborn as sns



plt.figure(figsize=(10, 10))



sns.heatmap(admission.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")



plt.show()
admission.Research.value_counts()



sns.countplot(x="University Rating",data=admission)
admission.Research.value_counts()



sns.countplot(x="University Rating",data=admission)
sns.barplot(x="University Rating", y="Chance of Admit ", data=admission)
#splittin the input data(x) and output labels(y) into train data and test data 

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20) # test_size defins the volume of train data and test data here 0.2 means 20% of the data belongs to the test data
X_train.shape
X_test.shape
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

X_train.head()
from sklearn.ensemble import RandomForestRegressor

rgr=RandomForestRegressor()

rgr.fit(X_train,y_train)
rgr.score(X_test,y_test)
import xgboost as xgb

xg = xgb.XGBRegressor()

xg.fit(X_train,y_train)
xg.score(X_test,y_test)
y_predict=rgr.predict(X_test)

y_predict

#Y_test.shape
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

import numpy as np



print('Mean Absolute Error:', mean_absolute_error(y_test, y_predict))  

print('Mean Squared Error:', mean_squared_error(y_test, y_predict))  

print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_predict)))