import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import seaborn as sns

sns.set_palette('husl')

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
import warnings



def fxn():

    warnings.warn("deprecated", DeprecationWarning)



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    fxn()
Data_Universe=pd.read_csv('../input/Iris.csv')

# Checking the sample of data

Data_Universe.head()
Data_Universe.shape
Data_Universe.columns
Data_Universe.drop(['Id'],axis=1,inplace=True)
# Check are their any NA vlaues present in Dataset

Data_Universe.isna().sum()
#Check the Kurtosis Values for all columns

Data_Universe.kurtosis()
# At the same check the skewness for all columns

Data_Universe.skew()
#Check how columns are co-realated to each other

Data_Universe.corr()

#Lets plot above Results to see actual Correalation between the Feilds
tmp = Data_Universe

g = sns.pairplot(tmp, hue='Species', markers='+')

plt.show()
#Sampling of Data

x=pd.DataFrame(Data_Universe.iloc[:,:-1])

y=pd.DataFrame(Data_Universe.iloc[:,-1])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
y_train.head()
# To Check why we selected the values of k=13 . Check below code which decides the values of K 

Model_1=KNeighborsClassifier(13)

Model_1.fit(x_train,y_train)

y_pred=Model_1.predict(x_test)
from sklearn.metrics import  accuracy_score

matrix=accuracy_score(y_test,y_pred)

print("Accuracy of Model Without Cross validation approach is " ,matrix,"percent")
k_value=range(1,31,1)

accuracy=[]

y_pred=[]

for i in k_value:

    Model=KNeighborsClassifier(i)

    Model.fit(x_train,y_train)

    y_pred=pd.Series(Model.predict(x_test))

    Score=accuracy_score(y_test,y_pred)

    accuracy.append(Score)    
plt.plot(k_value, accuracy)

plt.xlabel('Value of k for KNN')

plt.ylabel('Accuracy Score')

plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

plt.show()
# Logistic Regression Model Without K fold Cross Validation Approach

from sklearn.linear_model import LogisticRegression

Logit_Model_without_kfold=LogisticRegression()

Logit_Model_without_kfold.fit(x_train,y_train)

logit_pred=Logit_Model_without_kfold.predict(x_test)

Accuracy=accuracy_score(y_test,logit_pred)

print('Logistic Regression Model accuracy without K fold Corss Validation is',Accuracy)
# Logistic Regression Model With K fold Cross Validation Approach

from sklearn.linear_model import LogisticRegressionCV

#LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial').fit(X, y)

#multi_class : str, {‘ovr’, ‘multinomial’, ‘auto’}, default: ‘ovr’

Logit_Model_with_kfold=LogisticRegressionCV(cv=5,random_state=0,multi_class='multinomial')

Logit_Model_with_kfold.fit(x_train,y_train)

logit_pred=Logit_Model_with_kfold.predict(x_test)

Accuracy=accuracy_score(y_test,logit_pred)

print('Logistic Regression Model accuracy without K fold Corss Validation is',Accuracy)