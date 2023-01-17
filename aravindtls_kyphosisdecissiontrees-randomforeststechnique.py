

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline 

#We use "matplotlib inline" to print the graphs and charts in the command line itself
df = pd.read_csv('../input/kyphosis.csv')
#Let us check the head of the data set 
df.head()
#It contains four columns as seen above , we need to chose the right coloumns for further progress
#Let us check the entries and datatypes 
df.info()
#Could see that it has 81 entries , its quite a small data set to work 
sns.pairplot(df,hue='Kyphosis') #I am passing data as the dataframe and the hues as Kyphosis column
from sklearn.cross_validation import train_test_split
#We need to create the X and y for fitting the model .'X' will be the set of all cols except the need to be predicted variable and 'y' will only be the predicted variable
X = df.drop('Kyphosis',axis = 1)
y = df['Kyphosis']
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state = 59)
#We need to train the single decision tree now from the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtreec = DecisionTreeClassifier() #Created an object here 
#We need to fit the model with the training data 
dtreec.fit(X_train,y_train)
#Now lets see how our decisopon tree was able to predicts based on the data and the model we have trained 
predict = dtreec.predict(X_test)
#Lets test our models accuracy and error values using the classification report and confusion matrix
from sklearn.metrics import classification_report , confusion_matrix
print("The confusion Matrix is : \n",confusion_matrix(y_test,predict))
print('\n')
print("The classification report  : \n",classification_report(y_test,predict))
from sklearn.ensemble import RandomForestClassifier
#Now using Random forest classifier technique and check the diff in accuracy between the two models
rfc = RandomForestClassifier(n_estimators=200) 
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print("The confusion Matrix is : \n",confusion_matrix(y_test,rfc_pred))
print('\n')
print("The classification report  : \n",classification_report(y_test,rfc_pred))
#We could see a better recall score and the confusion matrix shows more positive outputs , but both gave us similar scores.
#This comes to our end of model building with the decison trees and random forests