#--------------------
#Using KNN Model to classify a class into 1 or 0 in Python
#Kernel written by Sikyun (George) Lee
#--------------------

#STARTING FROM HERE

#Before I start, I would like to acknowledge my teachings and resources from Jose Portilla, a data scientist and instructor of 'Python for Data Science and Machine Learning Bootcamp' at Udemy
#This was a project performed within class that I think is useful for new Pythons starters interested in classifying classes based on some numerical information
#The objective is to 'share' information about how to tackle such data only.

#Background Explanation
#our goal is to use the project dataset posted with this kernel to: 
#1) Separate a certain portion of the project data into training sets and testing sets
#2) Use the training sets to "train" our KNN classification model
#3) And apply model to the testing dataset to see how well our model correctly classifies the testing dataset

#Let's import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
#you need this to plot the necessary graphs from matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
#let's import the project data
#the project data is a random set of four-letter characteristics(AKA 'features' in columns) (e.g.: XVPM, GWYH,...etc)
#In a real-world setting, these may be some real features such as 'Credit Amount', 'Height', 'Weight', 'Length', etc...
#within these four-letter characteristics(features), there are certain numerical values(e.g.:1001.55, 1300.03,...etc)
#again, these numerical values will describe each characteristics (e.g.: length : 1001.55, weight : 1300,03,...etc)

#our goal is to: 
#1) Separate a certain portion of the project data into training sets and testing sets
#2) Use the training sets to "train" our KNN classification model
#3) And apply model to the testing dataset to see how well our model correctly classifies the testing dataset

df = pd.read_csv('../input/KNN_Project_Data')

#Now let's try Exploratory Data Analysis (AKA, "EDA")
#Let's try a pairplot to see if there are any relations to any other characteristics (column information) with respect to the "TARGET CLASS" class
#Note that this is a large plot, so it may take some time in some PC environments
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm') #I just like coolwarm palette, but this is changeable to others (e.g.: magma, etc.)

#Because the numerical values do not have a defined range or a "standard" from where it increases or decreases per characteristics(i.e.: columns), we will normalize the numbers using Scikit learn's StandardScaler library
from sklearn.preprocessing import StandardScaler

#create a blank StandardScaler() object 
scaler=StandardScaler()
#now fit the scaler object to the characteristics(column features)
scaler.fit(df.drop('TARGET CLASS',axis=1))
#we need to drop the 'TARGET CLASS' column since this is the column that we want to classify/predict from the original dataset
#axis =1 will mean we are dropping the column data instead of the row data

#use .transform() method to transform the characteristic(feature) to a scaled version
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


#now let's put the scaled_features object into a pandas dataframe to work on
#this is the "new" version of the data we will use from hereon
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
#the columns[:-1] means this excludes the 'TARGET CLASS' column
#now we will split the scaled dataset into train and test sets
#I will use 70-30% ratio split the dataset
#first, import the required dataset
from sklearn.model_selection import train_test_split
#Make Training X datasets and Y datasets along with Testing X and Y datasets with the split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)

#now we will import the KNN classifier library from Scikit learn library
from sklearn.neighbors import KNeighborsClassifier
#create a blank KNN model with n_neighbors, which is the initial number of neighbors, set at 1 
knn = KNeighborsClassifier(n_neighbors=1)

#let's fit this model to the training X and Y datasets
knn.fit(X_train,y_train)

#now, let's test if our model that is trained by our X and y training datasets can accurately classify our X testing datasets in respect to the acutal y test dataset classes

#we will define a pred object as our prediction from our KNN model applied to the X testing dataset
pred = knn.predict(X_test)
#to actually check the accuracy and prediction power, we use the confusion matrix and classification reports
#let's import the library from sklearn.metrics
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#to improve our model, we can adjust the K constant in a variety of ways
#In this Kernel, I will use the 'elbow method' to pick an appropriate K constant

#First, let's create a blank list called the 'error_rate'
error_rate = []
#Then let's make a loop that ranges from 1 to a large number (e.g.: 50) to test the k-neighbor constant
#This loop will test K=1,2,3,...50 to see if the predicted y test value is the same as the actual y test value
#If they are the same, they will be disregarded, but if different, then they will be appended to the error_rate list

#This will take some time based on PC environments
for i in range(1,50):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

#Based on where the Kth value seems to stablize, pick a Kth value (e.g.: 30)
#Now, we will re-train our KNN model with the K=30 parameter instead of K=1
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

#Now let's compare the classification/prediction power when K=30 with the confusion matrix and classificaiton report
#normally this increases the model's prediction power and accuracy
print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
