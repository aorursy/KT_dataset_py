# Importing appropriate libraries and packages for analysis

import pandas as pd                                                     

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns
# Reading the Data

data = pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")     
 # Getting to look at the first 5 rows of the data

data.head()                                                           
 # Understand the type of data in each column 

data.info()                                                           
 # Used for calculating basic statistical analysis

data.describe()                                                      
 # Gives the names of all the columns

data.columns                                                       
# To check whether the data has any missing values

data.isnull().sum()                                                 
 # Dropping the column 'timeindex' as it has no importance 

data.drop(['timeindex'],axis = 1,inplace = True)                  
data.head()
# Fitting a correlation matrix and visualizing with the help of an heat map 

plt.figure(figsize = (15,10))

sns.heatmap(data.corr(),annot = True,linewidth = 0.1,cmap="YlGnBu")

plt.show()
# Importing libraries that are esstential for building a model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn import preprocessing

from sklearn.metrics import f1_score
# Selecting the appropriate attributes to split and train the data



X = data[['currentBack', 'motorTempBack', 'positionBack',

       'refPositionBack', 'refVelocityBack', 'trackingDeviationBack',

       'velocityBack', 'currentFront', 'motorTempFront', 'positionFront',

       'refPositionFront', 'refVelocityFront', 'trackingDeviationFront',

       'velocityFront']]

y = data[['flag']]
# Using library to split the data and where we fit the model for Logistic and check the score



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

clf = LogisticRegression()

clf.fit(X_train,y_train)

clf.score(X_test,y_test)
# Checking the F1 score of the Logistic 

clf_predict = clf.predict(X_test)

f1_score(clf_predict,y_test)
# Using library to fit the model for KNN(K nearest neighbor) and checking the score

KNN = KNeighborsClassifier()

KNN.fit(X_train,y_train)

KNN.score(X_test,y_test)
# Checking the F1 score of the KNN 

KNN_predict = KNN.predict(X_test)

f1_score(KNN_predict,y_test)
# Using library to fit the model for Decision Tree and checking the score

model = DecisionTreeClassifier(criterion ='entropy', random_state = 1)

model.fit(X_train,y_train)

model.score(X_test,y_test)
# Checking the F1 score of the Decision Tree

Predictions = model.predict(X_test)

f1_score(Predictions,y_test)
# Using library to fit the model for Random Forest and checking the score

model_1 = RandomForestClassifier(criterion='entropy', random_state = 0)

model_1.fit(X_train,y_train)

model_1.score(X_test,y_test)
# Checking the F1 score of the Random Forest

Predictions = model_1.predict(X_test)

f1_score(Predictions,y_test)
# Finding the feature importance for a Random forest

model_1.feature_importances_
# Creating a dataframe for attibutes corresponding to their feature importance 

FI = pd.DataFrame({

    "Var_names":X.columns,

    "FI":model_1.feature_importances_

})
# Making sure we get the dataframe in an other and using cumsum to find the cummulative sum of the features after coverting it into percenatge

FI = FI.sort_values(by="FI",ascending=False)

FI['percentage'] = FI['FI']*100

FI['çum_sum'] = FI['percentage'].cumsum()

FI.reset_index()
# Selection the attributes with higher feature importance

X1 = data[['currentBack', 'motorTempBack', 'positionBack','trackingDeviationBack','velocityBack', 'currentFront',

          'motorTempFront','positionFront','trackingDeviationFront','velocityFront']]

Y1 = data[['flag']]
# Standardization of the attributes 

X1 = preprocessing.scale(X1)
# Using library to split the data and  we fit the model for Decision Tree and check the score



X_train,X_test,Y_train,Y_test = train_test_split(X1,Y1,test_size = 0.2)

model_2 = DecisionTreeClassifier(criterion ='entropy', random_state = 1)

model_2.fit(X_train,Y_train)

model_2.score(X_test,Y_test)
# Checking the F1 score of the Decision Tree



Predictions = model_2.predict(X_test)

f1_score(Predictions,Y_test)
# Using library to fit the model for Random Forest and checking the score



model_3 = RandomForestClassifier(criterion='entropy', random_state = 0)

model_3.fit(X_train,Y_train)

model_3.score(X_test,Y_test)
# Checking the F1 score of the Random Forest



Predictions = model_3.predict(X_test)

f1_score(Predictions,Y_test)