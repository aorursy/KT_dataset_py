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
#importing the librararies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#loading the train,test and sample datasets

train = pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")

test = pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")

sample = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
#displaying the first 5 rows of the training dataset

train.head()
#Extracting the information of each feature of the training dataset

train.info()
#To check the count of target variable

train['flag'].value_counts()
#To check the correlation of the freatures of training dataset

train.corr()
#To check the summary of the training dataset

train.describe()
#To check the correlation of currentBack and CurrentFront features with respect to the target variable of the training dataset

sns.scatterplot(x="currentBack", y="currentFront", hue="flag",data=train)
#To check the correlation of currentBack and CurrentFront features of the testing dataset

sns.scatterplot(x="currentBack", y="currentFront", data=test)
#To check the correlation of motorTempBack and motorTempFront features with respect to the target variable of the training dataset

sns.scatterplot(x="motorTempBack", y="motorTempFront", hue="flag",data=train)
#To check the correlation of currentBack and CurrentFront features of the testing dataset

sns.scatterplot(x="motorTempBack", y="motorTempFront", data=test)
#To check the distribution of the motortempback and motortempfront features of the training dataset

fig = plt.figure(figsize=(10,6))

#plotting the distribution of motorTempBack variable of the training data

s1 = sns.distplot(train.motorTempBack,color = "blue")

#plotting the distribution of motorTempFront variable of the training data

s2 = sns.distplot(train.motorTempFront, color = "red")

fig.legend(labels=['motorTempBack','motorTempFront'])

#To show the 2 distribution plots in one plot

plt.show(s1,s2)
#To check the distribution of the motorTempBack and motorTempFront features of the testing data

fig = plt.figure(figsize=(10,6))

#plotting the distribution of motorTempBack variable of the test data

s1 = sns.distplot(test.motorTempBack,color = "blue")

#plotting the distribution of motorTempFront variable of the test data

s2 = sns.distplot(test.motorTempFront, color = "red")

fig.legend(labels=['motorTempBack','motorTempFront'])

#To show the distribution plots in one graph

plt.show(s1,s2)
#To check the correlation between positionback and positionfront variables with respect to the target variable of the training data

sns.scatterplot(x="positionBack", y="positionFront", hue="flag",data=train)
fig = plt.figure(figsize=(10,6))

s1 = sns.distplot(train.positionBack)

s2 = sns.distplot(train.positionFront)

fig.legend(labels=['positionBack','positionFront'])

plt.show(s1,s2)
#To check the correlation between refpositionback and refpositionfront variables with respect to the target variable of the training data

sns.scatterplot(x="refPositionBack", y="refPositionFront", hue="flag",data=train)
#To check the correlation between refvelocityback and refvelocityfront variables with respect to the target variable of the tarining data

sns.scatterplot(x="refVelocityBack", y="refVelocityFront", hue="flag",data=train)
#To check the correlation between velocityback and velocityfront variables with respect to the target variable of the tarining data

sns.scatterplot(x="velocityFront", y="velocityBack", hue="flag",data=train)
#To check the correlation between trackingdeviationback and trackingdeviationfront features wrt the target variable of the tarining data

sns.scatterplot(x="trackingDeviationFront", y="trackingDeviationBack", hue="flag",data=train)
#Distribution plot of trackingdeviationback and trackingdeviationfront variables for training data

fig = plt.figure(figsize=(10,6))

#distribution of the variable trackingdeviationback of the training data

sns.distplot(train.trackingDeviationBack)

#distribution of the variable trackingdeviationfront of the training data

sns.distplot(train.trackingDeviationFront)

fig.legend(labels=['trackingDeviationBack','trackingDeviationFront'])

#To show the distribution plots

plt.show()
#Distribution plot of trackingdeviationback and trackingdeviationfront variables for training data

fig = plt.figure(figsize=(10,6))

#distribution of the variable trackingdeviationback of the test data

sns.distplot(test.trackingDeviationBack)

#distribution of the variable trackingdeviationfront of the test data

sns.distplot(test.trackingDeviationFront)

fig.legend(labels=['trackingDeviationBack','trackingDeviationFront'])

#To show the distribution plots

plt.show()
#Splitting the training data into X(independent variables) and Y(target variable)

#Dropping other 4 features along with target variable since the other 4 features has very low correlation value with respect to the target variable

X = train.drop(columns=['flag','refPositionBack','refPositionFront','positionBack','positionFront'], axis=1)

y = train['flag']

#Dropping the columns of testing data with respect to the training dataset

test = test.drop(columns=['refPositionBack','refPositionFront','positionBack','positionFront'],axis =1)
#Applying the feature scaling technique(standarization) to rescale the value between -3 to +3

#importing the Standarization module from sklearn

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

#fitting and transforming the standarization technique on training dataset

X = sc.fit_transform(X)

#transforming the testing dataset with respect to training data

test= sc.transform(test)
#Applying the dimensionality reduction technique

#importing the pca module from scikit learn library

from sklearn.decomposition import PCA

pca = PCA(n_components=None)

#fitting the pca to training data

X = pca.fit_transform(X)

#transforming the pca technique on testing data with respect to training data

test = pca.transform(test)

#To check the fraction of the variance explained by the pca

explained_variance = pca.explained_variance_ratio_
#splitting the training data into train(70%) and test(30%) data 

#import the train test split module from scikit learn library

from sklearn.model_selection import train_test_split

#splitting the training data into train and test 

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
#importing the Random forest algorithm from scikit learn library

from sklearn.ensemble import RandomForestClassifier

#fitting the model on splitted training data

classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=0)

classifier.fit(x_train,y_train)
#predicting the test values of the splitted data

y_pred = classifier.predict(x_test)

#TO get the accuracy scores of x_train and x_test

clft=classifier.score(x_train,y_train)

clftest=classifier.score(x_test,y_test)

print(clft,clftest)
#to get the f1 score on the spliited data

from sklearn.metrics import f1_score

f1_score(y_pred,y_test)
#Applying the Support Vector Classifier model

#Importing the Support vector classsifier algortihm from scikit learning library

from sklearn.svm import SVC

#fitting the model on splitted training data

cl = SVC(kernel='rbf',random_state=0)

cl.fit(x_train,y_train)

#predicting the test values of the splitted data

y_prediction = cl.predict(x_test)

#TO get the accuracy scores of x_train and x_test

clftr=cl.score(x_train,y_train)

clft=cl.score(x_test,y_test)

print(clftr,clft)
#to get the f1 score of the splitted data

from sklearn.metrics import f1_score

f1_score(y_prediction,y_test)
#importing the Random forest algorithm from scikit learn library

from sklearn.ensemble import RandomForestClassifier

#fitting the model on entire training data

classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=0)

classifier.fit(X,y)
#predicting the target values of test data by training the model on train dataset

y_predictions_RF = classifier.predict(test)

#evaluating the accuracy of the training dataset based on random forest algorithm

classifier.score(X,y)
#Replacing the predicted values of random forest model in sample csv file

sample = sample.assign(flag = y_predictions_RF)

#Exporting the sample file 

sample.to_csv('submit1.csv',index=False)

#displaying the first 5 records of sample csv file

sample.head()
#To check the sum of count of classes of predicted values 

sample['flag'].value_counts()
#Applying the Support Vector Classifier model

#Importing the Support vector classsifier algortihm from scikit learning library

from sklearn.svm import SVC

#fitting the Support vector classsifier model on entire training dataset

cl = SVC(kernel='rbf',random_state=0)

cl.fit(X,y)

#Predicting the values on test data

y_prediction_svc = cl.predict(test)

#getting a accuracy score of training data

cl.score(X,y)
#Replacing the predicted values in the sample csv file

sample = sample.assign(flag = y_prediction_svc)

#Extracting the sample csv file

sample.to_csv('submit2.csv',index=False)

#displaying the first 5 records of the sample csv file

sample.head()
##To check the sum of count of classes of predicted values 

sample['flag'].value_counts()