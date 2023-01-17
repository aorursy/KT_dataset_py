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
import pandas as pd                                                      #importing appropriate libraries and packages for analysis.

import numpy as np

import sklearn.model_selection as ms

import seaborn as sb

import matplotlib.pyplot as plt

import sklearn.preprocessing as pre

import sklearn.linear_model as lm

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score
data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")                   #reading the csv file for analysis

data.head()  
data.info()         #getting information about the data
data.isnull().sum()      #cheching for missing values 
data.describe()         #getting the summary of data
plt.figure(figsize = (20,10))

matrix = np.triu(data.corr())

sb.heatmap(data.corr(), annot=True, mask=matrix,fmt=".1g",vmin=-1, vmax=1, center= 0,cmap= 'coolwarm')   #getting corrleation matrix to understand the relationship between the variables
X=data[['currentBack', 'motorTempBack', 'positionBack',

       'refPositionBack', 'refVelocityBack', 'trackingDeviationBack',

       'velocityBack', 'currentFront', 'motorTempFront', 'positionFront',                      #selecting the appropriate variable to split and train the data

       'refPositionFront', 'refVelocityFront', 'trackingDeviationFront',

       'velocityFront']]

X.head()
Y=data[['flag']]                #taking flag as a dependent variable according to the problem statement 

Y.head()
x_train,x_test,y_train,y_test=ms.train_test_split(X,Y,test_size=0.2)         #splitting the data to train and test for fitting models and to train the dataset for prediction.

x_train.shape,x_test.shape,y_train.shape,y_test.shape         
lr=lm.LogisticRegression()                 #using logistic regression model for anamoly detection 

lr.fit(x_train,y_train)                 #fitting the train data using logistic regression model
lr.fit(x_test,y_test)                         # fitting the test data using logistic regression model
lr.score(x_train,y_train)                   #getting the score for fitted train data
lr.score(x_test,y_test)                 #getting the score for fitted test data
y_predict=lr.predict(x_test)           #predicting the flag values after training the data using logistic regression 

y_predict
cm=confusion_matrix(y_predict,y_test)               #getting confusion matrix.

acs=accuracy_score(y_predict,y_test)                #checking for accuracy

ps=precision_score(y_predict,y_test)                 #getting the precision score

rs=recall_score(y_predict,y_test)                    #recall score

print("Confusion Matrix :\n", cm)                    #prints the confusion matrix 

print("Accuracy :" , accuracy_score(y_test,y_predict))          #prints the accuracy value

print("Recall Score :",recall_score(y_test,y_predict))          #prints the recall score

print("Precision Score :",precision_score(y_test,y_predict))     #prints the precision score
f1_score(y_predict,y_test)          #getting the f1 score(f1 scores tells how well the model is behaving and how accurate the predictions)

print("The F1 score is:",f1_score(y_predict,y_test))
test_data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")       #reading test data to include the predicted flag values

test_data=test_data.drop(columns=['timeindex'])        #dropping the timeindex column since it's not much important for the analysis

test_data.head()           #displays the first 5 rows of the data
test_data.columns         #displays the column names in the dataset
test_data['flag']=lr.predict(test_data)           #adding a column to store the predicted flag values
test_data.head()           #displays the first 5 rows of the data
Sample_Submission = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")          #submitting the work to get the score

Sample_Submission['flag'] = test_data['flag']

Sample_Submission.to_csv("log model.csv",index=False)
from sklearn.ensemble import RandomForestClassifier          #library to run random forest model
x_train,x_test,y_train,y_test=ms.train_test_split(X,Y,test_size=0.2)     #splitting the data to train and test for fitting models and to train the dataset for prediction.

x_train.shape,x_test.shape,y_train.shape,y_test.shape              #gives the dimensions of the splitted data
clasif = RandomForestClassifier(n_estimators=40, random_state=0)              #performing random forest model 

clasif.fit(x_train, y_train)                                        #fitting random forest model for predictions

y_predict = clasif.predict(x_test)                                  #storing the predicted values to the variable y_predict
cm=confusion_matrix(y_predict,y_test)                              #getting confusion matrix.

acs=accuracy_score(y_predict,y_test)                               #checking for accuracy

ps=precision_score(y_predict,y_test)                               #checking for precison score

rs=recall_score(y_predict,y_test)                                  #checking for recall score

print("Confusion Matrix :\n", cm)                                  #prints confusion matrix        

print("Accuracy :" , accuracy_score(y_test,y_predict))             #prints the accuracy score

print("Recall Score :",recall_score(y_test,y_predict))             #prints the recall score

print("Precision Score :",precision_score(y_test,y_predict))       #prints the precison score
f1_score(y_predict,y_test) 

print("The F1 score is:",f1_score(y_predict,y_test))   #getting f1 score to know the performance of the model to check how good the model is behaving.
test_data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")        #reading test data for predictions. 

test_data=test_data.drop(columns=['timeindex'])             #dropping the timeindex column since it's not much important for the analysis

test_data.head()                #displays first 5 rows of the test dataset
test_data['flag']=clasif.predict(test_data)             #adding a column to store the predicted flag values
test_data.head()               #displays first 5 rows of the test dataset
Sample_Submission = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")          #submitting the work to get the score

Sample_Submission['flag'] = test_data['flag']

Sample_Submission.to_csv("random forest model.csv",index=False)
from sklearn.tree import DecisionTreeClassifier            #importing the library for decision tree model
x_train,x_test,y_train,y_test=ms.train_test_split(X,Y,test_size=0.2)        #splitting the data to train and test for fitting models and to train the dataset for prediction.

x_train.shape,x_test.shape,y_train.shape,y_test.shape           #displays the dimensions of the splitted data
dt = DecisionTreeClassifier()              #running the decision tree model

dt.fit(x_train,y_train)                 #fits the decision tree model for train data
y_predict = dt.predict(x_test)                 #storing the predicted values to the variable y_predict

y_predict
cm=confusion_matrix(y_predict,y_test)                              #getting confusion matrix.

acs=accuracy_score(y_predict,y_test)                               #checking for accuracy

ps=precision_score(y_predict,y_test)                               #checking for precison score

rs=recall_score(y_predict,y_test)                                  #checking for recall score

print("Confusion Matrix :\n", cm)                                  #prints confusion matrix        

print("Accuracy :" , accuracy_score(y_test,y_predict))             #prints the accuracy score

print("Recall Score :",recall_score(y_test,y_predict))             #prints the recall score

print("Precision Score :",precision_score(y_test,y_predict))       #prints the precison score
f1_score(y_predict,y_test)                #getting f1 score to know the performance of the model to check how good the model is behaving.

print("The F1 score is:",f1_score(y_predict,y_test))
test_data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")             #reading test data for predictions. 

test_data=test_data.drop(columns=['timeindex'])             #dropping the timeindex column since it's not much important for the analysis

test_data.head()                #displays first 5 rows of the test dataset

test_data['flag']=dt.predict(test_data)              #adding a column to store the predicted flag values
test_data.head()               #displays the first 5 rows of the test data
Sample_Submission = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")          #submitting the work to get the score

Sample_Submission['flag'] = test_data['flag']

Sample_Submission.to_csv("decision tree model.csv",index=False)