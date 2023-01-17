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
#first i have imported the required libraries
import pandas as pd        #used for analysis of data
import numpy as np         #for computation
import seaborn as sns      #seaborn library helps for plotting the graphs
import matplotlib.pyplot as plt #this library is also used for plotting
import math                #for calculating basic math functions


#imported the train dataset and stored it to the variable train
train=pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")

#checking the first 5 rows of the data makes us understand how the values are in each of the variable.
train.head()
#checking the dimensions of the dataset
train.shape
#here we check which datatype each variables are in
train.info()

#the summary of the data
train.describe()
#i have done this just to remember which all column are there
train.columns
#this code is to check if there are unique values in the columns
train.nunique()

#since "timeindex" column has the most unique values , we dont need them. so i dropped the column
train=train.drop(["timeindex"],axis=1)

#There is no missing values in any of the columns.
train.isnull().sum()
#it shows that chain of the production line is tensed lesser than the chain of production line is loose
#around 6600 values are loose and around 4200 values are tensed
sns.countplot(x="flag",data=train)
#relation analysis
corelation=train.corr()
corelation
sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)
#dropping dependent variable("flag") means the remaining variables are independent variables and i stored it to variable X
X=train.drop("flag",axis=1)
#here i have stored the dependent variable("flag") to Y
y=train["flag"]

#so basically we just defined the independent variable and dependent variable above
#used for splitting the train dataset into train and test subsets
from sklearn.model_selection import train_test_split    
#here as we can see, the test size is 30% and the train size is 70%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=1)
#here the RandomForestClassifier will graph on the ensemble
from sklearn.ensemble import RandomForestClassifier
#creating an instance of random forest classifier by storing it to the variable rfcmodel
mymodel=RandomForestClassifier(n_estimators=50) 
#here we just fit the model by passing x_train and y_train into it
mymodel.fit(X_train,Y_train)
#did prediction and stored it to variable rfpredict
rfpredict=mymodel.predict(X_test)
#importing accuracy_score function
from sklearn.metrics import accuracy_score
#gives us the accuracy score of the model
accuracy_score(rfpredict,Y_test)
#importing f1_score function
from sklearn.metrics import f1_score
#gives us the f1 score of the model
f1_score(Y_test,rfpredict)
#import the classification report
#the classification report gives us a report which contains details such as precision , recall and f1 score.
from sklearn.metrics import classification_report
#gives us the report
classification_report(Y_test,rfpredict)
#importing confusion matrix function
from sklearn.metrics import confusion_matrix
#shows the confusion matrix  
confusion_matrix(Y_test,rfpredict)
#imported the test dataset and stored it to the variable test

test=pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
test.head()
#checking the first 5 rows of the data.
test.head()
#checkking the dimensions of the dataset
test.shape
#here we check which datatype each variables are in
test.info()

#summary of the dataset
test.describe()
#list of columns in the dataset
test.columns
#checking unique values of the columns
test.nunique
#since "timeindex" column has the most unique values , we dont need them. so i dropped the column
test=test.drop(["timeindex"],axis=1)

#There is no missing values in any of the columns.
test.isnull().sum()
#here we predict the test dataset by using the used model and then i stored it to variable predict
predict=rfcmodel.predict(test)
#storing the sample submission file to sample variable
sample=pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
#predicting the sample
sample["flag"]=predict
#last 5 rows of the sample
sample.tail()
#saving it as submit csv
sample.to_csv("submit_4.csv",index=False)
