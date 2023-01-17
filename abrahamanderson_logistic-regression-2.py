#In this project we will be working with an advertising data set, 

#indicating whether or not a particular internet user clicked on an Advertisement.

#We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#First of all we read the data get an overall insight about the data

df=pd.read_csv("../input/advertising.csv")

df.head()
df.info()
df.describe(include="all") # we get some statistical information about the data
#Now we need to visualize to explore better 

import cufflinks as cf #plotly's library provides intercative and high quality plot as seen below

cf.go_offline()

df["Age"].iplot(kind="hist")
sns.jointplot(x="Age",y="Daily Time Spent on Site",data=df,kind="kde",color="red")

#It seem younger people spend more time on the site
sns.jointplot(x="Daily Time Spent on Site",y="Daily Internet Usage",data=df)
sns.pairplot(df, hue="Clicked on Ad")

#here we get an overal picture about the data.
# Because we will predict whether or not a particular internet user clicked on an Advertisement and the target is binary and classified

#Logistic Regression suits better for this data set

#But we need to split data into training set and testing set using train_test_split

X=df[["Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Male"]]

y=df["Clicked on Ad"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
#Now it is time to train our data and fit it into logistic regression model

from sklearn.linear_model import LogisticRegression

log_regression=LogisticRegression()

log_regression.fit(X_train,y_train)
#Now our model is ready to predict the test data

predictions=log_regression.predict(X_test)

predictions
#Now it is time to evaluate how good the predictions are

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

#The precision and accuracy precentages are over %90, it is very good
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)

#TP=149 : true positive

#FN=8   : false negative

#FP=14  :false positive 

#TN=129 : true negative

#The errors are not too high and absorable 