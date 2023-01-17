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
# In this project I will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import pandas as pd

ad_data = pd.read_csv("../input/advertising.csv")
ad_data.head()
# to get all the columns information...

# we dont have nul values...

ad_data.info()
# STatistical Information...

ad_data.describe()
#sns.distplot(ad_data['Age'],bins=40)

sns.distplot(ad_data['Age'],kde=False,bins=40)
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
sns.jointplot(ad_data['Age'],ad_data['Area Income'],data=ad_data,kind='scatter')
# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
sns.jointplot(ad_data['Age'],ad_data['Daily Time Spent on Site'],data=ad_data,kind='kde')
# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**
sns.jointplot(ad_data['Daily Time Spent on Site'],ad_data['Daily Internet Usage'],data=ad_data)
ad_data.info()
import matplotlib as mpl

mpl.rcParams['patch.force_edgecolor'] = True
sns.pairplot(data=ad_data,hue='Clicked on Ad',palette='bwr')
#       Logistic Regression

#       train test split, and train our model!
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

ad_data.columns
X = ad_data.drop(['Ad Topic Line', 'City','Country','Timestamp', 'Clicked on Ad'],axis=1)

y = ad_data['Clicked on Ad']
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
logr=LogisticRegression()
logr.fit(X_train,y_train)
predictions = logr.predict(X_test)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
# Predictions and Evaluations

#  ** Create a classification report for the model.**
print(classification_report(y_test,predictions))
# In this project I will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. 

#  We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.