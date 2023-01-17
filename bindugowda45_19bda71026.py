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
import pandas as pd

import numpy as np
data=pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv') #Loading Train dataset
data.head() #To retrieve first few rows of data
data.info() #To check about dataset (i.e,does it contain any null value, etc)
data.shape #To check number of rows and number of columns in dataset
from sklearn.ensemble import RandomForestClassifier #The package is loaded to split dataset into test and train
X = data.drop('flag', axis=1)

X.head() # X variable is defined which contains only independent variable present in dataset, drop is used to drop dependent variable.

y = data['flag'] # Y variable is defined which contains dependent variable.

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) ## Split dataset into training set and test set
cla = RandomForestClassifier (n_estimators=30,random_state=0) # Create Decision Tree classifer object
clf = cla.fit(X_train,y_train) #Train Decision Tree Classifer
y_pred = cla.predict(X_test) ##Predict the response for test dataset

y_pred
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
test=pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv') #Test dataset is loaded

test.head()
pred=clf.predict(test)

pred #Predicting anomalies (o or 1)
submission=pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv') #Loading submission file
sample=pd.DataFrame({

    'sl.No':submission['Sl.No'],

    'flag':pred

}) #extracting sl.no column from submission file and creating data frame with predicted values
sample.head() #Retrieving few rows of sample data
sample.to_csv("Predicted_score.csv",index=False) # Creating a file name for predicted values
from sklearn.metrics import confusion_matrix #To find confusion matrix which is technique for summarizing the performance of a classification algorithm

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report #A Classification report is used to measure the quality of predictions from a classification algorithm.

print(classification_report(y_test, y_pred))