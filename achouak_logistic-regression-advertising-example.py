# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split #split data to test and train data 

from sklearn.linear_model import LogisticRegression # Logistic Regression model

from sklearn.metrics import classification_report #evaluate the model

import matplotlib.pyplot as plt #visualisation 
#get the data path

ad_path="../input/advertising/advertising.csv"



#get the data

ad_data=pd.read_csv(ad_path)



#show the 5 rows of ad_data

ad_data.head()
ad_data.describe()
#The outcome 

y=ad_data["Clicked on Ad"]

# The features 

X= ad_data[["Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Male"]]
#33% of data is used for test the rest of data is used for training our model 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#create an instance of Logistic regression model 

lgmodel=LogisticRegression()



#fit the model on the training set

lgmodel.fit(X_train,y_train)
predictions =lgmodel.predict(X_test)
#Print a text report showing the main classification metrics

print(classification_report(y_test,predictions))
final_predictions = lgmodel.predict(X)
predic = pd.DataFrame({'Predict_Click on Ad': final_predictions})

output=pd.concat([X,predic], axis=1)

output.head()
output.to_csv('Ad_predictions.csv', index=False)

print("Your output was successfully saved!")