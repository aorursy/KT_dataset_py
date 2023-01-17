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
# Importing pandas library which is a data analysis library that will felicitate analysis and manipulation with much ease



import pandas as pd 
# Reading the train csv file



train = pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")
# Creating a variable named features to store the features that are prevelant. iloc returns the columns at particular indexes that are mentioned. Below mentioned include all columns/features except feature at column number 0(time index)



features = train.iloc[:,2:]
# Similarly iloc function given below chooses the target variable "flag" indexed at column number 1 and saves it in a variable named target



target = train.iloc[:,1]
# Importing train_test split from sklearn.model_selection to felicitate the splitting of the training data into test and train. The model fitted will be trained on the training subset and evaluated on the testing subset



from sklearn.model_selection import train_test_split
# The train data is split into 80% for training and 20% for validation, the variable names used is self explanatory



X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
# I have made use of the random forest classifier and imported the algorithm from sklearn.ensemble



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score   #Measuring accuracy of the model

from sklearn.metrics import f1_score         #Evaluating the model using f1 score as the evalution metric to know the goodness of the model
# Building the Random Forest Classifier Model



rfc_model = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', max_depth = 50)
# Finally fitting the model based on the independent variables(features) and in the dependent variable (target) from the training data 



rfc_model.fit(X_train, y_train)
# Using the above fit I am trying to predict the dependent variable for the validation data



y_pred = rfc_model.predict(X_val)
# Displaying the accuracy of the model



accuracy_score(y_val, y_pred)
#Displaying the f1 score of the model



f1_score(y_val, y_pred)
# Importing the test csv file in order to predict the dependent variable for all data point within this file



test = pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
# Using only the dependent varible(target) column from test data



test = test.iloc[:,1:]
# Finally the prediction



test_pred = rfc_model.predict(test)
# Displaying the shape of test_pred



test_pred.shape
# The predicted class labels for the test data



test_pred
# Creating the CSV for submission



csv = pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
csv["flag"]=test_pred
csv.to_csv("Sample Submission.csv", index = False)