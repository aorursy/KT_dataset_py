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
# Import required libraries 

import numpy as np 

import pandas as pd 



# Import the dataset 

dataset = pd.read_csv('../input/churn-telecom/telecom_industry_churn.csv') 
# Print all the features of the data 

dataset.columns 
# Glance at the first five records 

dataset.head() 
# Churners vs Non-Churners 

dataset['Churn'].value_counts() 
#To group data by Churn and compute the mean to find out if churners make more customer service calls than non-churners:

# Group data by 'Churn' and compute the mean 

dataset.groupby('Churn')['Customer service calls'].mean()
#To find out if one State has more churners compared to another.



# Count the number of churners and non-churners by State 

dataset.groupby('State')['Churn'].value_counts()
#Exploring Data Visualizations : To understand how variables are distributed.

# Import matplotlib and seaborn 

import matplotlib.pyplot as plt 

import seaborn as sns 



# Visualize the distribution of 'Total day minutes' 

plt.hist(dataset['Total day minutes'], bins = 100) 



# Display the plot 

plt.show() 
#To visualize the difference in Customer service calls between churners and non-churners

# Create the box plot 

sns.boxplot(x = 'Churn',y = 'Customer service calls',data = dataset,sym = "",hue = "International plan") 

# Display the plot 

plt.show() 
##In telco churn data, Churn, Voice mail plan, and, International plan, in particular, 

#are binary features that can easily be converted into 0’s and 1’s.

# Features and Labels 

X = dataset.iloc[:, 0:19].values 

y = dataset.iloc[:, 19].values # Churn 



# Encoding categorical data in X 

from sklearn.preprocessing import LabelEncoder 



labelencoder_X_1 = LabelEncoder() 

X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3]) 



labelencoder_X_2 = LabelEncoder() 

X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4]) 



# Encoding categorical data in y 

labelencoder_y = LabelEncoder() 

y = labelencoder_y.fit_transform(y) 
labelencoder_X_0 = LabelEncoder() 

X[:, 0] = labelencoder_X_2.fit_transform(X[:, 0]) 
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

X = scaler.fit_transform(X)
#To Create Training and Test sets

# Splitting the dataset into the Training and Test sets 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2,random_state = 0) 

#To scale features of the training and test sets

# Feature Scaling 

from sklearn.preprocessing import StandardScaler 

sc = StandardScaler() 

X_train = sc.fit_transform(X_train) 

X_test = sc.transform(X_test) 

#To train a Random Forest classifier model on the training set.

# Import RandomForestClassifier 

from sklearn.ensemble import RandomForestClassifier 



# Instantiate the classifier 

clf = RandomForestClassifier() 



# Fit to the training data 

clf.fit(X_train, y_train) 

#Making Predictions

# Predict the labels for the test set 

y_pred = clf.predict(X_test) 

#Evaluating Model Performance

# Compute accuracy 

from sklearn.metrics import accuracy_score 



accuracy_score(y_test, y_pred) 
#Confusion Matrix

from sklearn.metrics import confusion_matrix 

print(confusion_matrix(y_test, y_pred)) 
