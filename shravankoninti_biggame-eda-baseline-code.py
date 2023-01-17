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
import pandas as pd                                                  # to import csv and for data manipulation

import matplotlib.pyplot as plt                                      # to plot graph

import seaborn as sns                                                # for intractve graphs

import numpy as np                                                   # for linear algebra

import datetime                                                      # to deal with date and time

%matplotlib inline

from sklearn.preprocessing import StandardScaler                     # for preprocessing the data

from sklearn.ensemble import RandomForestClassifier                  # Random forest classifier

from sklearn.tree import DecisionTreeClassifier                      # for Decision Tree classifier

from sklearn.svm import SVC                                          # for SVM classification

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics, preprocessing, model_selection
train_df = pd.read_csv('/kaggle/input/who-wins-the-big-game/train.csv')

train_df.head()
test_df = pd.read_csv('/kaggle/input/who-wins-the-big-game/test.csv')

test_df.head()
sub_df = pd.read_csv('/kaggle/input/who-wins-the-big-game/sample_submission.csv')

sub_df.head()
print("The total number of Rows in Train dataset is : ", train_df.shape[0])

print("The total number of Rows in Test dataset is : ", test_df.shape[0])

print("The total number of Rows in both Train and Test dataset is : ", train_df.shape[0]+test_df.shape[0])
train_df.keys()
train_df.columns
test_df.columns
train_df.dtypes
test_df.dtypes
train_df['Won_Championship'].value_counts()
# Normalise can be set to true to print the proportions instead of Numbers.

train_df['Won_Championship'].value_counts(normalize=True)
train_df['Won_Championship'].value_counts().plot.bar(figsize=(4,4),title='Won_Championship - Split for Train Dataset')

plt.xlabel('Won_Championship')

plt.ylabel('Count')
for col in train_df.columns:

    if train_df[col].dtype==object:

        print(col)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))

        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))

        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
X = train_df.drop(['ID','Won_Championship'],axis=1)

y = train_df['Won_Championship']



test_X = test_df.drop(['ID'],axis=1)

# TODO: Shuffle and split the data into training and testing subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=100)



# Success

print ("Training and testing split was successful.")
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



model_log = LogisticRegression()

model_log.fit(X_train, y_train)

pred_cv = model_log.predict(X_valid)

accuracy_score(y_valid,pred_cv)
confusion_matrix = confusion_matrix( y_valid,pred_cv)

print("the recall for this model is :",confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0]))



fig= plt.figure(figsize=(6,3))# to plot the graph

print("TP",confusion_matrix[1,1,]) 

print("TN",confusion_matrix[0,0]) 

print("FP",confusion_matrix[0,1]) 

print("FN",confusion_matrix[1,0])

sns.heatmap(confusion_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)

plt.title("Confusion_matrix")

plt.xlabel("Predicted_class")

plt.ylabel("Real class")

plt.show()

print(confusion_matrix)

print("\n--------------------Classification Report------------------------------------")

print(classification_report(y_valid, pred_cv)) 
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

model_rf.fit(X_train, y_train)

pred_cv = model_rf.predict(X_valid)

accuracy_score(y_valid,pred_cv)
pred_test = model_rf.predict(test_X)

pred_test = pd.DataFrame(pred_test)

pred_test.columns = ['Won_Championship']
importances=pd.Series(model_rf.feature_importances_, index=X.columns).sort_values()

importances.plot(kind='barh', figsize=(20,20))

plt.xlabel('Importance of Attributes - Score')

plt.ylabel('Attribute Name')

plt.title("Attribute Importance by RandomForest Application")
sub_df = test_df[['ID']]

# # Fill the target variable with the predictions

sub_df['Won_Championship'] = pred_test['Won_Championship']

# # # Converting the submission file to csv format

sub_df.to_csv('submission.csv', index=False)
sub_df.shape
sub_df.head()