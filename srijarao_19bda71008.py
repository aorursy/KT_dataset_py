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
#import the necessary packages

import seaborn as sns

import matplotlib.pyplot as plt
#Path of the train csv file to read

df=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")

df
#prints the first five rows of the data

df.head()
# to print a summary of the dataframe

df.info()
#retrieve columns in a list

feat=df.columns

feat
#check for missing values

df.isnull().sum()
# to find the shape of  a dataframe

df.shape
# generates descriptive statistics that summarises the central tendancy

df.describe()
sns.pairplot(df)
#plots the count of each class in the flag

sns.countplot(df['flag'])
# retrieves target variable from the data and store in a variable

y=df['flag']

y
#find the correlation of flag variable with other variables

df.corr()['flag']
#dropping the target variable from the data

x=df.drop(['flag'],axis=1)

x
#splitting the dataset into train and test

import sklearn.model_selection as ms

x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.2)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
#importing metrics for calculating accuracy of the model

from sklearn import metrics
#for using Decision Tree Algoithm

from sklearn.tree import DecisionTreeClassifier 

#import f1 score package from sklearn library

from sklearn.metrics import f1_score
#importing confusion matrix,precision,recall packages

from sklearn.metrics import confusion_matrix,precision_score,recall_score
# the overall accuracy of Decision tree model

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

acc_dt = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the Decision Tree is', acc_dt)
#f1 score of Decision tree model

f1_score(y_pred,y_test)
# precision of Decision tree model

precision_score(y_pred,y_test)
#recall of Decision tree model

recall_score(y_pred,y_test)
# confusion matrix of the model

cm=confusion_matrix(y_test,y_pred)

cm
#plot the confusion matrix

fig, ax = plt.subplots(figsize=(4, 4))

ax.imshow(cm)

ax.grid(False)

ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))

ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))

ax.set_ylim(1.5, -0.5)

for i in range(2):

    for j in range(2):

        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.show()
# path of test csv file to read

df1=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")
#retrieve the features of test csv in a list

feat1=df1.columns

feat1
# assign those features to a variable

test_x=df1[feat1]

test_x
# predict the values of target variable 

df1['flag']=dt.predict(test_x)
#path of the submission csv file

subm=pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
# predicts the 

subm['flag'] = df1['flag']

subm.to_csv("decision_tree.csv",index=False)