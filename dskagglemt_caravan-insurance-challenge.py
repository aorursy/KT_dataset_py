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
df = pd.read_csv("/kaggle/input/caravan-insurance-challenge/caravan-insurance-challenge.csv")
df.head()
df.describe()
df.shape
df.info()
# Lets get the % of each null values.

total = df.isnull().sum().sort_values(ascending=False)

percent_1 = df.isnull().sum()/df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head()
import matplotlib.pyplot as plt

import seaborn as sns
# Visualizing the NULL data using Seaborn HeatMap.

sns.heatmap(df.isnull(), yticklabels = False, cbar = False)
categ_features = df.select_dtypes(include='object').columns

numeric_features = df.select_dtypes(include='int').columns

display(categ_features, numeric_features, len(categ_features), len(numeric_features))
df['ORIGIN'].unique()

# So this df data-set contains both train and test data.
# Get the count of Train and Test data.

df['ORIGIN'].value_counts()
# Get the count of Train and Test data.

df['CARAVAN'].value_counts()
#Using Pearson Correlation



plt.figure(figsize=(20,10))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
plt.figure(figsize=(20,5))

df.hist(column=numeric_features[1:5])

plt.show()
# Split DataFrame in train and test

train_df = df.loc[:5821]

test_df = df.loc[5822:]

display(train_df['ORIGIN'].unique(), test_df['ORIGIN'].unique())
# Get the count of Train and Test data.

display(train_df['CARAVAN'].value_counts(), test_df['CARAVAN'].value_counts())
# Split Train Data-set into train and valid data-sets, on which we will train our model. Before that will get our X and y.



X = train_df.drop(['CARAVAN'], axis = 1)

 

y = train_df['CARAVAN']

 

display(X.head(), y.head())    
# split the train_df into 2 DF's aka X_train, X_valid, y_train, y_valid.

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)



print (X_train.shape, y_train.shape)

print (X_valid.shape, y_valid.shape)

print(test_df.shape)
# test_df 

X_test  = test_df.drop(['CARAVAN'], axis = 1)

y_test = test_df['CARAVAN']

print (X_test.shape, y_test.shape)
# Now we do have proper data splitting.. will drop the column 'ORIGIN'.

X_train.drop('ORIGIN', inplace = True, axis = 1)

X_valid.drop('ORIGIN', inplace = True, axis = 1)

X_test.drop('ORIGIN', inplace = True, axis = 1)



print (X_train.shape, y_train.shape)

print (X_valid.shape, y_valid.shape)

print (X_test.shape, y_test.shape)
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Model Performance matrix

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, classification_report
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)
# Predict the model



Y_valid_pred_lr = logreg.predict(X_valid)

Y_test_pred_lr = logreg.predict(X_test)
print("LogisticRegression (on Train and Valid Data-Set) --> ")

print("Score : ", round(logreg.score(X_train, y_train) * 100, 2) )



print("Accuracy Score : ", round(accuracy_score(y_valid, Y_valid_pred_lr) * 100, 2) )



print("Confusion Matrix : " )

display( confusion_matrix(y_valid, Y_valid_pred_lr) )



print("ROC AUC Score : ", roc_auc_score(y_valid, Y_valid_pred_lr) )

print("LogisticRegression (Test Data-Set) --> ")

print("Score : ", round(logreg.score(X_test, y_test) * 100, 2) )



print("Accuracy Score : ", round(accuracy_score(y_test, Y_test_pred_lr) * 100, 2) )



print("Confusion Matrix : " )

display( confusion_matrix(y_test, Y_test_pred_lr) )



print("ROC AUC Score : ", roc_auc_score(y_test, Y_test_pred_lr) )