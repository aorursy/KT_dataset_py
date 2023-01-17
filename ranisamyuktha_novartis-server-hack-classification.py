# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score # accuracy score

from sklearn.metrics import recall_score   # recall score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#loading the train data

train = pd.read_csv("/kaggle/input/novartis-data/Train.csv")

#Shape of train

print(train.shape) #printing the shape of train

print(train.describe()) #printing the statistics of train

print(train.info()) #printing the information of train

print(train.head()) #printing the first five rows of the train data
#loading the test data

test = pd.read_csv("/kaggle/input/novartis-data/Test.csv")

#Shape of train

print(test.shape) #printing the shape of train

print(test.describe())#printing the statistics of train

print(test.info()) #printing the information of train

print(test.head()) #printing the first five rows of the test data
#Dropping the Incident_ID and Date from train data

train = train.drop(['INCIDENT_ID','DATE'], axis=1)

train.info()
#Verifying all the columns that has the null values in train data

null_columns=train.columns[train.isnull().any()]

train[null_columns].isnull().sum()
#Filled NaN values with "0" using fillna()

train["X_12"].fillna(0,inplace = True)

train.isnull().sum()
#Verifying all the columns that has the null values in test data

null_columns=test.columns[test.isnull().any()]

test[null_columns].isnull().sum()
#Filled NaN values with "0" using fillna()

test["X_12"].fillna(0,inplace = True)

#train["X_12"].ffill(axis = "rows")

test.isnull().sum()
train["X_12"] = train["X_12"].astype(np.int64)

train.info()
test["X_12"] = test["X_12"].astype(np.int64)

test.info()
#Removing the duplicated rows from train data

print("Train shape before removing the duplicates :" , train.shape)

train.drop_duplicates(keep='first', inplace=True)

print("Train shape After removing the duplicates :" , train.shape)
#Skewness of the train data

train.skew()
import matplotlib.pyplot as plt

import seaborn as sns

#Skewness of train data

sns.distplot(train.skew(),color='blue',axlabel ='Skewness')
f,ax = plt.subplots(1,1,figsize=(16,6))

sns.violinplot(train['MULTIPLE_OFFENSE'])

plt.show()

#skewness and kurtosis

print("Skewness: {}".format(train['MULTIPLE_OFFENSE'].skew()))

print("Kurtosis: {}".format(train['MULTIPLE_OFFENSE'].kurt()))
print("Number of training Mutiple Offence : {} ".format(len(train)))

print("Offense Rate {:.4}%".format(train["MULTIPLE_OFFENSE"].mean()*100))
#Creating Pie Chart for the target variable

labels = ['Hacked', 'Genuine']

plt.title('Multiple Offense')

train['MULTIPLE_OFFENSE'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',shadow=True,labels=labels,fontsize=10)
#histogram

train.hist(figsize=(14,14))

plt.show()
#Boxplot 

plt.subplots(figsize=(15, 6))

sns.boxplot(data = train, orient = 'v')
# create a correlation heatmap

sns.heatmap(train.corr(),annot=True, cmap='gist_ncar', linewidths=0.1)

fig=plt.gcf()

fig.set_size_inches(14,14)

plt.show()
#High Correlation of X_2 and X_3 using joint plot

sns.jointplot(train['X_2'],train['X_3'], kind="reg", color="b")
X_train = train.iloc[:,:-1]

y_train = train["MULTIPLE_OFFENSE"]

#Dropping the Incident_ID and Date from test data

X_test = test.drop(['INCIDENT_ID','DATE'], axis=1)

print("Shape of X_train : ",X_train.shape)

print("Shape of y_train : ",y_train.shape)

print("Shape of X_test : ",X_test.shape)
#Synthetic minority oversampling technique to balance the imbalanced data.

print('Before OverSampling, the shape of X_train: {}'.format(X_train.shape))

print('Before OverSampling, the shape of y_train: {} \n'.format(y_train.shape))

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

from imblearn.over_sampling import SMOTE

sampler = SMOTE(sampling_strategy='minority')

X_train_sm, y_train_sm = sampler.fit_sample(X_train, y_train)

print('After OverSampling, the shape of X_train: {}'.format(X_train_sm.shape))

print('After OverSampling, the shape of y_train: {} \n'.format(y_train_sm.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_sm==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_sm==0)))
#Spltting the data into train and validation

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train_sm, y_train_sm ,test_size=0.3, random_state=10)
#Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)

X_val_sc   = sc.transform(X_val)

X_test_sc  = sc.transform(X_test)
# Support Vector Classification

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

svc = SVC()

svc.fit(X_train_sc, y_train)

y_pred = svc.predict(X_val_sc)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

recall_svc = recall_score(y_pred, y_val)

print("Support Vector Classifier Accuracy Score:",acc_svc)

print('Support Vector Classifier Recall Score:',recall_svc)
#Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

gbk = GradientBoostingClassifier()

gbk.fit(X_train_sc, y_train)

y_pred = gbk.predict(X_val_sc)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

recall_gbk = recall_score(y_pred, y_val)

print("Gradient Boosting Classifier Accuracy Score:",acc_gbk)

print('Gradient Boosting Classifier Recall Score:',recall_gbk)
#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

randomforest = RandomForestClassifier()

randomforest.fit(X_train_sc, y_train)

y_pred = randomforest.predict(X_val_sc)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

recall_randomforest = recall_score(y_pred, y_val)

print("Random Forest Classifier Accuracy Score:",acc_randomforest)

print('Random Forest Classifier Recall Score:',recall_randomforest)
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

decisiontree = DecisionTreeClassifier()

decisiontree.fit(X_train_sc, y_train)

y_pred = decisiontree.predict(X_val_sc)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

recall_decisiontree = recall_score(y_pred, y_val)

print("Decision Tree Accuracy Score:",acc_decisiontree)

print('Decision Tree Recall Score:',recall_decisiontree)
# KNN or k-Nearest Neighbors Classifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train_sc, y_train)

y_pred = knn.predict(X_val_sc)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

recall_knn = recall_score(y_pred, y_val)

print("KNN Classifier Accuracy Score:",acc_knn)

print('KNN Classifier Recall Score:',recall_knn)
models = pd.DataFrame({

    'Model': ['Support Vector Machine', 

              'Random Forest', 

              'Decision Tree',

              'Gradient Boosting Classifier',

               'k-Nearest Neighbors Classifier'],

    'Accuracy Score': [acc_svc,acc_gbk, acc_randomforest,acc_decisiontree,acc_knn],

    'Recall Score'  :  [recall_svc,recall_gbk,recall_randomforest,recall_decisiontree,recall_knn]})

models.sort_values(by='Accuracy Score', ascending=False)
#I have chosen Gradient Boosting classifier amongst all classifiers

y_pred = gbk.predict(X_test_sc)

submission_df = pd.DataFrame({'INCIDENT_ID':test['INCIDENT_ID'], 'MULTIPLE_OFFENSE':y_pred})

submission_df.to_csv('Sample Submission GBK v1.csv', index=False)