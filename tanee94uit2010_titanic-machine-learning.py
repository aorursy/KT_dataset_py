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



# importing required libraries

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor
# importing train, test and gender_submission.csv file

titanic_train = pd.read_csv('/kaggle/input/train.csv')

titanic_test = pd.read_csv('/kaggle/input/test.csv')

gender_submission = pd.read_csv('/kaggle/input/gender_submission.csv')
# number of rows and columns in train set  

titanic_train.shape
# looking the spread of numerical variables of train set

titanic_train.describe()
# column names

titanic_train.columns
# train data information

titanic_train.info()
# plotting the number of people who survived

sns.countplot('Survived', data = titanic_train)

plt.show()
# percentage of passengers who survived

round((sum(titanic_train['Survived'])/len(titanic_train.index))*100, 2)
# Survived passengers based on Pclass and Sex

plt.figure(figsize = (14,13))

plt.subplot(2,2,1)

sns.countplot('Pclass', data = titanic_train)

plt.title('Count of Pclass')

plt.subplot(2,2,2)

titanic_train.groupby('Pclass')['Survived'].value_counts().sort_values(ascending = False).plot('bar')

plt.ylabel('Count')

plt.title('Pclass v/s Survived')

plt.subplot(2,2,3)

sns.countplot('Sex', data = titanic_train)

plt.title('Count of Male and Female')

plt.subplot(2,2,4)

titanic_train.groupby('Sex')['Survived'].value_counts().sort_values(ascending = False).plot('bar')

plt.ylabel('Count')

plt.title('Sex v/s Survived')

plt.show()
# SipSb and Parch

plt.figure(figsize = (14,13))

plt.subplot(2,2,1)

sns.countplot('SibSp', data = titanic_train)

plt.title('Count of SibSp')

plt.subplot(2,2,2)

titanic_train.groupby('SibSp')['Survived'].value_counts().sort_values(ascending = False).plot('bar')

plt.ylabel('Count')

plt.title('SibSp v/s Survived')

plt.subplot(2,2,3)

sns.countplot('Parch', data = titanic_train)

plt.title('Count of Parch')

plt.subplot(2,2,4)

titanic_train.groupby('Parch')['Survived'].value_counts().sort_values(ascending = False).plot('bar')

plt.ylabel('Count')

plt.title('Parch v/s Survived')

plt.show()
# spread of Fare and how Fare affected the possibility of survival

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)

sns.distplot(titanic_train.Fare)

plt.title('Histogram of Fare')

plt.subplot(1,2,2)

sns.scatterplot(y = 'Fare', x = 'Survived', data = titanic_train)

plt.title('Fare v/s Survived')

plt.show()
sns.lmplot(x = 'Age', y = 'Fare', hue='Pclass', col='Survived', data = titanic_train)

plt.ylim(0,300)
# Finding the number of missing values in the data set

titanic_train.isnull().sum()
# Finding percentage of missing values

round((titanic_train.isnull().sum()/len(titanic_train.index))*100, 2)
# Imputing the missing value of Age with it's median

titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].median())
# counting the different values in Embarked column

titanic_train['Embarked'].value_counts()
# Inputing the missing values of Embarked coloumn with it's mode 

titanic_train['Embarked'] = titanic_train['Embarked'].fillna('S')
# since Cabin has 77.10% of missing values therefore dropping the entire column

titanic_train = titanic_train.drop('Cabin', 1)

titanic_train.head()
# Again looking at the column for any missing values

round((titanic_train.isnull().sum()/len(titanic_train.index))*100, 2)
# Correlation among the columns

titanic_train.corr()
# visualizing the correlation of train data set

sns.heatmap(titanic_train.corr(), annot = True)

plt.show()
# pairplot 

sns.pairplot(titanic_train)

plt.show()
# Looking for outliers if any

titanic_train.describe()
sns.boxplot(y = 'Fare', data = titanic_train)

plt.show()
# Looking the train data set

titanic_train.head()
# dropping Name and Ticket columns as they doesn't seem much intuitive

titanic_train = titanic_train.drop('Name', 1)

titanic_train = titanic_train.drop('Ticket', 1)

titanic_train.head()
# creating dummy variables

def dummies(x,df):

    temp = pd.get_dummies(df[x], prefix = x, drop_first = True)

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df



titanic_train = dummies('Pclass', titanic_train)

titanic_train = dummies('Sex', titanic_train)

titanic_train = dummies('Embarked', titanic_train)
# dummy variable created

titanic_train.head()
# feature scaling

scaler = MinMaxScaler()



num_vars = ['Age', 'Fare']



titanic_train[num_vars] = scaler.fit_transform(titanic_train[num_vars])

titanic_train.head()
# splitting independent and dependent variables

y_train = titanic_train.pop('Survived')

X_train = titanic_train 
# Looking test set

titanic_test.head()
# Looking for null values if any

titanic_test.isnull().sum()
# Finding percentage of missing values

round((titanic_test.isnull().sum()/len(titanic_test.index))*100, 2)
# imputing missing value of Age with its median

titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())
# drooping Cabin column

titanic_test = titanic_test.drop('Cabin', 1)
# imputing missing value of Fare with its mean

titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test.Fare.mean())
# again looking the test set for null values if any

titanic_test.isnull().sum()
# dropping Name and Ticket columns as done for the train data set

titanic_test = titanic_test.drop('Name', 1)

titanic_test = titanic_test.drop('Ticket', 1)

titanic_test.head()
# creating dummies

def dummies(x,df):

    temp = pd.get_dummies(df[x], prefix = x, drop_first = True)

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df



titanic_test = dummies('Pclass', titanic_test)

titanic_test = dummies('Sex', titanic_test)

titanic_test = dummies('Embarked', titanic_test)
# Final test set

titanic_test.head()
logm1 = sm.GLM(y_train, sm.add_constant(X_train), family = sm.families.Binomial())

logm1.fit().summary()
logreg = LogisticRegression()

rfe = RFE(logreg, 6) # selecting 6 variables

rfe = rfe.fit(X_train, y_train)
# features selected after RFE are:

X_train.columns[~rfe.support_]
# rank wise features:

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
X_train.columns[~rfe.support_]
cols = X_train.columns[rfe.support_]
X_train.columns
# dropping the variables not selected after RFE

X_train = X_train[cols]

X_train.columns
X_train_sm = sm.add_constant(X_train)

logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
X_train = X_train.drop('Fare', 1)

X_train.columns
X_train_sm = sm.add_constant(X_train)

logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
# Check for the VIF values of the feature variables. 

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = 'VIF', ascending = True)

vif
# getting predited values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:5]
y_train_pred_final = pd.DataFrame({'Survived': y_train.values, 'Survived_Prob': y_train_pred.values})

y_train_pred_final['PassengerID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# confusion matrix

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.Predicted)

print(confusion)
#Actual/Predicted     not_survived    survived

#not_survived              463          86

#survived                  101          241 
# overall accuracy

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting survived when passenger does not survived

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.Survived_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survived_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['Final_Predicted'] = y_train_pred_final.Survived_Prob.map( lambda x: 1 if x > 0.4 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Final_Predicted))
confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.Final_Predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train_pred_final.Survived, y_train_pred_final.Final_Predicted))

print(recall_score(y_train_pred_final.Survived, y_train_pred_final.Final_Predicted))
from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.Survived_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
titanic_test.head()
num_vars = ['Age', 'Fare']



titanic_test[num_vars] = scaler.transform(titanic_test[num_vars])

titanic_test.head()
X_test = titanic_test[['Pclass_2', 'Pclass_3', 'Sex_male', 'Age', 'Embarked_S']]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)

y_test_pred.head()
# creating dataframe 

y_test_pred_final = pd.DataFrame()

y_test_pred_final['Survived_Prob'] = y_test_pred.values

y_test_pred_final.head()
# Putting PassengerId to index

y_test_pred_final['PassengerId'] = titanic_test.PassengerId.values

y_test_pred_final.head()
# Let's see the head of y_pred_final

y_test_pred_final['Final_Predicted'] = y_test_pred_final.Survived_Prob.map( lambda x: 1 if x > 0.4 else 0)

y_test_pred_final.head()
y_test_pred_final = pd.merge(y_test_pred_final, gender_submission, on='PassengerId', how='inner')

y_test_pred_final.head()
metrics.accuracy_score(y_test_pred_final.Survived, y_test_pred_final.Final_Predicted)
confusion2 = metrics.confusion_matrix(y_test_pred_final.Survived, y_test_pred_final.Final_Predicted)

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
submission = y_test_pred_final[['PassengerId','Final_Predicted']]

submission = submission.rename(columns={'Final_Predicted': 'Survived'})

submission.to_csv("submission.csv", index=False)

submission.head()