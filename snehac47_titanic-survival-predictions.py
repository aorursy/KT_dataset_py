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
import matplotlib.pyplot as plt

import seaborn as sns



# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing all datasets



df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')

y_test=pd.read_csv('../input/titanic/gender_submission.csv')
df_train.shape
df_train.info()
# performing pandas profiling for basic idea of variables

import pandas_profiling

pandas_profiling.ProfileReport(df_train)
#checking missing values

df_train.isnull().sum()
df_train.columns
df_test.isnull().sum()
plt.figure(figsize=(4,5))

sns.countplot(df_train.Survived)

plt.xlabel("Survived Value")

plt.ylabel("Count of Survived")

plt.title("Distribution of Variable-Survived")

plt.show()
plt.figure(figsize=(15,4))

ax1=plt.subplot(1, 3, 1)

ax1=sns.countplot(df_train.Pclass, hue=df_train.Survived, palette='husl')

plt.xlabel("Passenger Class Value")

plt.ylabel("Count of Value")

plt.title("Distribution of Pclass")



ax2=plt.subplot(1, 3, 2)

ax2=sns.countplot(df_train.Sex, hue=df_train.Survived, palette='husl')

plt.xlabel("Sex Value")

plt.ylabel("Count of Value")

plt.title("Distribution of Sex")



ax3=plt.subplot(1, 3, 3)

ax3=sns.countplot(df_train.Embarked, hue=df_train.Survived, palette='husl')

plt.xlabel("Embarked Value")

plt.ylabel("Count of Value")

plt.title("Distribution of Embarked")

plt.show()
plt.figure(figsize=(15,4))

ax1=plt.subplot(1, 3, 1)

ax1=sns.countplot(df_train.SibSp, hue=df_train.Survived, palette='husl')

plt.xlabel("Sibsp Value")

plt.ylabel("Count of Value")

plt.title("Distribution of Sibsp")



ax2=plt.subplot(1, 3, 2)

ax2=sns.countplot(df_train.Parch, hue=df_train.Survived, palette='husl')

plt.xlabel("Parch Value")

plt.ylabel("Count of Value")

plt.title("Distribution of Parch")



ax3=plt.subplot(1, 3, 3)

ax3=sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=df_train,palette='husl')

plt.xlabel("Plcass")

plt.ylabel("Fare")

plt.title("Distribution of Plcass vs Fare")

plt.show()
survived=df_train[df_train.Survived==1]

not_survived=df_train[df_train.Survived==0]
# plots

plt.figure(figsize=(15,5))

ax1=plt.subplot(1, 2, 1)

ax1=plt.hist(x=survived['Age']);

plt.xlabel("Age Values")

plt.ylabel("Count of Value")

plt.title("Distribution of Age for passengers who Survived")



ax2=plt.subplot(1, 2, 2)

ax2=plt.hist(x=not_survived['Age']);

plt.xlabel("Age Values")

plt.ylabel("Count of Value")

plt.title("Distribution of Age for passengers who did not Survive")



plt.show()
df_train.columns
#Imputng Missing value in Train Data



df_train['Age'].fillna(df_train.Age.mean(), inplace=True)

df_train['Embarked'].fillna(df_train.Embarked.mode()[0], inplace=True)

df_train.isnull().sum()
#Imputing Missing value in Test Data 



df_test['Age'].fillna(df_test.Age.mean(), inplace=True)

df_test['Fare'].fillna(df_test.Fare.mean(), inplace=True)

df_test['Embarked'].fillna(df_test.Embarked.mode()[0], inplace=True)

df_test.isnull().sum()
#Removing Coloumn Cabin from Datasets since it has too many missing values



df_train = df_train.drop('Cabin', axis=1)

df_test = df_test.drop('Cabin', axis=1)
#Removing Column Name & Ticket as they have high cardinality(too many unique values) and are not very useful



df_train = df_train.drop(['Name','Ticket'], axis=1)

df_test = df_test.drop(['Name','Ticket'], axis=1)
#Removing Column Passenger ID as it has high cardinality(too many unique values) and is not very useful

df_train = df_train.drop(['PassengerId'], axis=1)
df_train.nunique()
df_test.nunique()
df_train.dtypes
# converting categorical variables

df_train['Sex'] = df_train['Sex'].replace('male',0)

df_train['Sex'] = df_train['Sex'].replace('female',1)
df_test['Sex'] = df_test['Sex'].replace('male',0)

df_test['Sex'] = df_test['Sex'].replace('female',1)
df_train.nunique()
df_test.isnull().sum()
df_train.isnull().sum()
df_train.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(df_train[['Embarked']], drop_first=True)



# Adding the results to the master dataframe

df_train = pd.concat([df_train, dummy1], axis=1)

df_train = df_train.drop(['Embarked'], axis=1)

df_train.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy2 = pd.get_dummies(df_test[['Embarked']], drop_first=True)



# Adding the results to the master dataframe

df_test = pd.concat([df_test, dummy2], axis=1)

df_test = df_test.drop(['Embarked'], axis=1)

df_test.head()
# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%

df_train.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# Putting feature variable to X

X_train = df_train.drop(['Survived'], axis=1)



X_train.head()
# Putting response variable to y

y_train = df_train['Survived']



y_train.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train[['Age','Fare']] = scaler.fit_transform(X_train[['Age','Fare']])



X_train.head()
import statsmodels.api as sm



# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
cols=X_train.columns

cols
cols = cols.drop('Embarked_Q', 1)
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[cols])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
#checking VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train[cols].columns

vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
cols = cols.drop('Parch', 1)
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[cols])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
cols = cols.drop('Fare', 1)
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[cols])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
#checking VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train[cols].columns

vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Survival':y_train.values, 'Survival_Prob':y_train_pred})

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Survival, y_train_pred_final.predicted )

print(confusion)
#Predicted     not_survived    survived



#Actual



#not_survived       458          91



#survived            98         244  
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survival, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting survived when customer did not survive

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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survival, y_train_pred_final.Survival_Prob, drop_intermediate = False )

draw_roc(y_train_pred_final.Survival, y_train_pred_final.Survival_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head(10)
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Survival, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
#### From the curve above, 0.38 is the optimum point to take it as a cutoff probability.
y_train_pred_final['final_predicted'] = y_train_pred_final.Survival_Prob.map( lambda x: 1 if x > 0.36 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Survival, y_train_pred_final.final_predicted)
df_test.head()
# Putting feature variable to X

X_test = df_test.copy()



X_test.head()
X_test[['Age','Fare']] = scaler.transform(X_test[['Age','Fare']])
X_test = X_test[cols]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)

y_pred_1.head()
# Putting PassengerId to index

y_test_df = pd.DataFrame(df_test['PassengerId'])
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Survival_Prob'})

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.36 else 0)

y_pred_final.head()
submission = pd.DataFrame({

        "PassengerId": y_pred_final["PassengerId"],

        "Survived": y_pred_final['final_predicted']

    })

submission.to_csv('submission.csv', index=False)