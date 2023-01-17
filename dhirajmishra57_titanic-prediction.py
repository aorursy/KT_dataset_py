# Importing Modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import re

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Scikit learn libraries

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import RFE   # Feature selection module

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score,roc_curve



# Statsmodel Libraries

import statsmodels

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

os.listdir("../input/titanic/")
# Importing Data



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

submission = pd.read_csv('../input/titanic/gender_submission.csv')

train.head()
train.shape
train.info()
train.describe(percentiles=[.25,.45,.5,.6,.75,.85,.9,.95]).T
round(train.isna().sum() / train.shape[0] * 100,2)
# Let's remove the cabin column as it is almost null

def drop_Cabin(df):

    df.drop(columns='Cabin',axis=1,inplace=True)

    return df



drop_Cabin(train)

drop_Cabin(test)

train.head()
def impute_Embarked(df):

    df.Embarked = train['Embarked'].fillna(train['Embarked'].value_counts().index[0])

    return df



impute_Embarked(train)

impute_Embarked(test)



round(train.isna().sum() / train.shape[0] * 100,2)
# We will impute Age with Median value

def impute_age(df):

    df.Age.fillna(train.groupby(by='Sex')['Age'].transform('median'),inplace=True)

    return df



impute_age(train)

impute_age(test)



round(train.isna().sum() / train.shape[0] * 100,2)
train.head()
def del_unwanted_col(df):

    df.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)

    return df



def encoding_Embarked(df):

    dummy = pd.get_dummies(df['Embarked'],drop_first=True,prefix='Embarked')

    df = pd.concat([df,dummy],axis=1)

    df = df.drop(columns='Embarked',axis=1)

    return df



def encoding_Sex(df):

    df['Sex'] = df['Sex'].astype('category').cat.codes

    return df



train = encoding_Embarked(train)

test  = encoding_Embarked(test)

train = del_unwanted_col(train)

test  = del_unwanted_col(test)

train = encoding_Sex(train)

test  = encoding_Sex(test)

train.head()
df_train,df_test = train_test_split(train,test_size=0.3,random_state=100)



# Training set

y_train = df_train.pop('Survived')

X_train = df_train



# Test set

y_test = df_test.pop('Survived')

X_test = df_test



X_train.head()
model = LogisticRegression()

model.fit(X_train,y_train)



print('Accuracy of Logaistic regression classifier on training set : {:.3f}'.format(model.score(X_train,y_train)))

print('Accuracy of Logistic regression classifier on testing set : {:.3f}'.format(model.score(X_test,y_test)))
log = LogisticRegression()

rfe = RFE(log,n_features_to_select=6,verbose=1)



rfe.fit(X_train,y_train)



X_train.loc[:,rfe.support_]

X_test.loc[:,rfe.support_][:5]
# Let's only use the fetaures which was given to us by the RFE

X_train = X_train.loc[:,rfe.support_]

X_test = X_test.loc[:,rfe.support_]
model = LogisticRegression()

model.fit(X_train,y_train)



print('Accuracy of Logaistic regression classifier on training set : {:.3f}'.format(model.score(X_train,y_train)))

print('Accuracy of Logistic regression classifier on testing set : {:.3f}'.format(model.score(X_test,y_test)))
df_train,df_test = train_test_split(train,test_size=0.3,random_state=100)



# Training set

y_train = df_train.pop('Survived')

X_train = df_train



# Test set

y_test = df_test.pop('Survived')

X_test = df_test



X_train.head()
# Adding Constant

X_train_sm = sm.add_constant(X_train) 

X_test_sm = sm.add_constant(X_test)



# Building Model

model1 = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial())

model1 = model1.fit()

print(model1.summary())



vif = pd.DataFrame()

vif["VIF_Factor"] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif["features"] = X_train_sm.columns

vif = vif.sort_values(by = "VIF_Factor", ascending = False)

print(vif)



model1.predict()
# Let's Drop 'Q' and rebuild the model

X_train_sm = X_train_sm.drop(columns='Embarked_Q',axis=1)

X_test_sm = X_test_sm.drop(columns='Embarked_Q',axis=1)



# Building Model

model2 = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial())

model2 = model2.fit()

print(model2.summary())



vif = pd.DataFrame()

vif["VIF_Factor"] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif["features"] = X_train_sm.columns

vif = vif.sort_values(by = "VIF_Factor", ascending = False)

vif
# Let's Drop 'Q' and rebuild the model

X_train_sm = X_train_sm.drop(columns='Embarked_S',axis=1)

X_test_sm = X_test_sm.drop(columns='Embarked_S',axis=1)



# Building Model

model3 = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial())

model3 = model3.fit()

print(model3.summary())



vif = pd.DataFrame()

vif["VIF_Factor"] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif["features"] = X_train_sm.columns

vif = vif.sort_values(by = "VIF_Factor", ascending = False)

vif
# Let's Drop 'Q' and rebuild the model

X_train_sm = X_train_sm.drop(columns='Fare',axis=1)

X_test_sm = X_test_sm.drop(columns='Fare',axis=1)



# Building Model

model4 = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial())

model4 = model4.fit()

print(model4.summary())



vif = pd.DataFrame()

vif["VIF_Factor"] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif["features"] = X_train_sm.columns

vif = vif.sort_values(by = "VIF_Factor", ascending = False)

vif
# Let's Drop 'Q' and rebuild the model

X_train_sm = X_train_sm.drop(columns='Parch',axis=1)

X_test_sm = X_test_sm.drop(columns='Parch',axis=1)



# Building Model

model5 = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial())

model5 = model5.fit()

print(model5.summary())



vif = pd.DataFrame()

vif["VIF_Factor"] = [variance_inflation_factor(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

vif["features"] = X_train_sm.columns

vif = vif.sort_values(by = "VIF_Factor", ascending = False)

vif
# Train prediction

y_train_prob = model5.predict(X_train_sm)

y_train_pred = y_train_prob.apply(lambda x: 1 if x > 0.5 else 0)



#Test prediction

y_test_prob = model5.predict(X_test_sm)

y_test_pred = y_test_prob.apply(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final = pd.DataFrame({'Survival':y_train.values, 'Survival_Prob':y_train_prob})

y_train_pred_final['Predicted'] = y_train_pred

y_train_pred_final['PassengerID'] = y_train.index

y_train_pred_final.head()
# Confusion matrix 

confusion = confusion_matrix(y_train_pred_final.Survival, y_train_pred_final.Predicted )

print(confusion)
auc_score = roc_auc_score(y_train_pred_final.Survival,y_train_pred_final.Survival_Prob)

fpr,tpr,_ = roc_curve( y_train_pred_final.Survival,y_train_pred_final.Survival_Prob,drop_intermediate=False)

plt.figure(figsize=[5,5])

plt.plot(fpr,tpr,label='ROC curve (area = %.2f)' % auc_score)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r-')

plt.show()
numbers = [float(x)/10 for x in range(0,10)]

numbers

for i in numbers:

    y_train_pred_final[i] = y_train_pred_final['Survival_Prob'].map(lambda x : 1 if x > i else 0)

y_train_pred_final.head()
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

cutoff_df



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = confusion_matrix(y_train_pred_final.Survival, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

cutoff_df
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Survival_Prob.map( lambda x: 1 if x > 0.3 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

accuracy_score(y_train_pred_final.Survival, y_train_pred_final.final_predicted)
confusion2 = confusion_matrix(y_train_pred_final.Survival, y_train_pred_final.final_predicted )

confusion2
X_train_sm.head()
y_train_pred_final.head()
# test_sm = sm.add_constant(test)



df = pd.read_csv('../input/titanic/test.csv')

y_test_pred_final = pd.DataFrame()

y_test_pred_final['Testing_Pop_Pred'] = model5.predict(test_sm.drop(columns=['Parch','Fare','Embarked_Q','Embarked_S'],axis=1))

y_test_pred_final['Testing_Pred'] = y_test_pred_final['Testing_Pop_Pred'].map(lambda x : 1 if x > 0.3 else 0)

y_test_pred_final['PassengerID'] = df.PassengerId

y_test_pred_final.head()
my_submission = pd.DataFramea()

my_submission['PassengerId'] = df['PassengerId']

my_submission['Survived'] = y_test_pred_final['']

my_submission.to_csv('my_Submission.csv',sep=',')