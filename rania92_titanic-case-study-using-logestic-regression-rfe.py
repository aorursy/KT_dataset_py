# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing Pandas and NumPy

import pandas as pd

import numpy as np
# Importingdatasets

train = pd.read_csv("../input/train.csv")

train.head()
# Importingdatasets

test = pd.read_csv("../input/test.csv")

test.head()
test1 = test['PassengerId']

test1.head()
test1 = test1.to_frame(name=None)
test1 = test1.set_index('PassengerId')
train.columns
# Let's check the dimensions of the dataframe

train.shape
test.shape
# let's look at the statistical aspects of the dataframe

train.describe()
# Let's see the type of each column

train.info()
# List of variables to map



varlist1 =  ['Sex']



# Defining the map function

def binary_map(x):

    return x.map({'male': 1, "female": 0})



# Applying the function to the train list

train[varlist1] = train[varlist1].apply(binary_map)
train.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(train[['Embarked', 'Pclass']], drop_first=True)



# Adding the results to the dataframe

train = pd.concat([train, dummy1], axis=1)
train.head()
# We have created dummies for the below variables, so we can drop them

train = train.drop(['Embarked', 'Pclass','Name', 'Ticket', 'Fare', 'Cabin'], 1)
train.info()
# Adding up the missing values (column-wise)

train.isnull().sum()
# Checking the percentage of missing values

round(100*(train.isnull().sum()/len(train.index)), 2)
train["Age"] = train["Age"].fillna(value=train["Age"].mean()) #replace NaN by mean
# Adding up the missing values (column-wise)

train.isnull().sum()
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = train.drop(['Survived','PassengerId'], axis=1)



X.head()
# Putting response variable to y

y = train['Survived']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_train[['Age']] = scaler.fit_transform(X_train[['Age']])



X_train.head()
### Checking the survival Rate

survived = (sum(train['Survived'])/len(train['Survived'].index))*100

survived
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Let's see the correlation matrix 

plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(train.corr(),annot = True)

plt.show()
plt.figure(figsize = (20,10))

sns.heatmap(X_train.corr(),annot = True)

plt.show()
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 5)             # running RFE with 13 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'survival_prob':y_train_pred})

y_train_pred_final['PassengerId'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.survival_prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )

print(confusion)
# Predicted     not_churn    churn

# Actual

# not_churn        3270      365

# churn            579       708  
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
# Let's drop TotalCharges since it has a high VIF

col = col.drop('Parch')

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
# Let's take a look at the confusion matrix again 

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )

confusion
# Actual/Predicted     not_survived    survived

        # not_churn        3269      366

        # churn            595       692  
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted)
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate

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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.survival_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Survived, y_train_pred_final.survival_prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.survival_prob.map(lambda x: 1 if x > i else 0)

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
y_train_pred_final['final_predicted'] = y_train_pred_final.survival_prob.map( lambda x: 1 if x > 0.25 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted )

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
#Looking at the confusion matrix again
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )

confusion
confusion[1,1]/(confusion[0,1]+confusion[1,1])
confusion[1,1]/(confusion[1,0]+confusion[1,1])
from sklearn.metrics import precision_score, recall_score
precision_score
precision_score(y_train_pred_final.Survived, y_train_pred_final.predicted)
recall_score(y_train_pred_final.Survived, y_train_pred_final.predicted)
from sklearn.metrics import precision_recall_curve
y_train_pred_final.Survived, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.survival_prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[['Age']] = scaler.transform(X_test[['Age']])
X_test = X_test[col]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Putting PassengerId to index

y_test_df['PassengerId'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'survival_prob'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex_axis(['PassengerId','Survived','survival_prob'], axis=1)
# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.survival_prob.map(lambda x: 1 if x > 0.4 else 0)
y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Survived, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Survived, y_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
test.isnull().sum()
test["Age"] = test["Age"].fillna(value=test["Age"].mean()) #replace NaN by mean
test.isnull().sum()
# List of variables to map



varlist =  ['Sex']



# Defining the map function

def binary_map(x):

    return x.map({'male': 1, "female": 0})



# Applying the function to the train list

test[varlist] = test[varlist].apply(binary_map)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy = pd.get_dummies(test[['Embarked', 'Pclass']], drop_first=True)



# Adding the results to the dataframe

test = pd.concat([test, dummy], axis=1)
# We have created dummies for the below variables, so we can drop them

test = test.drop(['Embarked', 'Pclass','Name', 'Ticket', 'Fare', 'Cabin'], 1)
test[['Age']] = scaler.transform(test[['Age']])
test = test[col]

test.head()
test_sm = sm.add_constant(test)
test_pred = res.predict(test_sm)
test_pred[:10]
# Converting y_pred to a dataframe which is an array

pred_1 = pd.DataFrame(test_pred)

# Let's see the head

pred_1.head()
# Converting y_test to dataframe

test_df1 = pd.DataFrame(test)
# Putting PassengerId to index

test_df1['PassengerId'] = test1.index
# Removing index for both dataframes to append them side by side 

pred_1.reset_index(drop=True, inplace=True)

test_df1.reset_index(drop=True, inplace=True)
# Appending test_df and pred_1

pred_final1 = pd.concat([test_df1, pred_1],axis=1)

pred_final1.head()
# Renaming the column 

pred_final1= pred_final1.rename(columns={ 0 : 'survival_prob'})
# Rearranging the columns

pred_final1 = pred_final1.reindex_axis(['PassengerId','survival_prob'], axis=1)

# Let's see the head of pred_final1

pred_final1.head()
pred_final1['Survived'] = pred_final1.survival_prob.map(lambda x: 1 if x > 0.4 else 0)

pred_final1.head()
gender_submission = pred_final1.drop('survival_prob', axis=1)

gender_submission.head()
submission = pred_final1[['PassengerId','Survived']]



submission.to_csv("submission.csv", index=False)
