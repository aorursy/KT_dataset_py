import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
submission_df = pd.read_csv('../input/titanic/gender_submission.csv')

submission_df.head()
train_df = pd.read_csv("../input/titanic/train.csv")

train_df.head()
train_df.shape
train_df.describe()
train_df.info()
test_df = pd.read_csv("../input/titanic/test.csv")

test_df.head()
test_df.info()
test_df.shape
test_df.describe()
train_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
# Imputing the missing data for AGE, Cabin 



train_df.loc[np.isnan(train_df['Age']), ['Age']] = train_df['Age'].mean()

train_df.loc[pd.isnull(train_df['Cabin']), ['Cabin']] = 'Others'



test_df.loc[np.isnan(test_df['Age']), ['Age']] = test_df['Age'].mean()

test_df.loc[pd.isnull(test_df['Cabin']), ['Cabin']] = 'Others'



check_df = test_df[test_df['Pclass'] == 3]

test_df.loc[np.isnan(test_df['Fare']), ['Fare']] = check_df['Fare'].mean()
train_df.isnull().sum()
train_df.head()
train_df.SibSp.value_counts()
train_df.describe()
sns.heatmap(train_df.corr())
train_df = train_df[~pd.isnull(train_df['Embarked'])]

test_df = test_df[~pd.isnull(test_df['Embarked'])]

train_df.shape
train_df.head()
train_df.drop(['PassengerId', 'Name'], axis=1, inplace=True)

train_df.head()
train_df.shape

train_df.head()
test_df.shape

test_df.head()
merged_df = pd.concat([train_df, test_df])

merged_df.drop(columns=['PassengerId', 'Name'], inplace=True)

merged_df.head()
train_df.shape
test_df.shape
merged_df.shape
dummies = pd.get_dummies(merged_df['Sex'], drop_first=True).rename(columns=lambda x: 'Sex_' + str(x))

dummies.head()
merged_df = pd.concat([merged_df, dummies], axis=1)

merged_df.drop(['Sex'], axis=1, inplace=True)

merged_df.head()
def creatingdummies(columnname, dropfirst, train_df):

    dummies = pd.get_dummies(train_df[columnname], drop_first=dropfirst).rename(columns=lambda x: columnname + '_' + str(x))

    train_df = pd.concat([train_df, dummies], axis=1)

    train_df.drop([columnname], axis=1, inplace=True)

    return train_df



merged_df.Pclass.value_counts()
merged_df = creatingdummies('Pclass', False, merged_df)

merged_df.head()
merged_df.drop(['Ticket'], axis=1, inplace=True)

merged_df.head()
merged_df = creatingdummies('Cabin', False, merged_df)

merged_df.head()
merged_df = creatingdummies('Embarked', False, merged_df)

merged_df.head()
merged_df.head()
train_df.shape
test_df.shape
X_train = merged_df[~pd.isnull(merged_df['Survived'])]

X_train.shape
y_train = X_train['Survived']

y_train.head()
X_train = X_train.drop('Survived', axis=1)

X_train.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])

X_train.head()
y_train.shape
X_train.shape
import statsmodels.api as sm
model1 = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Binomial())

model1.fit().summary()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)

rfe = rfe.fit(X_train, y_train)
rfe.support_
col = X_train.columns[rfe.support_]

col
X_train = X_train[X_train.columns[rfe.support_]]

X_train.head()
X_train_sm = sm.add_constant(X_train)

model2 = sm.GLM(y_train, X_train_sm, families=sm.families.Binomial)

model2.fit().summary()
X_train.drop('Cabin_B96 B98', axis=1, inplace = True)
X_train_sm = sm.add_constant(X_train)

model2 = sm.GLM(y_train, X_train_sm, families=sm.families.Binomial)

model2.fit().summary()
X_train.drop('Cabin_G6', axis=1, inplace = True)
X_train_sm = sm.add_constant(X_train)

model2 = sm.GLM(y_train, X_train_sm, families=sm.families.Binomial)

model2.fit().summary()
X_train.drop('Cabin_E121', axis=1, inplace = True)
X_train_sm = sm.add_constant(X_train)

model2 = sm.GLM(y_train, X_train_sm, families=sm.families.Binomial)

model2.fit().summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('Age', axis=1, inplace = True)

X_train_sm = sm.add_constant(X_train)

model2 = sm.GLM(y_train, X_train_sm, families=sm.families.Binomial)

res = model2.fit()

model2.fit().summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred_final = pd.DataFrame({'Actual':y_train.values, 'Predicted':y_train_pred})

y_train_pred_final.head()
y_train_pred_final['predicted_val'] = y_train_pred_final.Predicted.map(lambda x: 1 if x > 0.45 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics

confusion = metrics.confusion_matrix(y_train_pred_final.Actual, y_train_pred_final.predicted_val)

confusion
metrics.accuracy_score(y_train_pred_final.Actual, y_train_pred_final.predicted_val)
from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_train_pred_final.Actual, y_train_pred_final.Predicted)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
y_train_pred_final['48_predicted_val'] = y_train_pred_final.Predicted.map(lambda x: 1 if x > 0.48 else 0)



# Let's see the head

y_train_pred_final.head()
metrics.accuracy_score(y_train_pred_final.Actual, y_train_pred_final['48_predicted_val'])
numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Predicted.map(lambda x: 1 if x > i else 0)

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

    cm1 = metrics.confusion_matrix(y_train_pred_final.Actual, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Actual, y_train_pred_final['48_predicted_val'], drop_intermediate = False )
draw_roc(y_train_pred_final.Actual, y_train_pred_final['48_predicted_val'])
X_test =merged_df[pd.isnull(merged_df['Survived'])]

X_test.head()
X_test.shape
X_train.columns
X_test = X_test[X_train.columns]

X_test.head()
y_test_pred = res.predict(sm.add_constant(X_test)).values.reshape(-1)

y_test_pred[:20]
y_test_pred_df = pd.DataFrame(y_test_pred)

y_test_pred_df= y_test_pred_df.rename(columns={ 0 : 'Prob'})

y_test_pred_df.head()
test_df.shape
test_df['Survived'] = y_test_pred_df.Prob.map(lambda x: 1 if x > 0.48 else 0)

test_df.head()
final_csv_df = test_df[['PassengerId', 'Survived']]

final_csv_df.head()
final_csv_df.to_csv('Result.csv', index=False)