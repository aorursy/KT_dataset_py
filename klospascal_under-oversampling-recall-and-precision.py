import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, classification_report

from sklearn.metrics import confusion_matrix



import warnings  

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/creditcard.csv')

df.head()
#Checking for missing values

if df.isnull().any().unique() == False:

    print("No missing values")

else:

    print(df.isnull().any())
df["Amount"].mean()
#Outliers detection

sns.boxplot(df["Amount"], orient='v')

plt.show()
#Remove Outliers

def remove_outlier(col):

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = q3-q1

    #Cut-Off

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    #Remove Outliers

    df2 = df.loc[(df[col] > fence_low) & (df[col] < fence_high)]

    return df2

num_rows_org = df.shape[0]

columns = ["Amount"]

for i in columns:

    df = remove_outlier(i) 

num_rows_new = df.shape[0]

print(num_rows_org-num_rows_new,"Outliers Removed")
sns.boxplot(df["Amount"], orient='v')

plt.show()
sns.distplot(df["Time"])

plt.show()
corr = df.corr()

sns.heatmap(corr)
#Imbalanced Data

sns.countplot(df["Class"])

plt.show()
x = df.drop(["Class"], axis = 1)

y = df["Class"]



x_train, x_test, y_train, y_test = train_test_split(x,y)



model = LogisticRegression()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Model Score:', model.score(x_test, y_test))



roc_auc = roc_auc_score(y_test, pred)

print('ROC_AUC_Score',roc_auc)



print(classification_report(y_test, pred))
model = RandomForestClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Model Score:', model.score(x_test, y_test))



roc_auc = roc_auc_score(y_test, pred)

print('ROC_AUC_Score',roc_auc)



print(classification_report(y_test, pred))



con = confusion_matrix(y_test, pred)

sns.heatmap(con, annot=True)

plt.show()
from imblearn.under_sampling import RandomUnderSampler



x = df.drop(["Class"], axis = 1)

y = df["Class"]



x_train, x_test, y_train, y_test = train_test_split(x,y)



rus = RandomUnderSampler(return_indices=True)

x_train, y_train, id_rus = rus.fit_sample(x_train, y_train)



sns.countplot(y_train)

plt.show()
model = RandomForestClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Model Score:', model.score(x_test, y_test))



roc_auc = roc_auc_score(y_test, pred)

print('ROC_AUC_Score',roc_auc)



print(classification_report(y_test, pred))



con = confusion_matrix(y_test, pred)

sns.heatmap(con, annot=True)

plt.show()
from imblearn.over_sampling import RandomOverSampler



x = df.drop(["Class"], axis = 1)

y = df["Class"]



x_train, x_test, y_train, y_test = train_test_split(x,y)



ros = RandomOverSampler(return_indices=True)

x_train, y_train, id_rus = ros.fit_sample(x_train, y_train)



sns.countplot(y_train)

plt.show()
model = RandomForestClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Model Score:', model.score(x_test, y_test))



roc_auc = roc_auc_score(y_test, pred)

print('ROC_AUC_Score',roc_auc)



print(classification_report(y_test, pred))



con = confusion_matrix(y_test, pred)

sns.heatmap(con, annot=True)

plt.show()
from imblearn.over_sampling import SMOTE



x = df.drop(["Class"], axis = 1)

y = df["Class"]



x_train, x_test, y_train, y_test = train_test_split(x,y)



sm = SMOTE(random_state=0)

x_train, y_train = sm.fit_sample(x_train, y_train)



model = RandomForestClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Model Score:', model.score(x_test, y_test))



roc_auc = roc_auc_score(y_test, pred)

print('ROC_AUC_Score',roc_auc)



print(classification_report(y_test, pred))



con = confusion_matrix(y_test, pred)

sns.heatmap(con, annot=True)

plt.show()
from imblearn.combine import SMOTEENN



x = df.drop(["Class"], axis = 1)

y = df["Class"]



x_train, x_test, y_train, y_test = train_test_split(x,y)



sm = SMOTEENN(random_state=0)

x_train, y_train = sm.fit_sample(x_train, y_train)



model = RandomForestClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Model Score:', model.score(x_test, y_test))



roc_auc = roc_auc_score(y_test, pred)

print('ROC_AUC_Score',roc_auc)



print(classification_report(y_test, pred))



con = confusion_matrix(y_test, pred)

sns.heatmap(con, annot=True)

plt.show()