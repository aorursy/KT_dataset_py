# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.info()
df.describe(include="O")
df.isnull().sum()
plt.figure(figsize=(8,5))

sns.heatmap(df.isnull(), cmap='viridis')
column=df.select_dtypes(include=['object'])

for col in column:

    display(df[col].value_counts())
sns.countplot(x = df['gender'], data=df)
df.gender.value_counts()
df.columns
sns.countplot(x = df['gender'], hue = df['status'], data=df)
df.head()
sns.boxplot(x = df['gender'], y = df['salary'], data=df)
df.groupby('gender')['salary'].mean()
df.columns
sns.countplot(x = df['hsc_b'], hue = df['hsc_s'], data=df)
plt.figure(figsize = (15, 15))

ax=plt.subplot(221)

sns.boxplot(x='status',y='ssc_p',data=df)

ax.set_title('Secondary school percentage')

ax=plt.subplot(222)

sns.boxplot(x='status',y='hsc_p',data=df)

ax.set_title('Higher Secondary school percentage')

ax=plt.subplot(223)

sns.boxplot(x='status',y='degree_p',data=df)

ax.set_title('UG Degree percentage')

ax=plt.subplot(224)

sns.boxplot(x='status',y='mba_p',data=df)

ax.set_title('MBA percentage')

sns.violinplot(x = df['gender'], y = df['salary'], hue = df['workex'], data=df)

plt.title("Gender vs Salary based on work experience")
sns.distplot(df['salary'], bins=50, hist=False)
df.head()
df.drop(['sl_no', 'salary'], axis=1, inplace=True)
df.head()
df['gender'] = df.gender.map({"M" : 0, "F" : 1})

df['ssc_b'] = df.ssc_b.map({"Other" : 0, "Central" : 1})

df['hsc_s'] = df.hsc_s.map({"Commerce" : 0, "Science" : 1, "Arts" : 2})

df['degree_t'] = df.degree_t.map({"Comm&Mgmt" : 0, "Sci&Tech" : 1, "Others" : 2})

df['workex'] = df.workex.map({"No" : 0, "Yes" :1})

df['specialisation'] = df.specialisation.map({"Mkt&Fin" : 0, "Mkt&HR" : 1})

df['status'] = df.status.map({"Not Placed" : 0, "Placed" : 1})
df.head()
df.drop(['ssc_b'], axis=1, inplace=True)
df.drop(['hsc_b'], axis=1, inplace=True)
df.head()
# Creating a correlation matrix



plt.figure(figsize=(10,10))



sns.heatmap(df.corr(), annot=True, cmap="RdYlGn")
df.head()
# Seperating our variables into Independent and Dependent variables



X = df[['ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t','workex', 'specialisation', 'mba_p', 'etest_p']] # Indepepndent variables



y = df['status'] # Dependent variables



from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        pred = clf.predict(X_train)

        print("Train Result:\n===========================================")

        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")

        print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")

        print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(X_train))}\n")

        

    elif train==False:

        pred = clf.predict(X_test)

        print("Test Result:\n===========================================")        

        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")

        print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")

        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
# Splitting the data into train test split



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)



print_score(tree, X_train, y_train, X_test, y_test, train=True)

print_score(tree, X_train, y_train, X_test, y_test, train=False)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=200,criterion='gini',max_depth= 4, max_features= 'auto',random_state=42)

rf.fit(X_train, y_train)



print_score(rf, X_train, y_train, X_test, y_test, train=True)

print_score(rf, X_train, y_train, X_test, y_test, train=False)
from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(X_train, y_train)



print_score(xgb, X_train, y_train, X_test, y_test, train=True)

print_score(xgb, X_train, y_train, X_test, y_test, train=False)
rf_probs = rf.predict_proba(X_test)[:,1]

dtree_probs = tree.predict_proba(X_test)[:,1]

xgb_probs = xgb.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score
print('roc_auc_score for Random forest: ', roc_auc_score(y_test, rf_probs))

print('roc_auc_score for Decision Tree: ', roc_auc_score(y_test, dtree_probs))

print('roc_auc_score for XGBoost: ', roc_curve(y_test, xgb_probs))
#ROC Curve

from sklearn.metrics import roc_curve

y_pred_prob1 = rf.predict_proba(X_test)[:,1]

fpr1 , tpr1, thresholds1 = roc_curve(y_test, rf_probs)



y_pred_prob2 = tree.predict_proba(X_test)[:,1]

fpr2 , tpr2, thresholds2 = roc_curve(y_test, dtree_probs)





y_pred_prob3 = xgb.predict_proba(X_test)[:,1]

fpr3 , tpr3, thresholds3 = roc_curve(y_test, xgb_probs)



plt.plot([0,1],[0,1], 'k--')

plt.plot(fpr1, tpr1, label= "Random Forest")

plt.plot(fpr2, tpr2, label= "Decision Tree")

plt.plot(fpr3, tpr3, label= "Xgboost")



plt.legend()

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.title('Receiver Operating Characteristic')

plt.show()