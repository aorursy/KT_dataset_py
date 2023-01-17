# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import RobustScaler



from sklearn.metrics import  confusion_matrix, plot_roc_curve, classification_report, roc_auc_score, plot_precision_recall_curve, accuracy_score



from sklearn.model_selection import cross_validate



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.isnull().sum()
sns.countplot(x='Outcome',data=df)

plt.show()
df.iloc[:,:8].hist(bins=15, figsize=(15, 6), layout=(2, 4));


sns.pairplot(df, hue="Outcome")

plt.show()
sns.heatmap(df.corr(),annot=True,xticklabels=True, yticklabels=True)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
sns.set(style="whitegrid")



sns.set(rc={'figure.figsize':(4,2)})

sns.boxplot(x=df['Insulin'])

plt.show()

sns.boxplot(x=df['BloodPressure'])

plt.show()

sns.boxplot(x=df['DiabetesPedigreeFunction'])

plt.show()
df = df[(df.BloodPressure != 0) & (df.BMI != 0) & (df.Glucose != 0) & (df.Insulin != 0)]

df.head()
X = df.drop(columns=['Outcome'])

y = df['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

result = []
lr = LogisticRegression(max_iter = 2000)

lr.fit(X_train,y_train)

y_pred = lr.predict_proba(X_test)[:,1]

scores=cross_val_score(lr,X_train,y_train,scoring='roc_auc',cv=10)

result.append(scores.mean())
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

scores=cross_val_score(dt,X_train,y_train,scoring='roc_auc',cv=10)

result.append(scores.mean())
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

scores=cross_val_score(rf,X_train,y_train,scoring='roc_auc',cv=10)

result.append(scores.mean())

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

scores=cross_val_score(knn,X_train,y_train,scoring='roc_auc',cv=10)

result.append(scores.mean())

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

scores=cross_val_score(gnb,X_train,y_train,scoring='roc_auc',cv=10)

result.append(scores.mean())

print(result)
ax=plt.figure(figsize=(9,4))

plt.plot(['Logistic Regression','Decision Tree','Random Forest','KNN','Naive Bayes'],result,label='ROC-AUC')

plt.ylabel('ROC Score')

plt.xlabel('Algortihms')

plt.show()


