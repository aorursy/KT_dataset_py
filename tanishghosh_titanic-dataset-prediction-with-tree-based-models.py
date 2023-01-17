# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,VotingClassifier

from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV

from sklearn.metrics import precision_score,recall_score,roc_curve,auc,confusion_matrix
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.head()
train_data.shape
print(train_data.isna().sum())
train_data.drop(columns=['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

passenger_id = test_data.PassengerId

test_data.drop(columns=['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
print(train_data.head())
train_data.info()
age_median = train_data['Age'].median()

train_data['Age'].fillna(age_median,inplace=True)



age_test_median = test_data['Age'].median()

fare_test_median = test_data['Fare'].median()

test_data['Age'].fillna(age_test_median,inplace=True)

test_data['Fare'].fillna(fare_test_median,inplace=True)





plt.hist(train_data['Age'],bins=10)

plt.show()
plt.figure(figsize=(11,10))

sns.heatmap(train_data.corr(),annot=True)

plt.show()
sns.pairplot(train_data)
si = SimpleImputer(missing_values=np.nan,strategy='most_frequent')

train_data.Embarked = si.fit_transform(np.asarray(train_data['Embarked']).reshape(-1,1))

test_data.Embarked = si.fit_transform(np.asarray(test_data['Embarked']).reshape(-1,1))
le = LabelEncoder()

train_data['Sex'] = le.fit_transform(train_data['Sex'])

train_data['Embarked'] = le.fit_transform(train_data['Embarked'])

test_data['Sex'] = le.fit_transform(test_data['Sex'])

test_data['Embarked'] = le.fit_transform(test_data['Embarked'])

train_data.head()
X = train_data.iloc[:,1:]

X_test = test_data.iloc[:,1:]

X.head()
y = train_data.iloc[:,0]

y_test = test_data.iloc[:,0]

y.head()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
parameters={'max_depth':[2,3,5], 'min_samples_leaf':[10], 'min_samples_split':[15], 'n_estimators':[50,100,200,500]}
folds = 15

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 43)



rfc = RandomForestClassifier(random_state=42)

rfc.fit(X_train,y_train)
clf = GridSearchCV(rfc, parameters,cv = skf.split(X_valid,y_valid),return_train_score=True)

clf.fit(X_valid,y_valid)

y_pred = clf.predict(X_valid)
print("Accuracy:",clf.score(X_valid,y_valid)*100)

print("Precision:",precision_score(y_valid, y_pred)*100)

print("Recall:",recall_score(y_valid,y_pred)*100)
probs = clf.predict_proba(X_valid)

left_prob = probs[:,1]

fpr, tpr, thresholds = roc_curve(y_valid, left_prob)

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.show()
conf_mat = confusion_matrix(y_valid,y_pred)

conf_mat
ada = AdaBoostClassifier(n_estimators=65, random_state=0)

ada.fit(X_train,y_train)

y_pred = ada.predict(X_valid)
print("Accuracy:",ada.score(X_valid,y_valid)*100)

print("Precision:",precision_score(y_valid, y_pred)*100)

print("Recall:",recall_score(y_valid,y_pred)*100)
probs = clf.predict_proba(X_valid)

left_prob = probs[:,1]

fpr, tpr, thresholds = roc_curve(y_valid, left_prob)

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.show()
conf_mat = confusion_matrix(y_valid,y_pred)

conf_mat
eclf1 = VotingClassifier(estimators=[('For', rfc), ('Grad', gbc), ('Ada', ada)], voting='hard')

eclf1.fit(X_train, y_train)
eclf1.score(X_valid,y_valid)
y_pred = eclf1.predict(X_valid)
print("Accuracy:",ada.score(X_valid,y_valid)*100)

print("Precision:",precision_score(y_valid, y_pred)*100)

print("Recall:",recall_score(y_valid,y_pred)*100)
probs = clf.predict_proba(X_valid)

left_prob = probs[:,1]

fpr, tpr, thresholds = roc_curve(y_valid, left_prob)

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.show()
conf_mat = confusion_matrix(y_valid,y_pred)

conf_mat
parameters={'learning_rate':[x for x in np.linspace(0.1,0.5,5)],

            'max_depth':[5,7,8], 'min_samples_leaf':[10], 'min_samples_split':[57],

            'n_estimators':[x for x in range(40,80,10)],'max_features':['sqrt'],

            'subsample':[0.8]

            }
gbc = GradientBoostingClassifier(random_state=42)

gbc.fit(X_train,y_train)

y_pred = gbc.predict(X_valid)
clf = GridSearchCV(gbc, parameters,return_train_score=True)

clf.fit(X_train,y_train)

y_pred = gbc.predict(X_valid)
print("Accuracy:",clf.score(X_valid,y_valid)*100)

print("Precision:",precision_score(y_valid, y_pred)*100)

print("Recall:",recall_score(y_valid,y_pred)*100)
probs = clf.predict_proba(X_valid)

left_prob = probs[:,1]

fpr, tpr, thresholds = roc_curve(y_valid, left_prob)

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.show()
conf_mat = confusion_matrix(y_valid,y_pred)

conf_mat
clf.fit(X_train,y_train)

y_pred = gbc.predict(test_data)
output_df = pd.DataFrame({"ID":passenger_id,"Prediction":y_pred})

output_df.head()
output_df.shape
output_df.to_csv('my_submission.csv')