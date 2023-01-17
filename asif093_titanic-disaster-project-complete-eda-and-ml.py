# linear algebra

import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 

import warnings





#Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns 





#Setting Instances

sns.set()

%matplotlib inline



import os



warnings.filterwarnings('ignore')
#from google.colab import drive

#drive.mount('/content/gdrive')
train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

sub =pd.read_csv("../input/gender_submission.csv")
train.head()
# Data types of all columns

train.dtypes
# Describing Numerical values in train data

train.describe()
# Checking Null Values in data

train.isnull().sum()
test.isnull().sum()
# Defining Function for Adding Title

def Adding_title(data):

  

  

  import re 

  a='Braund, Mr.Owen Harris'

  re.search(' ([A-Z][a-z]+)\.', a).group(1)

  

  data['Title'] = data['Name'].apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

  

  data['Title'] = data['Title'].replace('Mme', 'Mrs')

  data.loc[(~data['Title'].isin(['Mr', 'Mrs', 'Miss', 'Master'])), 'Title'] = 'Rare Title'
Adding_title(train)

Adding_title(test)
# Validating Title

print(train.Title.unique(), test.Title.unique())
#Dropping Names column for both the dataset

train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
# Defining function for the addition of family  size column

def fam_size(data):

  data['Fsize'] = data['SibSp'] + data['Parch']+1
fam_size(train)

fam_size(test)
# Validating the change 

print(train.columns, test.columns)
# filling Age missing values in train

data = [train, test]



for dataset in data:

    mean = train["Age"].mean()

    std = test["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train["Age"].astype(int)
#Fiiling missing age values in test data

for t in test['Title'].unique():

  for p in train['Pclass'].unique():

      test.loc[(test['Title'] == t) & (test['Pclass']==p) &  (test['Age'].isnull()), 'Age']== test.loc[(test['Title'] == t) & (test['Pclass']==p)  , 'Age'].median()
# Validating the change in data

train.isnull().sum()
# Unique values in Embarked column

train['Embarked'].value_counts()
#Filling Embarked value in Train data

train = train.fillna({"Embarked": "S"})



#Filling Fare in test data with o

test = test.fillna({"Fare":0})
test.isnull().sum()
train.drop('Cabin', axis=1, inplace = True)

test.drop('Cabin', axis=1, inplace = True)
#Validating The Change in the data now.....



print(train.shape, test.shape)
train.head()
test.head()
sns.countplot(train.Survived)

plt.title('Distribution of Survived')

plt.show()
sns.countplot(x='Sex', hue='Survived',data=train)

plt.title('Sex distribution with respect to survived')

plt.show()
sns.countplot(x='Pclass', data=train)

plt.title(' Pclass Distribution')

plt.show()
sns.countplot(x='Pclass', hue='Survived', data=train)

plt.title('Survival rate in Pclass')

plt.show()


sns.factorplot(x='Sex', col='Pclass',hue='Survived', data=train, kind='count');

plt.title('Sex survival rate against Pclass')

plt.show()
plt.figure(figsize=(12,4))

sns.countplot(x='Title',hue='Survived', data=train)

plt.title('Survival rate against titlte')

plt.show()
sns.countplot(x='Fsize', hue='Survived', data=train);

plt.title('Survival rate against family size')

plt.legend()

plt.show()
sns.pairplot(train);
plt.figure(figsize=(10,5))

sns.distplot(train['Age']);

plt.title('Age Distribution in data')

plt.show()
plt.figure(figsize=(18,6))

sns.countplot(x='Age', hue='Survived', data=train)

plt.xticks(rotation=90)

plt.show()
sns.countplot(x='Embarked', hue='Survived', data=train)

plt.title('Embarked Distribution against survival')

plt.show()
sns.jointplot(x="Age", y="Fare", data=train);
train.drop('PassengerId', axis=1, inplace=True)

plt.figure(figsize=(12,6))

cor = train.corr()

sns.heatmap(cor, annot=True);

plt.title('Correlation in Data')

plt.show()
from sklearn.preprocessing import LabelEncoder
train= train.apply(LabelEncoder().fit_transform)

test = test.apply(LabelEncoder().fit_transform)
train.head()
#!pip install catboost
import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, precision_score, accuracy_score

from sklearn.metrics import recall_score, classification_report, f1_score, roc_curve, auc

import xgboost as xgb

from catboost import CatBoostClassifier, Pool, cv
def plot_roc_curve(fpr, tpr, label=None):

    plt.figure(figsize=(8,6))

    plt.title('ROC Curve')

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.005, 1, 0, 1.005])

    plt.xticks(np.arange(0,1, 0.05), rotation=90)

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc='best')
X = train.drop('Survived', axis=1)

Y = train['Survived']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=5, test_size = 0.2)
print(xtrain.shape, ytrain.shape)

print(xtest.shape, ytest.shape)
xgb_clf = xgb.XGBClassifier(learning_rate =0.001,n_estimators=11,max_depth=10,min_child_weight=1,subsample=1.0,colsample_bytree=0.55,

                                 reg_alpha=0, nthread=4,seed=5,random_state=5)
xgb_clf.fit(xtrain, ytrain)
xgb_pred = xgb_clf.predict(xtest)

xgb_pproba = xgb_clf.predict_proba(xtest)[:,1]
accuracy_xgb = accuracy_score(ytest,xgb_pred)

print("Accuracy: {}".format(accuracy_xgb))
print(classification_report(ytest,xgb_pred))
xgb_auc = roc_auc_score(ytest, xgb_pproba)

fpr,tpr,threshold=roc_curve(ytest,xgb_pproba)

plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% xgb_auc)
features_tuple=list(zip(X.columns,xgb_clf.feature_importances_))

feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])

feature_imp=feature_imp.sort_values("Importance",ascending=False)
plt.figure(figsize=(12,6))

sns.barplot(x="Feature Names",y="Importance", data=feature_imp, color='b')

plt.xlabel("Titanic Features")

plt.ylabel("Importance")

plt.xticks(rotation=90)

plt.title("Random Forest Classifier - Features Importance")
wanted_tc = xtrain.columns

wanted_tc
fpredictions = xgb_clf.predict(test[wanted_tc])
sub.head()
submission = pd.DataFrame()

submission['PassengerId'] = sub['PassengerId']

submission['Survived'] = predictions 

submission.head()
submission.to_csv('/content/gdrive/My Drive/Python Work/DataSets/titanic/Submission.csv', index=False)