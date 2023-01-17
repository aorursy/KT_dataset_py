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

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

from pandas_profiling import ProfileReport

%matplotlib inline



# Import label encoder:

import sklearn

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder



#Import libraries for model selection and building:

import sklearn.tree as tree

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score,roc_curve

from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import export_graphviz



import warnings

warnings.filterwarnings("ignore")



#print(os.getcwd())
train = pd.read_csv("/kaggle/input/titanic/train.csv").set_index('PassengerId')

test = pd.read_csv("/kaggle/input/titanic/test.csv").set_index('PassengerId')

survive = pd.read_csv("/kaggle/input/titanic/gender_submission.csv").set_index('PassengerId')
print("The dimension of Train Dataset is:", train.shape)

train.head()
print("The dimension of Test Dataset is:", test.shape)

test.head()
# embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# 1 -> Survived, 0-> Did not survive



print("Train Dataset - Survival Number as per Sex, PClass, Embark")

survival_rate = pd.DataFrame(train.groupby(["Sex", "Pclass", "Embarked"])["Survived"].count()).style.background_gradient(cmap="bone_r")

survival_rate
ProfileReport(train)
ProfileReport(test)
# Drop the columns 'cabin' from training and test data as it contains more than 30% missing values



col_drop = ['Cabin']



train.drop(col_drop, axis = 1, inplace = True)

test.drop(col_drop, axis = 1, inplace = True)
# Filling the missing Age variable with mean value 



train['Age'] =train['Age'].fillna(train['Age'].mean())

test['Age'] =test['Age'].fillna(test['Age'].mean())
# Filling the missing Emberked variable with 'S' as the largerst value is 'S' 



train['Embarked'] =train['Embarked'].fillna('S')

test['Embarked'] =test['Embarked'].fillna('S')
# Filling the missing Age variable with mean value 

test['Fare'] =test['Fare'].fillna(test['Fare'].mean())
# Seperating the title from the name

train['Title'] = train.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

test['Title'] = test.Name.str.split(',').str[1].str.split('.').str[0].str.strip()



print("Train[Title] List of unquie values", "(", train['Title'].nunique(), ") :" , train['Title'].unique())

print(" ")

print("Test[Title] List of unquie values", "(", test['Title'].nunique(), ") :" ,test['Title'].unique())
# Grouping the Title into 5 Category (Mr, Mrs, Miss, Master, Dr)



title_replace_Mr = ['Don','Rev','Sir','Col','Capt','Jonkheer','Major']

title_replace_Mrs = ['Mme','Lady','Mlle','the Countess']

train['Title'] = train['Title'].replace('Ms','Miss')

train['Title'] = train['Title'].replace(title_replace_Mr,'Mr')

train['Title'] = train['Title'].replace(title_replace_Mrs,'Mrs')



test_title = ['Col','Rev']

test['Title'] = test['Title'].replace('Ms','Miss')

test['Title'] = test['Title'].replace('Dona','Mrs')

test['Title'] = test['Title'].replace(test_title,'Mr')
# Dropping unnecessary variable

cols_to_drop = ['Name','Ticket']   

train = train.drop(cols_to_drop, axis=1)   

test = test.drop(cols_to_drop, axis=1)
# Title is as per age

# 1 -> Survived, 0-> Did not survive



print("Train Dataset - Survival Numbers as per Pclass & Title")

survival_rate1 = pd.DataFrame(train.groupby(["Pclass", "Title"])["Survived"].count()).style.background_gradient(cmap="bone_r")

survival_rate1
# label_encoder object knows how to understand word labels. 

le = LabelEncoder()

  

# Encode labels in column 'species'. 

train['Sex']= le.fit_transform(train.Sex)

train['Embarked'] = le.fit_transform(train.Embarked)

train['Title'] = le.fit_transform(train.Title)



test['Sex']= le.fit_transform(test.Sex)

test['Embarked'] = le.fit_transform(test.Embarked)

test['Title'] = le.fit_transform(test.Title)
X_train = train.drop('Survived', axis=1)

Y_train = train['Survived']

X_test = test.copy()

Y_test = survive["Survived"]



print("Dimension of X_Train:", X_train.shape)

print("Dimension of X_Test:", Y_train.shape)

print("Dimension of Y_Train:", X_test.shape)

print("Dimension of Y_Test:", Y_test.shape)
DT = tree.DecisionTreeClassifier(max_depth=3, random_state=200)

DT.fit(X_train, Y_train)

DT_Y_Pred = DT.predict(X_test)
print("Decsison Tree Model Results:\n")

print("Accuracy Score:", round(accuracy_score(Y_test, DT_Y_Pred),2)*100,"%")

print("***************************************************\n")

print("Classification Report:\n", classification_report(Y_test, DT_Y_Pred))

print("***************************************************\n")

print("Confusion Matrix:\n", confusion_matrix(Y_test, DT_Y_Pred))

print("***************************************************")
CM = pd.DataFrame(confusion_matrix(Y_test, DT_Y_Pred))



plt.figure(figsize=(10,5))

sns.heatmap(CM, annot=True,  fmt=".0f", annot_kws={"size": 20}, cmap="cividis_r", linewidths=0.9)

plt.title('Confusion matrix for Decision Tree Model', y=1.1, fontdict = {'fontsize': 20})

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
## ROC curve for RF:

fpr, tpr, _ = metrics.roc_curve(Y_test, DT_Y_Pred)

auc = metrics.roc_auc_score(Y_test, DT_Y_Pred)



plt.figure(figsize=(10,5))

plt.style.use('seaborn')

plt.plot(fpr,tpr,label="AUC ="+str(auc))

plt.plot([0,1],[0,1],"r--")

plt.title("ROC for Decision Tree model", fontdict = {'fontsize': 20})

plt.xlabel("True positive_rate")

plt.ylabel("False positive_rate")

plt.legend(loc= 4, fontsize = "x-large")
Final_Submission = pd.DataFrame()

Final_Submission['Survived'] = DT_Y_Pred

Final_Submission.to_csv('F:\BA - Jigsaw\Kaggle\Titanic Machine Learning from Disaster\Final_Submission.csv')