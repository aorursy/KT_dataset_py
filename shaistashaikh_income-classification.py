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
df=pd.read_csv('/kaggle/input/income-classification/income_evaluation.csv')
df
df.columns = df.columns.str.lstrip()
df

gender = {" Male": 1, " Female": 2 }

df['sex'] = df['sex'].map(gender)

df
df['sex']=df['sex'].map( {'Male':1, 'Female':0} )
df
df.describe()
df.info()
df.head()
df.shape
df.columns
#df.columns.str.replace(' ', '')
df.columns
df.isnull().sum()
df.columns
numeric_features = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','income']

# Identify Categorical features
cat_features = ['workclass','education','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native']
cat_features = ['workclass','education','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cat_features
numeric_features = ['age','fnlwgt','education-num','capital-gain','capital-loss', 'sex','hours-per-week','income']

numeric_features
sns.countplot(df['income'],label="Count")

g = sns.heatmap(df[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

g = sns.factorplot(x="capital-gain",y="income",data=df,kind="bar",size = 6,palette = "muted")
g.despine(left=True)
g = g.set_ylabels(">50K probability")
g = sns.factorplot(x="education-num",y="income",data=df,kind="bar",size = 6,palette = "muted")
g.despine(left=True)
g = g.set_ylabels(">50K probability")
g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "age")

# Fill Missing Category Entries
df["workclass"] = df["workclass"].fillna("X")
df["occupation"] = df["occupation"].fillna("X")
df["native-country"] = df["native-country"].fillna("United-States")

# Confirm All Missing Data is Handled
df.isnull().sum()
g = sns.factorplot(x="age",y="income",data=df)
#g = g.set_ylabel("Income >50K Probability")


target=list(df.columns)[14]
print(target)
#df.drop(df.columns[[0,2,4,10,11,12,14]],axis=1)
#To Drop column use the index no...
x=df.drop(['income'],axis=1)
y=df[['income']]
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
X_train=X_train.drop(X_train.columns[[1,3,5,6,7,8,13]],axis=1)
X_train
X_train
X_train.info()
y_train.info()
from sklearn.tree import DecisionTreeClassifier  
classifier1 = DecisionTreeClassifier(criterion='gini')  
classifier1.fit(X_train, y_train) 

X_test=X_test.drop(X_test.columns[[1,3,5,6,7,8,13]],axis=1)
X_test
X_test.info()
y_predtrain = classifier1.predict(X_train) 
y_predtrain
y_predtrainlist=list(y_predtrain)
TrainCountOfgt50K=y_predtrainlist.count(' >50K')
TrainCountOflt50K=y_predtrainlist.count(' <=50K')
print(TrainCountOfgt50K)
print(TrainCountOflt50K)
y_pred = classifier1.predict(X_test)  
print(y_pred)
y_predlist=list(y_pred)
TestCountOfgt50K=y_predlist.count(' >50K')
TestCountOflt50K=y_predlist.count(' <=50K')
print(TestCountOfgt50K)
print(TestCountOflt50K)
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package
acc = accuracy_score(y_test,y_pred)
print("Accuracy for this model {} %".format(acc*100))
print(classifier1.feature_importances_)
from sklearn.neighbors import KNeighborsClassifier


classifier3 = KNeighborsClassifier(n_neighbors= 7)  
classifier3.fit(X_train, y_train) 
y_pred_3 = classifier3.predict(X_test)  
print(y_pred_3)
acc_3 = accuracy_score(y_test,y_pred_3)
print("Accuracy  model {} %".format(acc_3*100))
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred_3))
print(classification_report(y_test, y_pred_3)) 