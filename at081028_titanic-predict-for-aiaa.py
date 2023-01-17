# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os #os模塊提供了基本的文件和目錄操作，並且是跨平台的

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
train=pd.read_csv("../input/titanicdataset/Titanic_train.csv" )
tit1=train.select_dtypes(include=['float64','int64','object'])
train.info()

test=pd.read_csv("../input/titanicdataset/Titanic_test.csv")
tit2=test.select_dtypes(include=['float64','int64','object'])
test.info()
print("train shape:",train.shape)
tit1.head()
print("test shape :",test.shape)
tit2.head()
tit2['survived']=np.nan
print("test shape :",tit2.shape)
tit2.head()
plt.figure(figsize=(4,4))
plt.title('survived',size=20)
tit1.survived.value_counts().plot.bar(color=['red','green'])

plt.figure(figsize=(4,4))
plt.title('sex',size=20)
tit1.sex.value_counts().plot.bar(color=['skyblue','pink'])
percent=round(np.mean(train['survived']),3)*100
print("Percentage of Survivors:",percent,"%")
total=train['survived'].sum()
total
men=train[train['sex']=='male']
women=train[train['sex']=='female']
m=men['sex'].count()
w=women['sex'].count()
print("male:",m)
print("female:",w)
print("percentage of women:",round(w/(m+w)*100),"%")
print("percentage of men:",round(m/(m+w)*100),"%")
plt.figure(figsize=(5,5))
plt.title("CLASS DIVISION",size=20)
tit1.pclass.value_counts().plot.bar(color=['olive','coral','gold'])
train['fare'].hist(bins = 80, color = 'orange')
plt.title("FARE",size=20)
plt.figure(figsize=(5,5))
plt.title("embarked",size=20)
tit1.embarked.value_counts().plot.bar(color=['olive','coral','gold'])
plt.figure(figsize=(5,5))
sns.countplot(x = 'survived', hue = 'sex', data = train)
plt.title("SURVIVED AND SEX",size=20)
plt.figure(figsize=(5,5))
sns.countplot(x = 'survived', hue = 'pclass', data = train)
plt.title("SURVIVED AND PCLASS",size=20)
plt.figure(figsize=(5,5))
sns.countplot(x = 'survived', hue = 'embarked', data = train)
plt.title("SURVIVED AND EMBARKED",size=20)
#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train)
train.isnull().sum()
train = train.drop(['ticket', 'cabin'], axis=1)
test = test.drop(['ticket', 'cabin'], axis=1)
train['age'].hist(bins=40,color='salmon')
plt.title("AGE",size=20)
age_group = train.groupby("pclass")["age"]
print(age_group.median())
train.loc[train.age.isnull(),'age']=train.groupby("pclass").age.transform('median')
test.loc[test.age.isnull(),'age']=test.groupby("pclass").age.transform('median')
print(train['age'].isnull().sum())
from statistics import mode
train["embarked"] = train["embarked"].fillna(mode(train["embarked"]))
test["embarked"] = test["embarked"].fillna(mode(test["embarked"]))
train["fare"] = train["fare"].fillna(mode(train["fare"]))
print(train.isnull().sum())
print(test.isnull().sum())
train["sex"][train["sex"] == "male"] = 0
train["sex"][train["sex"] == "female"] = 1

test["sex"][test["sex"] == "male"] = 0
test["sex"][test["sex"] == "female"] = 1

train["embarked"][train["embarked"] == "S"] = 0
train["embarked"][train["embarked"] == "C"] = 1
train["embarked"][train["embarked"] == "Q"] = 2

test["embarked"][test["embarked"] == "S"] = 0
test["embarked"][test["embarked"] == "C"] = 1
test["embarked"][test["embarked"] == "Q"] = 2

train['fam']=train['sibsp']+train['parch']+1
test['fam']=test['sibsp']+test['parch']+1
train['isAlone'] = 0
train.loc[train['fam'] == 1, 'isAlone'] = 1
test['isAlone'] = 0
test.loc[test['fam'] == 1, 'isAlone'] = 1
train['age']=train['age'].astype(str)
test['age']=test['age'].astype(str)
import re
train['age'] = train['age'].map(lambda x: re.compile("[0-9]").search(x).group())
train['age'].unique().tolist()
#cat={'2':1, '3':2 , '5':3, '1':4 ,'4':5,'8':6,'6':7,'7':8,'0':9,'9':10}
cat={'0':1,'1':2 ,'2':3, '3':3 , '4':5,'5':6, '6':7,'7':8,'8':9,'9':10}
train['age']=train['age'].map(cat)
train['age'].unique().tolist()
test['age'] = test['age'].map(lambda x: re.compile("[0-9]").search(x).group())
test['age'].unique().tolist()
test['age']=test['age'].map(cat)
test['age'].unique().tolist()
train['title'] = train.name.str.extract(' ([A-Za-z]+)\.', expand=False)
train['title'] = train['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['title'] = train['title'].replace('Mlle', 'Miss')
train['title'] = train['title'].replace('Ms', 'Miss')
train['title'] = train['title'].replace('Mme', 'Mrs')
    
train[['title', 'survived']].groupby(['title'], as_index=False).mean()
test['title'] = test.name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['title'] = test['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test['title'] = test['title'].replace('Mlle', 'Miss')
test['title'] = test['title'].replace('Ms', 'Miss')
test['title'] = test['title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train['title'] = train['title'].map(title_mapping)
train['title'] = train['title'].fillna(0)
test['title'] = test['title'].map(title_mapping)
test['title'] = test['title'].fillna(0)
train.head()
train = train.drop(['name'], axis = 1)
train = train.drop(['parch'], axis = 1)
train = train.drop(['fare','sibsp'], axis = 1)
#train = train.drop(['fam'], axis = 1)
test = test.drop(['name'], axis = 1)
test = test.drop(['parch'], axis = 1)
test = test.drop(['fare','sibsp'], axis = 1)
#test = test.drop(['fam'], axis = 1)
train.head()

test.head()
#train = pd.get_dummies(train)
#pd.DataFrame(train)

#test = pd.get_dummies(test)
#pd.DataFrame(test)
print(train)
print(test)
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train.drop(['survived','ID'], axis=1), 
                                                    train['survived'], test_size = 0.2, 
                                                    random_state = 42)
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)
predictions = logisticRegression.predict(X_test)
acc_LOG = round(accuracy_score(predictions, y_test) * 100, 2)
print(acc_LOG)
print(predictions)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
accuracy=((tp+tn)/(tp+tn+fp+fn))
print('accuracy is: ', (round(accuracy, 2)*100),'%')
ids = test['ID']
#predictions = randomforest.predict(test.drop('ID', axis=1))
predictions = logisticRegression.predict(test.drop('ID', axis=1))

output = pd.DataFrame({ 'ID' : ids, 'survived': predictions })
output.to_csv('submission.csv', index=False)
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)
ids = test['ID']
#predictions = randomforest.predict(test.drop('ID', axis=1))
predictions = clf.predict(test.drop('ID', axis=1))

output = pd.DataFrame({ 'ID' : ids, 'survived': predictions })
output.to_csv('submission_clf.csv', index=False)
from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print(acc_svc)

ids = test['ID']
#predictions = randomforest.predict(test.drop('ID', axis=1))
predictions = svc.predict(test.drop('ID', axis=1))

output = pd.DataFrame({ 'ID' : ids, 'survived': predictions })
output.to_csv('submission_svc.csv', index=False)
ids = test['ID']
#predictions = randomforest.predict(test.drop('ID', axis=1))
predictions = svc.predict(test.drop('ID', axis=1))

output = pd.DataFrame({ 'ID' : ids, 'survived': predictions })
output.to_csv('submission_svc.csv', index=False)