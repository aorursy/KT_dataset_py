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
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
train_df = pd.read_csv('../input/titanic/train.csv')
print("Shape of the training set : " ,train_df.shape)
train_df.head()
train_df.count()
test_df = pd.read_csv('../input/titanic/test.csv')
print("Shape of the test set : " ,test_df.shape)
test_df.head()
#sort the ages into logical categories
train_df["Age"] = train_df["Age"].fillna(-0.5)
test_df["Age"] = test_df["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)
test_df['AgeGroup'] = pd.cut(test_df["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train_df)
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (15,5))
sns.heatmap(train_df.isnull(), cmap = 'magma', ax = ax1)
sns.heatmap(test_df.isnull(), cmap = 'mako_r', ax = ax2)
fig, (ax1, ax2 ) = plt.subplots(ncols=2 , figsize=(20, 6))
for s in [1,2,3]:
    train_df.Age[train_df.Pclass == s].plot(kind='kde',ax = ax1,title = "Age wrt class" )
plt.legend(('1st','2nd','3rd'))
train_df.Embarked.value_counts(normalize=True).plot(kind='bar',alpha=0.5,ax = ax2, color = "chocolate")   
plt.title("Embarked")
sns.countplot(train_df['Embarked'],palette = "Accent")

#visualise gender
fig, (ax1, ax2, ax3 , ax4 ) = plt.subplots(ncols=4 , figsize=(20, 6))

train_df.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5 ,ax = ax1, color = "olive",title = "Survived")

train_df.Survived[train_df.Sex == 'male'].value_counts(normalize=True).plot(kind='bar',alpha=0.5 ,ax = ax2, color = "slateblue",title = "Male survivors")

train_df.Survived[train_df.Sex == 'female'].value_counts(normalize=True).plot(kind='bar',alpha=0.5,color='fuchsia' ,ax = ax3,title = "Female survivors")

train_df.Sex[train_df.Survived == 1].value_counts(normalize=True).plot(kind='bar',alpha=0.5,color='coral' ,ax = ax4,title = "Sex Survived")

plt.subplot2grid((2,3),(1,0),colspan=4)
for s in [1,2,3]:
    train_df.Survived[train_df.Pclass == s].plot(kind='kde')
plt.title("Survived wrt class")
plt.legend(('1st','2nd','3rd'))
plt.show()
fig, (ax1, ax2, ax3 , ax4 ) = plt.subplots(ncols=4 , figsize=(20, 6))

train_df.Survived[(train_df.Sex == 'male') & (train_df.Pclass ==1)].value_counts(normalize=True).plot(kind='bar',alpha=0.5,color='olive',title = "Rich men survivors",ax = ax1)

train_df.Survived[(train_df.Sex == 'male') & (train_df.Pclass ==3)].value_counts(normalize=True).plot(kind='bar',alpha=0.5,color='springgreen',title = "Poor men survivors",ax = ax2)

train_df.Survived[(train_df.Sex == 'female') & (train_df.Pclass ==1)].value_counts(normalize=True).plot(kind='bar',alpha=0.5,color='hotpink',title = "Rich women survivors",ax = ax3)

train_df.Survived[(train_df.Sex == 'female') & (train_df.Pclass ==3)].value_counts(normalize=True).plot(kind='bar',alpha=0.5,color='tomato',title = "Poor women survivors",ax = ax4)

df_m = train_df[train_df['Sex'] == 'male']
df_f = train_df[train_df['Sex'] == 'female']
df_m = df_m['Survived'].value_counts()
df_f = df_f['Survived'].value_counts()

trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Male', marker = dict(color = 'slateblue'))
trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Female', marker = dict(color = 'salmon'))
data = [trace1, trace2]
layout = go.Layout(height = 400, width = 500, title='Survival Distribution')
fig = go.Figure(data = data, layout= layout)
py.iplot(fig)
df_m = train_df[train_df['Sex'] == 'male']
df_f = train_df[train_df['Sex'] == 'female']
df_m = df_m['Embarked'].value_counts()
df_f = df_f['Embarked'].value_counts()

trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Male', marker = dict(color = 'lime'))
trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Female', marker = dict(color = 'orangered'))
data = [trace1, trace2]
layout = go.Layout(height = 400, width = 500, title='Embarked Distribution')
fig = go.Figure(data = data, layout= layout)
py.iplot(fig)
df_m = train_df[train_df['Survived'] == 0]
df_f = train_df[train_df['Survived'] == 1]
df_m = df_m['Parch'].value_counts()
df_f = df_f['Parch'].value_counts()

trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Not Survived', marker = dict(color = 'burlywood'))
trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Survived', marker = dict(color = 'darkolivegreen'))
data = [trace1, trace2]
layout = go.Layout(height = 400, width = 500, title='Parch Distribution')
fig = go.Figure(data = data, layout= layout)
py.iplot(fig)
df_m = train_df[train_df['Survived'] == 0]
df_f = train_df[train_df['Survived'] == 1]
df_m = df_m['Pclass'].value_counts()
df_f = df_f['Pclass'].value_counts()

trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Not Survived', marker = dict(color = 'firebrick'))
trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Survived', marker = dict(color = 'gold'))
data = [trace1, trace2]
layout = go.Layout(height = 400, width = 500, title='Pclass Distribution', )
fig = go.Figure(data = data, layout= layout)
py.iplot(fig)
import pandas as pd
train= pd.read_csv('../input/titanic/train.csv')
train["Hyp"] = 0
train.loc[train.Sex == "female","Hyp"]=1

train["Result"]=0
train.loc[train.Survived == train["Hyp"],"Result"]=1
print("The percentage of survivors :")
print( train["Result"].value_counts(normalize  = True))
df = [train_df, test_df]
for data in df:
    data['Title'] = data['Name'].str.extract(r', (\w+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex']).transpose()
for data in df:
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title']).mean()

labels = {'Mr':1, 'Mrs':2, 'Master':3, 'Miss':4, 'Rare':5}
test_df.replace({'Title':labels}, inplace = True)
train_df.replace({'Title':labels}, inplace = True)
train_df['Title'] = train_df['Title'].fillna(0)
train_df['Title'] = train_df['Title'].astype(int)                     # this is performed beacuse it was giving float values of title
train_df["Age"] = train_df["Age"].fillna(-0.5)
test_df["Age"] = test_df["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)
test_df['AgeGroup'] = pd.cut(test_df["Age"], bins, labels = labels)
mr_age = train_df[train_df["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train_df[train_df["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train_df[train_df["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train_df[train_df["Title"] == 4]["AgeGroup"].mode() #Baby
rare_age = train_df[train_df["Title"] == 5]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult"}

for x in range(len(train_df["AgeGroup"])):
    if train_df["AgeGroup"][x] == "Unknown":
        train_df["AgeGroup"][x] = age_title_mapping[train_df["Title"][x]]
        
for x in range(len(test_df["AgeGroup"])):
    if test_df["AgeGroup"][x] == "Unknown":
        test_df["AgeGroup"][x] = age_title_mapping[test_df["Title"][x]]


age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train_df['AgeGroup'] = train_df['AgeGroup'].map(age_mapping).astype(int)
test_df['AgeGroup'] = test_df['AgeGroup'].map(age_mapping).astype(int)
train_df['Embarked'].fillna('S', inplace = True)

label = {'S':1, 'C':2, 'Q':3}
train_df.replace({'Embarked':label}, inplace = True)
test_df.replace({'Embarked':label}, inplace = True)
train_df["Cabin"] = (train_df["Cabin"].notnull().astype('int'))
test_df["Cabin"] = (test_df["Cabin"].notnull().astype('int'))
#if we check the data info then fare feature is a category not int, so to convert we are performing this: 
train_df['Fare'] = pd.to_numeric(train_df['Fare'])
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)

train_df['Fare'] = pd.qcut(train_df['Fare'], 4, labels = [1, 2, 3, 4])
test_df['Fare'] = pd.qcut(test_df['Fare'], 4, labels = [1, 2, 3, 4])
label = {'male':1, 'female':0}
train_df.replace({'Sex':label}, inplace = True)
test_df.replace({'Sex':label}, inplace = True)
for data in df:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for data in df:
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'FamilySize'], axis = 1, inplace = True)
test_df.drop(['Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'FamilySize'], axis = 1, inplace = True)
train_df.head()
test_df.head()
pd.DataFrame({'Train':train_df.isnull().sum(), 'Test':test_df.isnull().sum()})
train_df.info()
train_df.describe()
train_df.info()
train_df.head()

X = train_df.drop('Survived', axis = 1)
y = train_df['Survived']

X.dropna(inplace = True)
X.isnull().sum()
from sklearn import linear_model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

#Create classifier

LR = linear_model.LogisticRegression()
LR_ = LR.fit(X_train,y_train )

y_pred = LR_.predict(X_test)

print("Train score: " , LR_.score(X_train,y_train))
print("Test score: " , LR_.score(X_test,y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn import model_selection
dc_tree= DecisionTreeClassifier(random_state = 42)
dc_ = dc_tree.fit(X_train,y_train)
print(dc_.score(X_train,y_train))

score = model_selection.cross_val_score(dc_tree ,X_train,y_train,scoring='accuracy' , cv=50)
print("After removing overfitting score is " + str(score.mean()))


dc_tree=DecisionTreeClassifier(random_state = 1 , max_depth =7 , min_samples_split=2)
dc_ = dc_tree.fit(X_train,y_train)
print(dc_.score(X_train,y_train))

score = model_selection.cross_val_score(dc_tree ,X_train,y_train,scoring='accuracy' , cv=50)
print("After removing overfitting score is " + str(score.mean()))


y_pred = dc_tree.predict(X_test)

print("Train score: " , dc_tree.score(X_train,y_train))
print("Test score: " , dc_tree.score(X_test,y_test))

from sklearn.metrics import classification_report
print("Classification Report :")
print(classification_report(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Train score: " , classifier.score(X_train,y_train))
print("Test score: " , classifier.score(X_test,y_test))

from sklearn.metrics import classification_report
print("Classification Report :")
print(classification_report(y_test, y_pred))


error_rate = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize = (8,6))
plt.plot(range(1,30), error_rate, color='salmon', linestyle='dashed', marker='o', markerfacecolor='b', markersize=10)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Train score: " , clf.score(X_train,y_train))
print("Test score: " , clf.score(X_test,y_test))

from sklearn.metrics import classification_report
print("Classification Report :")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
rfc = RandomForestClassifier(max_depth=10 , random_state=0)
rfc.fit(X_train , y_train)

y_pred = rfc.predict(X_test)

print("Train score: " , rfc.score(X_train,y_train))
print("Test score: " , rfc.score(X_test,y_test))

from sklearn.metrics import classification_report
print("Classification Report :")
print(classification_report(y_test, y_pred))
adb = AdaBoostClassifier(rfc, n_estimators = 200)
adb.fit(X_train, y_train)
y_pred = adb.predict(X_test)

print('Classification Report: \n', classification_report(y_pred, y_test))
adb_train_acc = round(adb.score(X_train, y_train) * 100, 2)
print('Training Accuracy: ', adb_train_acc)
adb_test_acc = round(adb.score(X_test, y_test) * 100, 2)
print('Testing Accuracy: ', adb_test_acc)
gdb = GradientBoostingClassifier()
params = {'learning_rate':[0.01,0.1,1,10],'n_estimators':[100,150,200,300],'subsample':[0.6,0.8,1.0],'max_depth':[2,3,4,6],'min_samples_leaf':[1,2,4,6]}
gcv = GridSearchCV(estimator=gdb, param_grid=params, cv=5, n_jobs=-1)
gcv.fit(X_train, y_train)
gcv.best_params_

gdb = GradientBoostingClassifier(max_depth = 2, n_estimators = 300, subsample = 0.6)
gdb.fit(X_train, y_train)
y_pred = gdb.predict(X_test)


print("Train score: " , gdb.score(X_train,y_train))
print("Test score: " , gdb.score(X_test,y_test))

from sklearn.metrics import classification_report
print("Classification Report :")
print(classification_report(y_test, y_pred))
# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('Random Forest', rfc))
models.append(('ADA booster', adb))
models.append(('GradientBoostingClassifier', gdb))



# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize =(20,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
X_train.head()
test_df.head()
test_df['Fare'] = pd.to_numeric(test_df['Fare'])
test_df['Survived'] = gdb.predict(test_df.drop(['PassengerId',"Survived"], axis = 1))
test_df[['PassengerId', 'Survived']].to_csv('MySubmission.csv', index = False)
