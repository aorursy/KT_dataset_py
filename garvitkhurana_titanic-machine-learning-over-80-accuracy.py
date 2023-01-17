import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.info()
train.describe()
train.hist(figsize=(20,12))
def  bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    
bar_chart("Sex")
bar_chart("Pclass")
bar_chart("SibSp")
bar_chart('Embarked')
sns.barplot(y="Survived",x="Sex",data=train)
sns.barplot(y="Survived",x="Embarked",data=train)
t=sns.barplot(y="Survived",x="Parch",data=train)
t=sns.barplot(y="Survived",x="SibSp",data=train)
train = pd.read_csv("../input/train.csv")
y=train.Survived.values
test = pd.read_csv("../input/test.csv")
pas_id=test.PassengerId.values
train_test_data = pd.concat([train, test],axis=0)
train_test_data.head()
train_test_data.shape
title=[]
for i in train_test_data['Name']:
    t=i.split(",")[1].split(".")[0].strip()
    title.append(t)
train_test_data["Title"]=title
train_test_data.Title.value_counts()
train_test_data['FamilySize'] = train_test_data['Parch'] + train_test_data['SibSp'] + 1
train_test_data.FamilySize.value_counts()
train_test_data['Singleton'] = train_test_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train_test_data['SmallFamily'] = train_test_data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
train_test_data['LargeFamily'] = train_test_data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
CleanTicket=[]
for i in train_test_data.Ticket:
    i = i.replace('.', '')
    i = i.replace('/', '')
    i = i.split()[0]
    if i.isalpha():
        CleanTicket.append(i)
    else:
        CleanTicket.append("Number")
train_test_data["CleanTicket"]=CleanTicket
plt.figure(figsize=(20,12))
sns.barplot("FamilySize","Survived",data=train_test_data)
plt.figure(figsize=(20,12))
sns.barplot("Title","Survived",data=train_test_data)
plt.figure(figsize=(20,12))
sns.barplot("CleanTicket","Survived",data=train_test_data,ci=False)
train_test_data.drop(["Survived"],axis=1,inplace=True)
train_test_data.drop(["Name","FamilySize","Ticket"],axis=1,inplace=True)
def cable_name(x):
    try:
        return x[0]
    except TypeError:
        return "None"
train_test_data["Cabin"]=train_test_data.Cabin.apply(cable_name)
train_test_data.describe()
train_test_data['Age'].fillna(np.mean(train_test_data.Age),inplace=True)
train_test_data['Fare'].fillna(np.mean(train_test_data.Fare),inplace=True)
train_test_data['Fare'] = StandardScaler().fit_transform(train_test_data['Fare'].values.reshape(-1, 1))
train_test_data.describe()
num_data=train_test_data.select_dtypes(exclude=object).columns
train_test_data[num_data].head()
cat_data=train_test_data.select_dtypes(include=object).columns
print(train_test_data[cat_data].info())
train_test_data['Embarked'].fillna(train_test_data['Embarked'].mode()[0], inplace = True)
for i in cat_data:
    train_test_data[i].fillna("Missing",inplace=True)
    dummies=pd.get_dummies(train_test_data[i],prefix=i)
    train_test_data=pd.concat([train_test_data,dummies],axis=1)
    train_test_data.drop(i,axis=1,inplace=True)
train_test_data.head()
test_data=train_test_data.iloc[891:]
train_data=train_test_data.iloc[:891]
y=y
train=train_data
targets=y
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(25, 25))
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train_data)
print(train_reduced.shape)
test_reduced = model.transform(test_data)
print(test_reduced.shape)
clf=RandomForestClassifier(min_samples_split=2,max_depth=6,bootstrap=True,min_samples_leaf=1,n_estimators=100,max_features='auto')
clf.fit(train_reduced,y)
pred=clf.predict(test_reduced)
from sklearn.model_selection import cross_val_score,train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_data,y,test_size=0.2)
clf=RandomForestClassifier(min_samples_split=2,max_depth=6,bootstrap=True,min_samples_leaf=1,n_estimators=100,max_features='auto')
clf.fit(X_train,y_train)
pred=clf.predict(X_test).reshape(-1,1)
cv_result = cross_val_score(clf,pred,y_test,cv=6) 
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/6)
# test_data["Survived"]=pred
# test_data[["PassengerId","Survived"]].to_csv("data/predictions/feature_engg_3.csv",index=False)
# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50,10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train_data, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train_data, y)