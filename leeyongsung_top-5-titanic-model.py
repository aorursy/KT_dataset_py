from IPython.core.display import display, HTML
display(HTML("<style>.container {width:90% !important;}</style>"))



import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('dark')
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_path='/kaggle/input/titanic/train.csv'
test_path='/kaggle/input/titanic/test.csv'
titanic_train=pd.read_csv(train_path)
titanic_test=pd.read_csv(test_path)


titanic_train.head()
titanic_test.head()
titanic_train.describe()
titanic_train.describe(include ='O')
titanic_train.info()
sns.pairplot(data = titanic_train)
titanic_train['Survived'].value_counts().plot(kind = 'pie',autopct='%1.1f%%')
age_means = titanic_train['Age'].mean()
titanic_train['Age'].fillna(age_means, inplace = True)

count, bin_dividers = np.histogram(titanic_train['Age'], bins = 8)
bin_names = ['10세▼','10대','20대','30대','40대','50대','60대','70대▲']

titanic_train['Age_cut'] = pd.cut(titanic_train['Age'], bins = bin_dividers, labels = bin_names, include_lowest=True)
titanic_train['Age_cut'].value_counts()
fare_means = titanic_train['Fare'].mean()
titanic_train['Fare'].fillna(fare_means, inplace = True)
count, bin_dividers_Fare = np.histogram(titanic_train['Fare'], bins=6)
bin_name = ['85▼','85','170','256','341','426▲']
titanic_train['Fare_cut'] = pd.cut(titanic_train['Fare'], bins = bin_dividers_Fare, labels = bin_name, include_lowest=True)
hist_col = ['Age_cut', 'SibSp','Parch','Fare_cut']
fig, axe = plt.subplots(1,4 , figsize=(20,5))
x = 0
for i in hist_col:
    sns.barplot(x = i, y = 'Survived' ,data = titanic_train, ax = axe.flatten()[x])
    x+=1
fig.tight_layout()
titanic_train['Cabin']

titanic_train['Cabin'] = titanic_train['Cabin'].str[:1]
titanic_train['Embarked'].fillna('S', inplace = True)
titanic_train['Cabin'].fillna('X', inplace = True)

dist_col = ['Sex', 'Cabin', 'Embarked']
ff, axx = plt.subplots(1,3, figsize = (12,5))
x = 0
for i in dist_col:
    sns.countplot(x = i,hue = 'Survived', data = titanic_train, ax = axx.flatten()[x])
    x+=1
ff.tight_layout()
f, ax = plt.subplots(1,2,figsize=(12,5))
table = titanic_train.pivot_table(index = ['Sex','Survived'], columns =['Pclass'], aggfunc='size')
sns.heatmap(table, annot = True, fmt='d', cmap = 'YlGnBu', linewidth = .5, cbar = False, ax = ax[0])
sns.barplot(x= 'Sex', y='Survived', hue = 'Pclass', data = titanic_train, ax = ax[1])

fff, axx = plt.subplots(1,4,figsize=(20,5))
sns.barplot(x='Embarked', y='Age',hue='Survived', data=titanic_train, ax=axx[0])
sns.barplot(x='Embarked', y='Fare',hue='Survived', data=titanic_train, ax=axx[1])
sns.barplot(x='Embarked', y='SibSp',hue='Survived', data=titanic_train, ax=axx[2])
sns.barplot(x='Embarked', y='Parch',hue='Survived', data=titanic_train, ax=axx[3])
fff, axx = plt.subplots(1,4,figsize=(20,5))
sns.barplot(x='Pclass', y='Age',hue='Survived', data=titanic_train, ax=axx[0])
sns.barplot(x='Pclass', y='Fare',hue='Survived', data=titanic_train, ax=axx[1])
sns.barplot(x='Pclass', y='SibSp',hue='Survived', data=titanic_train, ax=axx[2])
sns.barplot(x='Pclass', y='Parch',hue='Survived', data=titanic_train, ax=axx[3])
sns.lmplot('Fare', 'Survived', data=titanic_train)
import re

def cleaning(x):
    tikets = re.compile('[^a-zA-z+]')
    clean_ticket = tikets.sub('', x)
    return clean_ticket

titanic_train['Ticket_cut'] = titanic_train['Ticket'].apply(lambda x : cleaning(x))
titanic_train['Ticket_cut'].replace('', 'number', inplace = True)

plt.subplots(1,1,figsize = (40,20))
sns.countplot(x= 'Ticket_cut', hue = 'Survived', data =titanic_train)
titanic_train['Ticket_cut'].value_counts()
titanic_train['Family'] = titanic_train['SibSp'] + titanic_train['Parch']
table = titanic_train.pivot_table(index = ['Survived'], columns =['Family'], aggfunc='size')

fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.heatmap(table, annot = True, cmap = 'YlGnBu', linewidth = .5, cbar = False, ax = ax[0])
sns.countplot('Family', hue='Survived', data = titanic_train, ax = ax[1])
titanic_train['Name_cut'] = titanic_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


titanic_train['Name_cut'] = titanic_train['Name_cut'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
titanic_train['Name_cut'] = titanic_train['Name_cut'].replace(['Mlle','Mme','Ms','Mr'], 'Other')
titanic_train['Name_cut'] = titanic_train['Name_cut'].replace(['Mrs'], 'Miss')
titanic_train['Name_cut'].value_counts()
sns.countplot(x = 'Name_cut', hue = 'Survived', data = titanic_train)
from sklearn.preprocessing import LabelEncoder

titanic_train=pd.read_csv(train_path)
titanic_test=pd.read_csv(test_path)

def data_clean(df):
    
    def cleaning(x):
        tikets = re.compile('[^a-zA-z+]')
        clean_ticket = tikets.sub('', x)
        return clean_ticket
    

    df.drop('Name', axis = 1, inplace = True)
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Cabin'].fillna('X', inplace = True)
    df['Embarked'].fillna('S', inplace= True)
    
    for i in range(len(df['Age'])):
        if df['Age'][i] <= 10:
            df['Age'][i] = 1
        elif df['Age'][i] <= 30:
            df['Age'][i] = 2
        elif df['Age'][i] <= 50:
            df['Age'][i] = 3
        else:
            df['Age'][i] = 4
            
    for i in range(len(df['Fare'])):
        if df['Fare'][i] <= 7.910400:
            df['Fare'][i] = 1
        elif df['Fare'][i] <= 14.454200:
            df['Fare'][i] = 2
        elif df['Fare'][i] <= 31.000000:
            df['Fare'][i] = 3
        elif df['Fare'][i] <= 86:
            df['Fare'][i] = 4
        else:
            df['Fare'][i] = 5
            
    
    df['Cabin'] = df['Cabin'].str[:1]
    
    df['Ticket'] = df['Ticket'].apply(lambda x : cleaning(x))
    df['Ticket'].replace('', 'number', inplace = True)
    
    df['Family'] = df['SibSp'] + df['Parch']
    
    
    
    return df
    
def Labelencoding(df):
    
    features = ['Cabin', 'Embarked', 'Sex', 'Ticket']
    
    for feature in features:
        Le = LabelEncoder()
        le = Le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df



    
titanic_train = data_clean(titanic_train)        
titanic_train = Labelencoding(titanic_train)
titanic_test = data_clean(titanic_test)
titanic_test = Labelencoding(titanic_test)
label = titanic_train['Survived']
titanic_train.drop('Survived', axis = 1 , inplace = True)
X = titanic_train
titanic_test.columns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRFClassifier,XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

X_train, X_test, y_train, y_test = train_test_split(X,label, test_size = 0.2, random_state = 42)
cv=KFold(n_splits=10, random_state=42)

model=XGBRFClassifier()
param_grid={'booster' :['gbtree'],
                 'max_depth':[4,6,8],
                 'gamma':[0,1,2,3],
                 'n_estimators':[250, 350, 450],
                 'random_state':[42],
                'learning_rate':[0.1]}

xgbrf_clf =GridSearchCV(model, param_grid=param_grid,scoring = 'accuracy',cv = cv, n_jobs=-1)
 

model=XGBClassifier()
param_grid={'booster' :['gbtree'],
                 'max_depth':[4,6,8],
                 'gamma':[0,1,2,3],
                 'n_estimators':[250, 350, 450],
                 'random_state':[42],
                'learning_rate':[0.1]}

xgb_clf =GridSearchCV(model, param_grid=param_grid,scoring = 'accuracy',cv = cv, n_jobs=-1)

model=RandomForestClassifier()
param_grid={     
                 'max_depth':[4,6,8],
                 'n_estimators':[250, 350, 450],
                 'max_features': ['auto'],
                 'random_state':[42],
            }

rf_clf=GridSearchCV(model, param_grid=param_grid,scoring = 'accuracy',cv = cv, n_jobs=-1)

ada_clf_rf = AdaBoostClassifier(algorithm='SAMME.R',base_estimator=RandomForestClassifier(n_estimators=200, max_depth= 4, min_samples_leaf= 2, min_samples_split= 6),
                             random_state=42,n_estimators=300,
                            learning_rate = 0.1)

ada_clf_ds = AdaBoostClassifier(algorithm='SAMME.R',base_estimator=DecisionTreeClassifier(max_depth= 4, min_samples_leaf= 2, min_samples_split= 6),
                             random_state=42,n_estimators=300,
                            learning_rate = 0.1)

grd_clf = GradientBoostingClassifier(max_depth=2, n_estimators=250)


models=[
        ('xgbrf_clf', xgbrf_clf),
        ('xgb_clf', xgb_clf),
        ('rf_clf', rf_clf),
        ('ada_clf_rf', ada_clf_rf),
        ('ada_clf_ds',ada_clf_ds),
        ('grd_clf',grd_clf),
    
]


vot_soft = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
vot_soft.fit(X_train, y_train)
soft_pred = vot_soft.predict(titanic_test)
a = pd.DataFrame({
    'PassengerId' : titanic_test['PassengerId'],
    'Survived' : soft_pred
})

a.to_csv('good_score.csv', index = False)