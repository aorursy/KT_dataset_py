import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv("../input/adult-census-income/adult.csv")
df.head()
l = ['race','relationship','sex','hours.per.week','marital.status','education','workclass','age','education.num','relationship']

for i in l:

    plt.figure(i,figsize=(15,5))

    sns.countplot(df[i])
plt.figure(figsize=(15,6))

sns.heatmap(df.corr(),annot=True)
df[df=='?'] = np.nan
df.isnull().sum()
for i in ['workclass','occupation','native.country']:

    df[i].fillna(df[i].mode()[0],inplace=True)
df.head()

df['income']=df['income'].map({'<=50K': 0, '>50K': 1})
sns.catplot('education.num','income',data=df,kind='bar')
sns.catplot('marital.status','income',data=df,kind='bar',height=5,aspect=3)
sns.catplot('education.num','income',data=df,kind='bar',height=5,aspect=3)
l = ['race','relationship','sex','hours.per.week','marital.status','education','workclass','age','education.num','relationship']

for i in l:

    plt.figure(i,figsize=(15,5))

    sns.catplot(i,'income',data=df,kind='bar',height=5,aspect=3)
df[(df['age']>=90) & (df['income']==1)]
df['native.country'].value_counts()[:15].plot(kind='bar')
df.head()
df_columns = df.select_dtypes('object').columns

lnc = LabelEncoder()

for i in df_columns:

    df[i] = lnc.fit_transform(df[i])
numerical_column_names = ['age','fnlwgt','education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

sc = StandardScaler()

df[numerical_column_names] = sc.fit_transform(df[numerical_column_names])
df.head()
X = df.drop(['income'],axis=1)

y = df['income']
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier

from lightgbm import LGBMClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV,train_test_split
cross_val_score(DecisionTreeClassifier(),X,y)
cross_val_score(ExtraTreesClassifier(),X,y)
cross_val_score(RandomForestClassifier(),X,y)
cross_val_score(GaussianNB(),X,y)
cross_val_score(LogisticRegression(),X,y)
cross_val_score(KNeighborsClassifier(),X,y)
cross_val_score(XGBClassifier(),X,y)
cross_val_score(LGBMClassifier(),X,y)
# Best Algorith:

#     LGBMClassifier,

#     XGBClassifier,

#     KNeighborsClassifier,

#     LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# For Logisctic

params = {

    'penalty' : ['l1', 'l2'],

    'C' : np.logspace(-4, 4, 20),

    'solver' : ['liblinear']}

grid_search = GridSearchCV(estimator=LogisticRegression(),param_grid=params,n_jobs=-1,cv=10)

grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

print(grid_search.best_score_)
cross_val_score(LogisticRegression(**grid_search.best_params_,),X,y)

#not much effect
# # RandomForest

# params = {

#     'n_estimators' : list(range(10,50,10)),

#     'max_features' : list(range(6,32,5))

# }



# grid_search = GridSearchCV(estimator=RandomForestClassifier(),param_grid=params,n_jobs=-1)

# grid_search.fit(X_train,y_train)
params = {"learning_rate"    : [0.25, 0.30,0.35,0.40] ,

         "n_estimators":[100,200,300]}

grid_search = GridSearchCV(estimator=XGBClassifier(),param_grid=params,n_jobs=-1)

grid_search.fit(X_train,y_train)
print(grid_search.best_score_)

print(grid_search.best_params_)
params = [{'n_neighbors': list(range(1,20)),

'weights': ['distance'],

'algorithm': ['kd_tree']}]



grid_search = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=params,n_jobs=-1)

grid_search.fit(X_train,y_train)
print(grid_search.best_score_)

print(grid_search.best_params_)
params = {

"criterion" : ['gini', 'entropy'],

"max_depth" : [4,6,8,12,13,15,17],

}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params,n_jobs=-1)

grid_search.fit(X_train,y_train)
print(grid_search.best_score_)

print(grid_search.best_params_)