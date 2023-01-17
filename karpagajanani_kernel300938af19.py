# data processing
import pandas as pd

# linear algebra
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
data.head()

data.info()

100 * data.isnull().sum()/len(data)
data.describe()
data.shape
data.dtypes.unique()
## Numeric columns
num_cols = [col for col in data.columns if data[col].dtype in ['int64','float64']]
len(num_cols)
## Categorical columns
cat_cols = [col for col in data.columns if data[col].dtype in ['object']]
len(cat_cols)
data[num_cols].head()
data[cat_cols].head()
plt.figure(figsize=(40,90))
for i in enumerate(cat_cols):
    ax=plt.subplot(45,1,i[0]+1)
    sns.boxplot(x=i[1],y='Survived',data=data)
def bivariate_continuos(df, target):
    
    cols = list(df.columns)
    
    for i in enumerate(cols):
        plt.figure(figsize=(20,90))
        plt.subplot(25,4,i[0]+1)
        sns.boxplot(x=target,y=i[1],data=df)
        plt.yscale('log')
        plt.show()
bivariate_continuos(data[num_cols], data['Survived'])
pd.crosstab(data.Survived,data.Sex,normalize=True)
bins= [0,16,32,48,64,150]
labels = [0,1,2,3,4]
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
print (data)
sns.countplot('AgeGroup',hue='Survived',data=data)

pd.crosstab(data.Survived,data.AgeGroup,normalize=True)
data.drop("AgeGroup", axis = 1, inplace = True)
plt.figure(figsize=(15,6))
sns.heatmap(data.corr(), vmax=0.6, square=True, annot=True)
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=data, split=True, ax=ax1)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=data, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=data, split=True, ax=ax3)
data.drop("Cabin", axis = 1, inplace = True)
test_df.drop("Cabin", axis = 1, inplace = True)
data.Embarked.value_counts()
data["Embarked"]=data["Embarked"].fillna("S")
test_df["Embarked"]=test_df["Embarked"].fillna("S")
data.head()
titles = set()
for name in data['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
# we extract the title from each name
data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
test_df['Title'] = test_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

# a map of more aggregated title
# we map each title

data['Title'] = data.Title.map(Title_Dictionary)
test_df['Title'] = test_df.Title.map(Title_Dictionary)

data[data['Title'].isnull()]
test_df[test_df['Title'].isnull()]
test_df['Title'].fillna("Royalty",inplace=True)
#Lets drop name and ticket
data.drop(["Name",'Ticket'], axis = 1, inplace = True)
test_df.drop(["Name",'Ticket'], axis = 1, inplace = True)
data.head()
sns.countplot(data['Title'],hue='Survived',data=data)

# Lets drop passenger id since it is not important
data.drop("PassengerId", axis = 1, inplace = True)
test_df.drop("PassengerId", axis = 1, inplace = True)
data.head()
data.dtypes
data.Sex.replace(['female','male'],[0,1],inplace=True)
test_df.Sex.replace(['female','male'],[0,1],inplace=True)
data.Embarked.replace(['S','C','Q'],[0,1,2],inplace=True)
test_df.Embarked.replace(['S','C','Q'],[0,1,2],inplace=True)
data.Title.replace(['Officer','Royalty','Mr','Mrs','Miss','Master'],[0,1,2,3,4,5],inplace=True)
test_df.Title.replace(['Officer','Royalty','Mr','Mrs','Miss','Master'],[0,1,2,3,4,5],inplace=True)

from fancyimpute import IterativeImputer
ii=IterativeImputer()
df_clean=pd.DataFrame(ii.fit_transform(data))
df_clean.columns=data.columns
test_df_clean=pd.DataFrame(ii.fit_transform(test_df))
test_df_clean.columns=test_df.columns

100 * df_clean.isnull().sum()/len(df_clean)
100 * test_df_clean.isnull().sum()/len(test_df_clean)
sns.boxplot(df_clean['Age'])
df_clean.drop(df_clean.index[df_clean['Age'] <0],inplace=True)
sns.boxplot(df_clean['Age'])
sns.boxplot(df_clean['Fare'])

df_clean.drop(df_clean.index[df_clean['Fare'] >500],inplace=True)
sns.boxplot(df_clean['Fare'])

df_clean.columns
for i in ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch','Embarked','Title']:
  df_clean[i] = pd.to_numeric(df_clean[i])
  df_clean[i] = df_clean[i].astype(int)
df_clean.head()
df_clean.drop("Fare",axis=1)
cat_df = df_clean[['Pclass', 'SibSp', 'Parch','Embarked','Title']]
cat_df.head()
df_dummies = pd.get_dummies(cat_df, drop_first = True)
df_dummies.head()
pd.concat([df_clean,df_dummies],axis=1)
df_clean.head()
df_clean.columns,test_df_clean.columns
X_train = df_clean.drop("Survived", axis=1)
y_train = df_clean["Survived"]
X_test  = test_df_clean.copy()
X_train.shape, y_train.shape, X_test.shape
## Scaling the train and test data
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train )
X_test = scaler.transform(X_test)
100 * y_train.value_counts(normalize=True)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(penalty = 'l1' , solver  = 'saga')
logreg.fit(X_train , y_train)

y_pred_l1=logreg.predict(X_test)

aux = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.DataFrame({
        "PassengerId": aux["PassengerId"],
        "Survived": y_pred_l1
    })

submission[['PassengerId','Survived']].to_csv('LogReg_pen_l1_saga_V2.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(class_weight = "balanced",criterion='gini',min_samples_split=16,n_estimators=700)
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)

aux = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.DataFrame({
        "PassengerId": aux["PassengerId"],
        "Survived": y_pred_rf
    })

#submission[['PassengerId','Survived']].to_csv('RF_V1.csv', index=False)
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
clf = RandomForestClassifier()
clf = clf.fit(X_train,y_train)
features = pd.DataFrame()
features['feature'] = df_clean.drop("Survived", axis=1).columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 15))
logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()

models = [logreg, logreg_cv, rf]
for model in models:
    print(model.__class__)
    score = compute_score(clf=model, X=X_train, y=y_train, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('------------------------------------')
# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
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

    grid_search.fit(X_train, y_train)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else: 
    parameters = {'bootstrap': True, 'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 50}
    
    model = RandomForestClassifier(**parameters)
    model.fit(X_train, y_train)
output = model.predict(X_test).astype(int)
aux = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.DataFrame({
        "PassengerId": aux["PassengerId"],
        "Survived": output
    })

submission[['PassengerId','Survived']].to_csv('RF_HP_FT_V9.csv', index=False)
