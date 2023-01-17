# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.preprocessing import LabelEncoder

import re

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, learning_curve, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier

%matplotlib inline

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.info()
train.describe()
train[train['Survived'] == 1]['Survived'].count()/train['Survived'].count()
sns.countplot(x='Survived', data=train, hue='Sex')
train[["Sex", "Survived"]].groupby('Sex', as_index=False).mean()
sns.distplot(train['Fare'])
train['Fare'].mean()
train[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()
# box plot of age for pclass

plt.figure(figsize=(12, 8))

sns.boxplot(x='Pclass', y='Age', hue='Survived', data=train)
sns.distplot(train['Age'].dropna())
train['Age'].dropna().mean()
g = sns.FacetGrid(train, col='Survived', row='Pclass')

g.map(plt.hist, 'Age', bins=20)
sns.countplot(x='Embarked', hue='Survived', data=train)
plt.figure(figsize=(12, 5))

sns.countplot(x='Parch', hue='Survived', data=train)
plt.figure(figsize=(12, 5))

sns.countplot(x='SibSp', hue='Survived', data=train)
sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar")
def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop] 
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


train_len = len(train)
combined = pd.concat([train, test], ignore_index=True)
combined.drop('PassengerId', axis=1, inplace=True)
sns.heatmap(combined.isnull())
np.sum(train.isnull())
np.sum(test.isnull())
embarked_mode = train['Embarked'].mode()[0]
embarked_mode
combined['Embarked'] = combined['Embarked'].fillna(embarked_mode)
combined['Cabin'].dropna().apply(str).apply(lambda x: x[0]).unique()
combined['Cabin'].fillna('U', inplace=True)
combined['Cabin'] = combined['Cabin'].apply(str).apply(lambda x : x[0])
sns.countplot(combined["Cabin"])
np.where(np.isnan(combined['Fare']))
combined.iloc[1033]
sns.heatmap(train[["Fare","Age","Sex","SibSp","Parch","Pclass"]].corr(), annot=True, cmap='coolwarm')
train[(train['Pclass'] == 3)]['Fare'].median()
combined['Fare'][1033] = train[(train['Pclass'] == 3)]['Fare'].median()
# combined["Fare"] = combined["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
train['Name'].head()
combined['Title'] = combined['Name'].apply(lambda x : (x.split(', ')[1].split('.')[0]))

combined.drop('Name', axis=1, inplace=True)
combined['Title'].unique()
combined['Title'].value_counts()
titles_map = {

 'Capt' : 'Rare',

 'Col' : 'Rare',

 'Don': 'Rare',

 'Dona': 'Rare',

 'Dr' : 'Rare',

 'Jonkheer' :'Rare' ,

 'Lady': 'Rare',

 'Major': 'Rare',

 'Master': 'Master',

 'Miss' : 'Miss',

 'Mlle' : 'Rare',

 'Mme': 'Rare',

 'Mr': 'Mr',

 'Mrs': 'Mrs',

 'Ms': 'Rare',

 'Rev': 'Rare',

 'Sir': 'Rare',

 'the Countess': 'Rare'

}
combined['Title'] = combined['Title'].map(titles_map)
combined[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
sns.heatmap(train[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot=True, cmap='coolwarm')
def impute_age(row):

    pclass = row['Pclass']

    parch = row['Parch']

    sibsp = row['SibSp']

    age = row['Age']

    if pd.isnull(age):

        age_median = train['Age'].median()

        similar_age =  train[(train['Pclass'] == pclass) & (train['Parch'] == parch)

                       & (train['SibSp'] == sibsp)]['Age'].median()

        if( similar_age > 0): return similar_age

        else :  return age_median

    else :return age
combined['Age'] = combined.apply(impute_age, axis=1)
combined['Family_size'] = combined.apply(lambda row : 1 + (row['Parch'] + row['SibSp']), axis=1)

combined['Alone'] = combined.apply(lambda row : 1 if (row['Parch'] + row['SibSp']) == 0 else 0, axis=1)
sns.countplot(x='Family_size' , data=combined, hue='Survived')
combined['Small_family'] = combined.apply(lambda row : 1 if 2 <= (row['Family_size']) <= 4 else 0, axis=1)
combined.drop(['Parch', 'SibSp'], axis=1, inplace=True)
combined['Ticket'] = combined['Ticket'].apply(lambda x : 'X' if x.isdigit() else x)
combined['Ticket'] = combined['Ticket'].apply(lambda x : re.sub("[\d\.]", "", x).split('/')[0].strip() if not x.isdigit() else x)
combined = pd.get_dummies(combined, columns = ["Embarked"], prefix="Em")



combined = pd.get_dummies(combined, columns = ["Cabin"], prefix="Cb")



combined = pd.get_dummies(combined, columns = ["Title"], prefix="Title")
sex_encoder = LabelEncoder().fit(combined['Sex'])

combined['Sex'] = sex_encoder.transform(combined['Sex'])
ticket_encoder = LabelEncoder().fit(combined['Ticket'])

combined['Ticket'] = ticket_encoder.transform(combined['Ticket'])
np.sum((combined.drop('Survived', axis=1).isnull()))
combined.head()
train = combined[:train_len]

test = combined[train_len:]
test.drop('Survived', axis=1, inplace=True)
np.sum((test.isnull()))
train.head()
train.head()
train['Survived'] = train['Survived'].astype(int) 
X = train.drop('Survived', axis=1)

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


random_state = 42

 

model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'SVC', 

              'RandomForestClassifier', 'XGBClassifier', 'ExtraTreesClassifier'

              , 'GradientBoostingClassifier','AdaBoostClassifier','GaussianNB']

models = [ ('LogisticRegression',LogisticRegression(random_state=random_state)),

          ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),

          ('SVC', SVC(random_state=random_state)),

          ('RandomForestClassifier',RandomForestClassifier(random_state=42)),

          ('XGBClassifier',XGBClassifier(random_state=random_state)),

          ('ExtraTreesClassifier',ExtraTreesClassifier(random_state=random_state)),

          ('GradientBoostingClassifier',GradientBoostingClassifier(random_state=random_state)),

          ('AdaBoostClassifier',AdaBoostClassifier(random_state=random_state)),

          ('GaussianNB',GaussianNB())

         ]



model_accuracy = []

for k,model in models:

    model.fit(X, y)

    accuracy = cross_val_score(model, X_train, y_train, cv=10).mean()

    model_accuracy.append(accuracy)

pd.concat([pd.Series(model_names), pd.Series(model_accuracy)], axis=1).sort_values(by=1, ascending=False)
best_models=[]



xgboot_param_grid = {

     'n_estimators': [100,200,300],

     'max_depth': [4, 6, 8],

     'learning_rate': [.4, .45, .5, .55, .6],

     'colsample_bytree': [.6, .7, .8, .9, 1]

}



ada_param_grid = {

 'n_estimators':[100,200,300],

 'learning_rate' : [0.01,0.05,0.1,0.3,1],

 'algorithm' : ['SAMME', 'SAMME.R']

 }



gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300, 400],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10, 15],

              "min_samples_split": [2, 3, 10, 15],

              "min_samples_leaf": [1, 3, 10, 15],

              "bootstrap": [False],

              "n_estimators" :[100,200,300, 400],

              "criterion": ["gini"]}





rf_param_grid  = { 

    'n_estimators': [100,200,300, 400],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}



log_param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}



svv_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



models = [ 

    ('AdaBoostClassifier',AdaBoostClassifier(), ada_param_grid),

          ('XGBClassifier',XGBClassifier(), xgboot_param_grid),

          ('GradientBoostingClassifier',GradientBoostingClassifier(), gb_param_grid),

        ('RandomForestClassifier',RandomForestClassifier(), rf_param_grid),

          ('ExtraTreesClassifier',ExtraTreesClassifier(), ex_param_grid),

    ('SVC',SVC(probability=True), svv_param_grid),

    ('LogisticRegression',LogisticRegression(), log_param_grid)

         ]





for name, model, param in  models:

    grid_search = GridSearchCV(model,

                               scoring='accuracy',

                               param_grid=param,

                               cv=10,

                               verbose=2,

                               n_jobs=-1)

    grid_search.fit(X, y)

    print (name, ':', grid_search.best_score_, '\n')

    best_models.append(grid_search.best_estimator_)





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



for model in best_models:

    plot_learning_curve(model,model.__class__.__name__ + " RF mearning curves",X,y,cv=5)
def plot_feature_importances(clf, X_train, y_train=None, 

                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):

    '''

    plot feature importances of a tree-based sklearn estimator

    

    Note: X_train and y_train are pandas DataFrames

    

    Note: Scikit-plot is a lovely package but I sometimes have issues

              1. flexibility/extendibility

              2. complicated models/datasets

          But for many situations Scikit-plot is the way to go

          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

    

    Parameters

    ----------

        clf         (sklearn estimator) if not fitted, this routine will fit it

        

        X_train     (pandas DataFrame)

        

        y_train     (pandas DataFrame)  optional

                                        required only if clf has not already been fitted 

        

        top_n       (int)               Plot the top_n most-important features

                                        Default: 10

                                        

        figsize     ((int,int))         The physical size of the plot

                                        Default: (8,8)

        

        print_table (boolean)           If True, print out the table of feature importances

                                        Default: False

        

    Returns

    -------

        the pandas dataframe with the features and their importance

        

    Author

    ------

        George Fisher

    '''

    

    __name__ = "plot_feature_importances"

    

    

    from xgboost.core     import XGBoostError

    from lightgbm.sklearn import LightGBMError

    

    try: 

        if not hasattr(clf, 'feature_importances_'):

            clf.fit(X_train.values, y_train.values.ravel())



            if not hasattr(clf, 'feature_importances_'):

                raise AttributeError("{} does not have feature_importances_ attribute".

                                    format(clf.__class__.__name__))

                

    except (XGBoostError, LightGBMError, ValueError):

        clf.fit(X_train.values, y_train.values.ravel())

            

    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    

    feat_imp['feature'] = X_train.columns

    feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    feat_imp = feat_imp.iloc[:top_n]

    

    feat_imp.sort_values(by='importance', inplace=True)

    feat_imp = feat_imp.set_index('feature', drop=True)

    feat_imp.plot.barh(title=title, figsize=figsize)

    plt.xlabel('Feature Importance Score')

    plt.show()

    

    if print_table:

        from IPython.display import display

        print("Top {} features in descending order of importance".format(top_n))

        display(feat_imp.sort_values(by='importance', ascending=False))

        

    return feat_imp
for model in best_models:

    try:

        _ = plot_feature_importances(model, X_train, y_train, top_n=X.shape[1], title=model.__class__.__name__)

    except AttributeError as e:

        print(e)
pred = []

for model in best_models:

    pred.append(pd.Series(model.predict(test), name=model.__class__.__name__))
pred = pd.DataFrame(pred).transpose()
pred
pred.sum()
g= sns.heatmap(pred.corr(),annot=True, cmap='coolwarm')
ids = pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId']

votingC = VotingClassifier(estimators=[

                                    ('ada', best_models[0]),

                                       ('rf', best_models[3]),

                                       ('ext', best_models[4]),

                                       ('log', best_models[6]),

                                      ], voting='soft', n_jobs=-1)

votingC.fit(X, y)

test_Survived = pd.Series(votingC.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("ensemble_prediction.csv",index=False)
ids = pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId']

votingC = VotingClassifier(estimators=[

                                    ('ada', best_models[0]),

                                       ('rf', best_models[3]),

                                       ('log', best_models[6]),

                                      ], voting='soft', n_jobs=-1)

votingC.fit(X, y)

test_Survived = pd.Series(votingC.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("ensemble_prediction2.csv",index=False)
ada_best = best_models[0]

ada_best.fit(X, y)

test_Survived = pd.Series(ada_best.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("Ada.csv",index=False)
xgb_best = best_models[1]

xgb_best.fit(X, y)

test_Survived = pd.Series(xgb_best.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("XGB.csv",index=False)
grb_best = best_models[2]

grb_best.fit(X, y)

test_Survived = pd.Series(grb_best.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("GRB.csv",index=False)
rnf_best = best_models[3]

rnf_best.fit(X, y)

test_Survived = pd.Series(rnf_best.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("RNF.csv",index=False)
ext_best = best_models[4]

ext_best.fit(X, y)

test_Survived = pd.Series(ext_best.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("EXT.csv",index=False)
svc_best = best_models[5]

svc_best.fit(X, y)

test_Survived = pd.Series(svc_best.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("SVC.csv",index=False)
log_best = best_models[6]

log_best.fit(X, y)

test_Survived = pd.Series(log_best.predict(test), name="Survived")



results = pd.concat([ids,test_Survived],axis=1)



results.to_csv("LOG.csv",index=False)