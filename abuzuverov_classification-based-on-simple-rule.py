import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from fancyimpute import KNN
from fancyimpute import IterativeImputer
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')
# Load data
train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)

# Merge train and test sets
test.insert(1,'Survived',np.nan)
all = pd.concat([train, test])
# Perform corrections
corr_dict = {248: pd.Series([0,1], index=['SibSp', 'Parch'],),
             313: pd.Series([1,0], index=['SibSp', 'Parch'],),
             418: pd.Series([0,0], index=['SibSp', 'Parch'],),
             756: pd.Series([0,1], index=['SibSp', 'Parch'],),
             1041: pd.Series([1,0], index=['SibSp', 'Parch'],),
             1130: pd.Series([0,0], index=['SibSp', 'Parch'],),
             1170: pd.Series([2,0], index=['SibSp', 'Parch'],),
             1254: pd.Series([1,0], index=['SibSp', 'Parch'],),
             1274: pd.Series([1,0], index=['SibSp', 'Parch'],),
             539: pd.Series([1,0], index=['SibSp', 'Parch'],)
             }

all[['SibSp','Parch']] = all.apply(lambda s: corr_dict[s['PassengerId']]
    if s['PassengerId'] in [248,313,418,756,1041,1130,1170,1254,1274,539] else s[['SibSp','Parch']], axis = 1)
# Add Title
all['Title'] =  all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# Replace rare titles
all.loc[all['Title'].isin(['Ms','Mlle']), 'Title'] = 'Miss'
all.loc[all['Title'].isin(['Mme','Lady','Dona','Countess']), 'Title'] = 'Mrs'
all.loc[all['Title'].isin(['Col','Major','Sir','Rev','Capt','Don','Jonkheer']), 'Title'] = 'Mr'
all.loc[(all['Title'] == 'Dr') & (all['Sex'] == 'male'),'Title'] = 'Mr'
all.loc[(all['Title'] == 'Dr') & (all['Sex'] == 'female'),'Title'] = 'Mrs'
# Add Family Size and is-Alone
all['FamSize'] = all.apply(lambda s: 1+s['SibSp']+s['Parch'], axis = 1)
all['isAlone'] = all.apply(lambda s: 1 if s['FamSize'] == 1 else 0, axis = 1)
# Add Group Size
ticket_counts = all['Ticket'].value_counts()
all['GrSize'] = all.apply(lambda s: ticket_counts.loc[s['Ticket']], axis=1)

# Add has-Cabin
all['Cabin'].fillna('U',inplace=True)
all['hasCabin'] = all.apply(lambda s: 0 if s['Cabin'] == 'U' else 1,axis = 1)
# Add Family Name
all['Fname'] =  all.Name.str.extract('^(.+?),', expand=False)

# Search for passengers with siblings
Pas_wSib = []
all_x_0 = all[(all['SibSp'] > 0) & (all['Parch'] == 0)]
name_counts_SibSp = all_x_0['Fname'].value_counts()
for label, value in name_counts_SibSp.items():
    entries = all_x_0[all_x_0['Fname'] == label]
    if (entries.shape[0] > 1 and (not (entries['Title'] == 'Mrs').any())) or \
       (entries.shape[0] == 1 and entries['Title'].values[0] == 'Mrs'):
            Pas_wSib.extend(entries['PassengerId'].values.tolist())
    else:
        Pas_wSib.extend( \
            entries[(entries['Title'] == 'Miss')|(entries['GrSize'] == 1)]['PassengerId'].values.tolist())

# Search for Mrs-es with parents
Mrs_wPar = []
all_x_y = all[all['Parch'] > 0]
name_counts_Parch = all_x_y['Fname'].value_counts()
for label, value in name_counts_Parch.items():
    entries = all_x_y[all_x_y['Fname'] == label]
    if entries.shape[0] == 1:
        if entries['Title'].values[0] == 'Mrs' and entries['Age'].values[0] <= 30:
            Mrs_wPar.extend(entries['PassengerId'].values.tolist())

def get_features(row):

    features = pd.Series(0, index = ['wSib','wSp','wCh','wPar'])

    if row['PassengerId'] in Pas_wSib:
        features['wSib'] = 1
    else:
        if (row['SibSp'] != 0) & (row['Parch'] == 0):
            features['wSp'] = 1
        else:
            if  ( (row['Title']=='Mrs')&(not row['PassengerId'] in Mrs_wPar) )| \
                ( (row['Title']=='Mr')&(not row['PassengerId'] == 680)&
                                        ( ((row['Pclass']==1)&(row['Age']>=30))|
                                          ((row['Pclass']==2)&(row['Age']>=25))|
                                          ((row['Pclass']==3)&(row['Age']>=20)) ) ):
                features['wCh'] = 1
            else:
                features['wPar'] = 1

    return features

all[['wSib','wSp','wCh','wPar']] = all.apply(lambda s: get_features(s) if s['isAlone'] == 0 else [0,0,0,0], axis = 1)
all = all.drop(['Fname','Name','Cabin','Ticket','Fare','SibSp','Parch'], axis = 1)
all[all['Pclass'] == 1].groupby(['Title','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])
all[(all['Pclass'] == 1)&(all['Title'] == 'Mr') ].groupby(['hasCabin','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])
all[all['Pclass'] == 2].groupby(['Title','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])
all[all['Pclass'] == 3].groupby(['Title','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])
all[(all['Pclass'] == 3)&(all['Title'] != 'Mr')].groupby(['Title','FamSize'])['Survived'].agg(['count','size','mean'])
# Make FamSize bins
all['FamSizeBin'] = pd.cut(all['FamSize'], bins = [0,4,11], labels = False)
all = all.drop(['FamSize'], axis = 1)
all[(all['Pclass'] == 3)&(all['Title'] != 'Mr')].groupby(['Title','FamSizeBin','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','size','mean'])
def get_survived_1(row):
    if row['Pclass'] in [1,2]:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSizeBin'] == 1:
            survived = 0
        else:
            survived = 1

    return survived
# Form train and test sets
X_train = all.iloc[:891,:]
X_test = all.iloc[891:,:]
y_train = all.iloc[:891,:]['Survived']

# Make predictions (train)
y_train_hat = X_train.apply(lambda s: get_survived_1(s), axis = 1)

# Make predictions (test)
predictions = pd.DataFrame( {'PassengerId': test['PassengerId'], 'Survived': 0} )
predictions['Survived'] = X_test.apply(lambda s: get_survived_1(s), axis = 1)
predictions.to_csv('submission-1.csv', index=False)

# Train score
score = metrics.accuracy_score(y_train_hat, y_train)
print('Train Accuracy: {}'.format(score))
all[(all['Pclass'] == 3)&(all['Title'] != 'Mr')&(all['FamSizeBin'] == 0)].groupby(['Title','Embarked'])['Survived'].agg(['count','size','mean'])
def get_survived_2(row):
    if row['Pclass'] in [1,2]:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSizeBin'] == 1 or (row['Title'] == 'Miss' and row['Embarked'] == 'S'):
            survived = 0
        else:
            survived = 1

    return survived
# Make predictions (train)
y_train_hat = X_train.apply(lambda s: get_survived_2(s), axis = 1)

# Make predictions (test)
predictions['Survived'] = X_test.apply(lambda s: get_survived_2(s), axis = 1)
predictions.to_csv('submission-2.csv', index=False)

# Train score
score = metrics.accuracy_score(y_train_hat, y_train)
print('Train Accuracy: {}'.format(score))
all[(all['Pclass'] == 3)&(all['Title'] == 'Miss')&(all['FamSizeBin'] == 0)].groupby(['Title','wPar','Embarked'])['Survived'].agg(['count','size','mean'])
def get_survived_3(row):
    if row['Pclass'] in [1,2]:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSizeBin'] == 1 or \
        (row['Title'] == 'Miss' and row['Embarked'] == 'S' and row['wPar'] == 0):
            survived = 0
        else:
            survived = 1

    return survived
# Make predictions (train)
y_train_hat = X_train.apply(lambda s: get_survived_3(s), axis = 1)

# Make predictions (test)
predictions['Survived'] = X_test.apply(lambda s: get_survived_3(s), axis = 1)
predictions.to_csv('submission-3.csv', index=False)

# Train score
score = metrics.accuracy_score(y_train_hat, y_train)
print('Train Accuracy: {}'.format(score))
# Select and convert categorical features into numerical ones (1)
all['Sex'] = all['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
all['Embarked'].fillna(all['Embarked'].value_counts().index[0], inplace=True)
all_dummies =  pd.get_dummies(all, columns = ['Title','Pclass','Embarked'],\
                                 prefix=['Title','Pclass','Embarked'], drop_first = True)
all_dummies = all_dummies.drop(['PassengerId','Survived'], axis = 1)
# KNN imputation
all_dummies_i = pd.DataFrame(data=KNN(k=3, verbose = False).fit_transform(all_dummies).astype(int),
                            columns=all_dummies.columns, index=all_dummies.index)
# Convert categorical features into numerical ones (2)
all_dummies_i['isAlwSib'] = all_dummies_i.apply(lambda s: 1 if (s['isAlone'] == 1)|(s['wSib'] == 1) else 0 ,axis = 1)
all_dummies_i = all_dummies_i.drop(['isAlone','wSib','Sex','GrSize'], axis = 1)
# Form train and test sets
X_train = all_dummies_i.iloc[:891,:]
X_test = all_dummies_i.iloc[891:,:]
# Perform scaling
scaler = StandardScaler()
scaler.fit(X_train[['Age']])
X_train['Age'] = scaler.transform(X_train[['Age']])
X_test['Age'] = scaler.transform(X_test[['Age']])
# Cross-validation parameters
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
# Grid search parameters
svm_grid = {'C': [10,11,12,13,14,15,16,17,18,19,20], 'gamma': ['auto']}
svm_search = GridSearchCV(estimator = SVC(), param_grid = svm_grid, cv = cv, refit=True, n_jobs=1)
# Apply grid search
svm_search.fit(X_train, train['Survived'])
svm_best = svm_search.best_estimator_
print("Cross-validation accuracy: {}, standard deviation: {}, with parameters {}"
       .format(svm_search.best_score_, svm_search.cv_results_['std_test_score'][svm_search.best_index_],
               svm_search.best_params_))
y_train_hat = svm_best.predict(X_train)
print('Train Accuracy: {}'
        .format(metrics.accuracy_score(y_train_hat, y_train)))

predictions['Survived'] = svm_best.predict(X_test)
predictions.to_csv('submission-svm.csv', index=False)
def get_survived_svm_rule(row):
    if row['Pclass'] in [1,2]:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSizeBin'] == 1 or \
        (row['Title'] == 'Miss' and row['Embarked'] == 'S' and row['Age'] >= 18):
            survived = 0
        else:
            survived = 1

    return survived