import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.feature_selection import SelectFromModel

import xgboost as xgb



# To ignore unwanted warnings

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train.head()
test.head()
train.info()

print('_'*40)

test.info()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

# Train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')

# Test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
#missing amount for train set

missing= train.isnull().sum().sort_values(ascending=False)

percentage = (train.isnull().sum()/ train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])

missing_data.head(3)
#missing amount for test set

missing= test.isnull().sum().sort_values(ascending=False)

percentage = (test.isnull().sum()/ test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])

missing_data.head(3)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Embarked'].value_counts()
isn = pd.isnull(test['Fare'])

test[isn]
average_of_fare = test.groupby('Pclass')['Fare'].mean()

test['Fare'].fillna(value= average_of_fare[3], inplace=True)
# combine train and test then find mean age grouped by Pclass

age_mean_by_pclass = pd.concat([train, test]).groupby('Pclass')['Age'].mean()

age_mean_by_pclass
train.loc[train['Age'].isnull(), 'Age'] = train['Pclass'].map(age_mean_by_pclass)

test.loc[test['Age'].isnull(), 'Age'] = test['Pclass'].map(age_mean_by_pclass)
# Sex & Age

g = sns.FacetGrid(train, hue = 'Survived', col = 'Sex', height = 3, aspect = 2)

g.map(plt.hist, 'Age', alpha = .5, bins = 20)

g.add_legend()

plt.show()
train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)
def age_range(df):

    df['Age'].loc[df['Age'] <= 16 ] = 0

    df['Age'].loc[(df['Age'] > 16) & (df['Age'] <= 32)] = 1

    df['Age'].loc[(df['Age'] > 32) & (df['Age'] <= 48)] = 2

    df['Age'].loc[(df['Age'] > 48) & (df['Age'] <= 64)] = 3

    df['Age'].loc[df['Age'] > 64] = 4   

age_range(train)

age_range(test)
train['Cabin'].isnull().sum()
train.groupby(train['Cabin'].isnull())['Survived'].mean()
# train

train['Cabin'] = train['Cabin'].notnull().astype('int')

# test

test['Cabin']=test['Cabin'].notnull().astype('int')
train_titles, test_titles = set(), set()  # empty sets to save titles



for train_name, test_name in zip(train['Name'], test['Name']):

    train_titles.add(train_name.split(',')[1].split('.')[0].strip())

    test_titles.add(test_name.split(',')[1].split('.')[0].strip())

print(train_titles,'\n', test_titles)
def title(df):

    # all the titles will be replaced to one of the following: 'Mr', 'Ms', Master, 'Officer', 'Royalty', 'Miss'

    df['Title'] = df['Name'].str.split(', ').str[1].str.split('.').str[0]

    

    df['Title'] = df['Title'].replace(['Capt','Col','Major','Dr','Rev'], 'Officer')

    df['Title'] = df['Title'].replace(['Mme','Ms'], 'Mrs')

    df['Title'] = df['Title'].replace(['Jonkheer','Don','Dona','Sir','the Countess','Lady'], 'Royalty')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')



title(train)

title(test)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['FamilySize'].value_counts()
train['FamilySize'] = train['FamilySize'].astype(int)

test['FamilySize'] = test['FamilySize'].astype(int)

def family_range(df):

    df['FamilySize'].loc[df['FamilySize'] <= 1 ] = 0

    df['FamilySize'].loc[(df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)] = 1

    df['FamilySize'].loc[df['FamilySize'] >= 5] = 2   

family_range(train)

family_range(test)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

# Train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')

# Test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
# Train Data

train = pd.get_dummies(train, columns=['Sex','Embarked','Title'],drop_first=True)

# Test Data

test= pd.get_dummies(test, columns=['Sex','Embarked','Title'],drop_first=True)
fig=plt.figure(figsize=(18,10))

ax = fig.gca()

sns.heatmap(train.corr(), annot=True,ax=ax, cmap=plt.cm.YlGnBu)

ax.set_title('The correlations between all numeric features')

palette =sns.diverging_palette(80, 110, n=146)

plt.show()
corr_matrix = train.corr()

corr_matrix["Survived"].sort_values(ascending=False)
_ = sns.countplot(data=train, x='Survived', hue='Pclass')

_.set_title('Number of Survivals by Class')

_.set_xlabel('Survival')

_.set_xticklabels(['Dead', 'Survived'])

_.legend(['1st class', '2nd class', '3rd class']);
print('Survival rate per class: ')

train.groupby('Pclass')['Survived'].sum() / train['Pclass'].value_counts() # Survive percent by class
_ = sns.countplot(data=train, x='Pclass', hue='Sex_male')

_.set_title('Number of Males and Females by Class')

_.set_xticklabels(['1st class', '2nd class', '3rd class'])

_.legend(['Female', 'Male']);
_ = sns.countplot(data=train, x='Survived', hue='FamilySize')

_.set_title('Number of Deaths by Family Size')

_.set_xlabel('Survival')

_.set_xticklabels(['Dead','Survived'])

_.legend(['Single', 'Small (2 to 4)', 'Large (5 and more)'], title='Family Size');
print('Survival rate for people by there Family Size: ')

(train.groupby('FamilySize')['Survived'].sum() / train['FamilySize'].value_counts()).rename({0:'Single', 1:'Small', 2:'Large'})
_ = sns.countplot(data=train, x='Survived', hue='Sex_male')

_.set_title('Number of Survivals and Deaths Per Gender')

_.set_xlabel('Survival')

_.set_xticklabels(['Dead','Survived'])

_.legend(['Female', 'Male']);
print('Survival rate per gender: ')

(train.groupby('Sex_male')['Survived'].sum() / train['Sex_male'].value_counts()).rename(index={0:'female',1:'male'})
_ = sns.countplot(data=train, x='Survived')

_.set_title('Number of Survivals and Deaths')

_.set_xlabel('Survival')

_.set_xticklabels(['Dead','Survived']);
train['Survived'].value_counts()
# majority_count, minority_count = train['Survived'].value_counts()

# majority = train[train['Survived'] == 0]

# minority = train[train['Survived'] == 1]



# minority_overSamp = minority.sample(majority_count, replace=True, random_state=55)

# train_overSamp = pd.concat([majority, minority_overSamp])



# train_overSamp['Survived'].value_counts()
# majority_count, minority_count = train['Survived'].value_counts()

# majority = train[train['Survived'] == 0]

# minority = train[train['Survived'] == 1]



# majority_underSamp = majority.sample(minority_count)

# train_underSamp = pd.concat([majority_underSamp, minority], axis=0)



# train_underSamp['Survived'].value_counts()
# from imblearn.over_sampling import SMOTE

# oversample = SMOTE()

# X, y = oversample.fit_resample(X, y)
# from sklearn.feature_selection import SelectKBest

# from sklearn.feature_selection import chi2

# ksel = SelectKBest(chi2, k=9) 

# ksel.fit(X, y) 

# new_X = ksel.transform(X)

# new_testing = ksel.transform(testing)



# ix = ksel.get_support()

# pd.DataFrame(new_X, columns = X.columns[ix]).head(5)
# Train data

features_drop = ['PassengerId','Name', 'Ticket', 'Survived','SibSp','Parch']
selected_features = [x for x in train.columns if x not in features_drop]
# Test data

features_drop_test = ['PassengerId','Name', 'Ticket','SibSp','Parch']
selected_features_test = [x for x in test.columns if x not in features_drop_test]
# Train data

X = train[selected_features]

y = train['Survived']
# Test data

testing = test[selected_features_test]
ss = StandardScaler()

Xs = ss.fit_transform(X)
testing_s = ss.transform(testing)
X_train, X_test, y_train, y_test = train_test_split(

    Xs, y, test_size = .3, random_state= 42, stratify = y) 
def modeling(model, X_train, y_train, test_data, X_test=None, y_test=None, prefit=False):

    '''Takes model and data then print model results with some metrics then return predictions'''

    

    start = "\033[1m" # to create BOLD print

    end = "\033[0;0m" # to create BOLD print

    

    # Print bold model name 

    model_name = str(model)#.split('(')[0]

    print(''.join(['\n', start, model_name, end]))

    

    #Fit model

    if not prefit:

        model.fit(X_train, y_train)

    

    #Accuarcy score    

    print('Train Score', model.score(X_train, y_train))

    try:

        print('Test Score :', model.score(X_test, y_test))

    

        #confusion matrix

        X_test_pred = model.predict(X_test)

        print('\nconfusion matrix\n', confusion_matrix(y_test, X_test_pred))  

    except: pass

    

    #cross val score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_res = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    print('\nCV scores: ', cv_res)

    print('CV scores mean: ', cv_res.mean())  

    

    #predictions

    y_pred = model.predict(test_data)

    print('\nFirst 10 Predictions: \n', y_pred[:10])





    return y_pred
# Models

logreg = LogisticRegression(max_iter=300)

knn = KNeighborsClassifier(n_neighbors=7)  

svm_lin = svm.SVC(kernel='linear', C=33)

svm_poly = svm.SVC(kernel='poly', C=3)

svm_rbf = svm.SVC(kernel='rbf', C=33)



# randomF = RandomForestClassifier(max_depth=350, n_estimators=9, max_features=11, random_state=14, min_samples_split=3)

randomF = RandomForestClassifier(max_depth=350, random_state=42)

dtree= DecisionTreeClassifier(random_state=42)

extree = ExtraTreesClassifier(n_estimators=66, min_samples_split=7, random_state=42)

xgb_model = xgb.XGBClassifier(colsample_bytree= 0.8, gamma= 1, learning_rate= 0.002,

                              max_depth= 8, min_child_weight= 1,subsample= 0.8,)







models = [(logreg,'logreg'), (knn,'knn'), (svm_lin,'svm_lin'), (svm_poly,'svm_poly'), (svm_rbf,'svm_rbf'),

          (randomF,'randomF'), (dtree,'dtree'), (extree,'extree'), (xgb_model,'xgb_model')]



preds = {}    # empty dict to save all models predictions

for model, name in models:

    preds[name] = modeling(model, X_train, y_train, testing_s, X_test, y_test)
def g_search(model, param, X_train, y_train, test_data, X_test=None, y_test=None):

    '''Simple grid search with kfold'''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(model,

                  param,

                  scoring='accuracy',

                  cv=cv,

                  n_jobs=-1,

                  verbose=0)

    gs.fit(X_train, y_train)

    

    # Results

    y_pred = modeling(gs.best_estimator_, X_train, y_train, test_data, X_test, y_test, prefit=True) # print results and return predictions

    

    print('Best parameters: ', gs.best_params_)

    

    return y_pred
# grid search using all the data (Xs, y)

grid_logreg_pred = g_search(LogisticRegression(), {'C': np.arange(1, 40, 1)}, Xs, y, testing_s)

grid_knn_pred = g_search(KNeighborsClassifier(), {'n_neighbors': np.arange(1, 100, 1)}, Xs, y, testing_s)

grid_svm_lin_pred = g_search(svm.SVC(kernel='linear'), {'C': np.arange(1, 40, 1)}, Xs, y, testing_s)

grid_svm_poly_pred = g_search(svm.SVC(kernel='poly'), {'C': np.arange(1, 40, 1)}, Xs, y, testing_s)

grid_svm_rbf_pred = g_search(svm.SVC(kernel='rbf'), {'C': np.arange(1, 40, 1)}, Xs, y, testing_s)
dt = DecisionTreeClassifier()

dt_en = BaggingClassifier(base_estimator=dt, n_estimators=500, max_features=4, random_state=55)



dt_en_pred = modeling(dt_en, Xs, y, testing_s) 
adaboost = AdaBoostClassifier(n_estimators=67)

adaboost.fit(X_train, y_train)



adaboost_pred = modeling(adaboost, Xs, y, testing_s) 
xgb_model = xgb.XGBClassifier(



    colsample_bytree= 0.8,

    gamma= 1,

    learning_rate= 0.002,

    max_depth= 8,

    min_child_weight= 1,

    subsample= 0.8,

)



xgb_pred = modeling(xgb_model, Xs, y, testing_s)
param_grid = {

                   'n_estimators': np.arange(50,500,20),

                   'max_depth' : [i for i in range(1,15,1)],

#                    'gamma': [1,2,3,4],

#                    'reg_alpha': [0,1,2,3]

}

xgb_model = xgb.XGBClassifier(



    colsample_bytree= 0.8,

    learning_rate= 0.001,

    min_child_weight= 1,

    subsample= 0.8,

)



grid_xgb_pred = g_search(xgb_model, param_grid, Xs, y, testing_s)
thesubmission = gender_submission.copy()

thesubmission['Survived'] = xgb_pred

thesubmission['Survived'].head(10)
thesubmission.to_csv('thesubmission.csv', index=False)
# xgb_model = xgb.XGBClassifier(



#     colsample_bytree= 0.8,

#     gamma= 1,

#     learning_rate= 0.002,

#     max_depth= 8,

#     min_child_weight= 1,

#     subsample= 0.8     

# )



# Train Score 0.8709315375982043



# CV scores:  [0.8547486  0.85393258 0.83707865 0.83707865 0.8258427 ]

# CV scores mean:  0.8417362375243236



# First 10 Predictions: 

#  [0 1 0 0 1 0 1 0 1 0]



# features_drop = ['PassengerId','Name', 'Ticket', 'Survived','SibSp','Parch']



# X_train, X_test, y_train, y_test = train_test_split(

#     Xs, y, test_size=.3,random_state=55, stratify=y) 



# Kaggle score: 0.79186