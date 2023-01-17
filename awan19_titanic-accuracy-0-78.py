import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

#print(os.listdir("../input"))



# To plot the figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()
train.shape
test.head()
test.shape
# Missing Data 

train.info()
train.isnull().sum()
test.isnull().sum()
# Looking at the numerical data

train.describe()
# Survived feature

train['Survived'].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(18, 7))

pieLabels = ['Did not survive', 'Survived']



ax[0].pie(train['Survived'].value_counts(), explode=[0.05, 0.05], autopct='%1.1f%%', labels=pieLabels, shadow=True, 

          startangle=90)

ax[0].axis('equal')

ax[0].set_title('% age - Surivival')



sns.countplot('Survived', data=train, ax=ax[1])

ax[1].set_title('Count - Surivival')



plt.show()
# Pclass feature



train['Pclass'].value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
pd.crosstab(train['Pclass'], train['Survived']).style.background_gradient(cmap='cool')
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

train.groupby(['Pclass','Survived']).size().unstack().plot(kind='bar',stacked=True, ax=ax[0])

train.groupby(['Pclass','Survived']).size().unstack().plot(kind='bar',stacked=False, ax=ax[1])

ax[0].set_title('Pclass vs Survived - Stacked')

ax[1].set_title('Pclass vs Survived - Side by Size')

plt.show()
# Gender aka Sex



train['Sex'].value_counts()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
pd.crosstab(train['Sex'], train['Survived']).style.background_gradient(cmap='cool')
fix, ax = plt.subplots(1, 2, figsize=(15, 5))

train.groupby(['Sex', 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax[0])

train.groupby(['Sex', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax[1])



ax[0].set_title('Sex vs Survived - Stacked')

ax[1].set_title('Sex vs Survived - Side by Size')

plt.show()
# How does Pclass and Sex together compare wrt Survival ?



pd.crosstab([train['Sex'], train['Survived']], train['Pclass'], margins=True).style.background_gradient(cmap='cool')
train.groupby('Pclass').apply(lambda x:x.groupby('Sex')['Survived'].mean()).style.background_gradient(cmap='cool')
sns.catplot('Pclass', 'Survived', hue='Sex', data=train, kind='point')

plt.show()
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().style.background_gradient(cmap='cool')
pd.crosstab(train['SibSp'], train['Survived']).style.background_gradient(cmap='cool')
fig, ax = plt.subplots(2,2, figsize=(20, 15))



train.groupby(['SibSp', 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax[0, 0])

train.groupby(['SibSp', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax[0, 1])

ax[0, 0].set_title('SibSp vs Survived - Count - Stacked')

ax[0, 1].set_title('SibSp vs Survived - Count - Side by Side')



sns.barplot('SibSp','Survived', data=train, ax=ax[1, 0])

ax[1, 0].set_title('SibSp vs Survived - %age')



sns.catplot('SibSp','Survived', data=train, ax=ax[1, 1], kind='point')

ax[1, 1].set_title('SibSp vs Survived - %age')



plt.close(2)

plt.show()
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().style.background_gradient(cmap='cool')
pd.crosstab(train['Parch'], train['Survived']).style.background_gradient(cmap='cool')
f, ax = plt.subplots(2, 2, figsize=(18, 12))

train.groupby(['Parch', 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax[0, 0])

train.groupby(['Parch', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax[0, 1])

ax[0, 0].set_title('Parch vs Survived - Count - Stacked')

ax[0, 1].set_title('Parch vs Survived - Count - Side by Side')

sns.barplot('Parch','Survived', data=train, ax=ax[1, 0])

sns.catplot('Parch','Survived', data=train, ax=ax[1, 1], kind='point')

ax[1, 0].set_title('Parch vs Survived - %age ')

ax[1, 1].set_title('Parch vs Survived - %age')

plt.close(2)

plt.show()
train['Cabin'].nunique()
train['Ticket'].nunique()
train['PassengerId'].nunique()
train['Embarked'].value_counts()
# Since there are only two missing values we can replace them with the most frequent value 'S'

# There's no missing data for the feature 'Embarked' in the test data



train['Embarked'].fillna('S', inplace=True)
# Checking

train['Embarked'].value_counts()
train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().style.background_gradient(cmap='cool')
pd.crosstab(train['Embarked'],train['Survived']).style.background_gradient(cmap='cool')
sns.catplot('Embarked','Survived', data=train, kind='point')

fig=plt.gcf()

fig.set_size_inches(5,3)

plt.show()
pd.crosstab(train['Embarked'],train['Pclass']).style.background_gradient(cmap='cool')
f, ax = plt.subplots(1,2, figsize=(18, 6))

train.groupby(['Embarked', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax[0])

train.groupby(['Embarked', 'Pclass']).size().unstack().plot(kind='bar', stacked=False, ax=ax[1])

ax[0].set_title('Embarked vs Survived')

ax[1].set_title('Embarked vs Pclass')

plt.show()
sns.catplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=train, kind='point')

plt.show()
train['FamilySize'] = train['SibSp'] + train['Parch']

test['FamilySize'] = test['SibSp'] + test['Parch']
train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().style.background_gradient(cmap='cool')
pd.crosstab(train['FamilySize'], train['Survived']).style.background_gradient(cmap='cool')
f, ax = plt.subplots(1, 2, figsize=(18, 5))

sns.barplot('FamilySize', 'Survived', data=train, ax=ax[0])

ax[0].set_title('Family Size vs Survived')

sns.catplot('FamilySize', 'Survived', data=train, ax=ax[1], kind='point')

ax[0].set_title('Family Size vs Survived')

plt.close(2)

plt.show()
from sklearn.preprocessing import LabelEncoder
# Create a fare category 



test['Fare'].fillna(test['Fare'].median(), inplace = True)



train['FareCategory'] = pd.qcut(train['Fare'], 5)

test['FareCategory'] = pd.qcut(test['Fare'], 5)
train['Fare_Code'] = LabelEncoder().fit_transform(train['FareCategory'])
test['Fare_Code'] = LabelEncoder().fit_transform(test['FareCategory'])
train[['Fare_Code', 'Survived']].groupby(['Fare_Code']).mean().style.background_gradient(cmap='cool')
pd.crosstab(train['Fare_Code'], train['Survived']).style.background_gradient(cmap='cool')
f, ax = plt.subplots(1,2, figsize=(18, 6))

train.groupby(['Fare_Code','Survived']).size().unstack().plot(kind='bar', ax=ax[0])

sns.catplot('Fare_Code', 'Survived', data=train, kind='bar', ax=ax[1])

ax[0].set_title('Fare_Code vs Survived')

ax[1].set_title('Fare_Code vs Survived')

plt.close(2)

plt.show()
# Extract portion before the '.'



train['Title'] = 0

for salut in train:

    train['Title'] = train.Name.str.extract('([A-Za-z]+)\.')

    

test['Title'] = 0

for salut in test:

    test['Title'] = test.Name.str.extract('([A-Za-z]+)\.')  
pd.crosstab(train['Title'], train['Sex']).T.style.background_gradient(cmap='cool')
# Replacing the titles 

mapping = {'Mlle': 'Miss', 

           'Major': 'Mr', 

           'Col': 'Mr', 

           'Sir': 'Mr', 

           'Don': 'Mr',

           'Mme': 'Miss',

           'Jonkheer': 'Mr',

           'Lady': 'Mrs', 

           'Capt': 'Mr', 

           'Countess': 'Mrs', 

           'Ms': 'Miss', 

           'Dona': 'Mrs'

           }

train.replace({'Title': mapping}, inplace=True)

test.replace({'Title': mapping}, inplace=True)
data_df = train.append(test)



titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:

    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]

    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute    
# Substituting Age values in TRAIN_DF and TEST_DF:

train['Age'] = data_df['Age'][:891]

test['Age'] = data_df['Age'][891:]
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().style.background_gradient(cmap='cool')
plt.style.use('fivethirtyeight')

sns.catplot('Pclass','Survived', col='Title', data=train, kind='point')

plt.show()
plt.style.use('default')
train['AgeCategory'] = pd.qcut(train['Age'], 4)

test['AgeCategory'] = pd.qcut(test['Age'], 4)
train['Age_Code'] = LabelEncoder().fit_transform(train['AgeCategory'])

test['Age_Code'] = LabelEncoder().fit_transform(test['AgeCategory'])
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])



DEFAULT_SURVIVAL_VALUE = 0.5

data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0



print("Number of passengers with family survival information:", 

      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])
for _, grp_df in data_df.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

                        

print("Number of passenger with family/group survival information: " 

      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))



# # Family_Survival in TRAIN_DF and TEST_DF:

train['Family_Survival'] = data_df['Family_Survival'][:891]

test['Family_Survival'] = data_df['Family_Survival'][891:]
train['Sex'].replace(['male','female'],[0,1],inplace=True)

test['Sex'].replace(['male','female'],[0,1],inplace=True)
drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch','Ticket', 'Cabin', 'FareCategory', 'AgeCategory','Age', 'Fare', 'Title', 'Embarked']

train = train.drop(drop_elements, axis=1)

test = test.drop(drop_elements, axis=1)
# Copying the Survived column data 

y = train['Survived']
X = train[train.columns[1:]]

# Data is now clean and can be used in the models
train.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
all_features = ['Pclass', 'Sex', 'FamilySize', 'Fare_Code', 'Age_Code', 'Family_Survival']
all_transformer = Pipeline(steps = [

    ('stdscaler', StandardScaler())

])
all_preprocess = ColumnTransformer(

    transformers = [

        ('allfeatures', all_transformer, all_features),

    ]

)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=train['Survived'])
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier



from sklearn.model_selection import cross_val_score
classifiers = [

    LogisticRegression(random_state=42),

    RandomForestClassifier(random_state=42),

    SVC(random_state=42),

    KNeighborsClassifier(),

    SGDClassifier(random_state=42),

    ]
first_round_scores = {}

for classifier in classifiers:

    pipe = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', classifier)])

    pipe.fit(X_train, y_train)   

    print(classifier)

    score = pipe.score(X_test, y_test)

    first_round_scores[classifier.__class__.__name__[:10]] = score

    print("model score: %.3f" % score)
# Plot the model scores

plt.plot(first_round_scores.keys(), first_round_scores.values(), "ro", markersize=10)

fig=plt.gcf()

fig.set_size_inches(8,5)

plt.title('Model Scores of the Classifiers - with no tuning ')

plt.show()
grid_scores = {}

log_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', LogisticRegression(random_state=42))])



log_param_grid = {

    'classifier__C': [0.01, 0.1, 1.0, 10],

    'classifier__solver' : ['liblinear','lbfgs','sag', 'saga'],

    'classifier__max_iter' : [500],

}



log_grid_search = GridSearchCV(log_clf, log_param_grid, cv=10, iid=True)

log_grid_search.fit(X_train, y_train)



print('Logistic Regression - grid_search.best_params_ and best_scores_', log_grid_search.best_params_, log_grid_search.best_score_)

log_model_score = log_grid_search.score(X_test, y_test)

print("Logistic Regression - model score: ", log_model_score)

grid_scores['log'] = log_model_score
rf_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', RandomForestClassifier(random_state=42))])



rf_param_grid = {

    'classifier__n_estimators' : [50, 100],

    'classifier__max_features' : [2, 3],

    'classifier__criterion' : ['gini', 'entropy']

}



rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=10, iid=True)

rf_grid_search.fit(X_train, y_train)



print('Random Forest grid_search.best_params_ and best_scores_', rf_grid_search.best_params_, rf_grid_search.best_score_)

rf_model_score = rf_grid_search.score(X_test, y_test)

print("Random Forest - model score: ", rf_model_score)

grid_scores['rf'] = rf_model_score
svm_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', SVC(random_state=42))])

svm_param_grid = [

    {'classifier__kernel': ['linear'], 'classifier__C': [10., 30., 100., 300.]},

    {'classifier__kernel': ['rbf'], 'classifier__C': [1.0, 3.0, 10., 30., 100., 300.],

     'classifier__gamma': [0.01, 0.03, 0.1, 0.3, 1.0]},

    ] 



svm_grid_search = GridSearchCV(svm_clf, svm_param_grid, cv=10, iid=True)

svm_grid_search.fit(X_train, y_train)



print('SVM grid_search.best_params_ and best_scores_', svm_grid_search.best_params_, svm_grid_search.best_score_)

svm_model_score = svm_grid_search.score(X_test, y_test)

print("SVM - model score: ", svm_model_score)

grid_scores['svm'] = svm_model_score
knn_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', KNeighborsClassifier())])

knn_param_grid = {

    'classifier__n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26 ],

    'classifier__weights': ['uniform', 'distance' ],

    'classifier__leaf_size': list(range(1,50,5)),

}



knn_grid_search = GridSearchCV(knn_clf, knn_param_grid, cv=10, iid=True, scoring ='roc_auc')

knn_grid_search.fit(X_train, y_train)



print('KNN grid_search.best_params_ and best_scores_', knn_grid_search.best_params_, knn_grid_search.best_score_)

knn_model_score = knn_grid_search.score(X_test, y_test)

print("KNN - model score: ", knn_model_score)

grid_scores['knn'] = knn_model_score
sgd_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', SGDClassifier(random_state=42))])



sgd_param_grid = {

    'classifier__max_iter': [100, 200],

    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],

}



sgd_grid_search = GridSearchCV(sgd_clf, sgd_param_grid, cv=10, iid=True)

sgd_grid_search.fit(X_train, y_train)



print('SGD grid_search.best_params_ and best_scores_', sgd_grid_search.best_params_, sgd_grid_search.best_score_)

sgd_model_score = sgd_grid_search.score(X_test, y_test)

print("SGD - model score: ", sgd_model_score)

grid_scores['sgd'] = sgd_model_score
plt.plot(grid_scores.keys(), grid_scores.values(), "ro", markersize=10)

fig=plt.gcf()

fig.set_size_inches(8,5)

plt.title('Model Scores of the Classifiers after hyperparameter tuning')

plt.show()
final_pipe = Pipeline(steps=[('preprocessor', all_preprocess)])
X_final_processed = final_pipe.fit_transform(X)
test_final_processed = final_pipe.transform(test)
knn_hyperparameters = {

    'n_neighbors': [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22],

    'algorithm' : ['auto'],

    'weights': ['uniform', 'distance'],

    'leaf_size': list(range(1,50,5)),

}



gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = knn_hyperparameters,  

                cv=10, scoring = "roc_auc")



gd.fit(X_final_processed, y)

print(gd.best_score_)

print(gd.best_estimator_)
gd.best_estimator_.fit(X_final_processed, y)

y_pred = gd.best_estimator_.predict(test_final_processed)
# Tested with different values for n_neighbors and n_jobs and came to this KNN classifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 

                           metric_params=None, n_jobs=None, n_neighbors=6, p=2, 

                           weights='uniform')

knn.fit(X_final_processed, y)

y_pred = knn.predict(test_final_processed)
submission = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv("submission_AWAN_04.csv.csv", index = False)