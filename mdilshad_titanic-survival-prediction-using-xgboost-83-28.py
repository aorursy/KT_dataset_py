import re

import numpy as np

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

%matplotlib inline
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, accuracy_score

from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve

from sklearn.feature_selection import SelectKBest,chi2
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train['train'] = 1

df_test['train']  = 0

data = df_train.append(df_test, ignore_index=True)

data.info()
sns.distplot(data['Age'].dropna())

print('Min = {}, Max= {}'.format(data['Age'].min(), data['Age'].max()))
sns.distplot(data['Fare'].dropna())

print('Min = {}, Max= {}'.format(data['Fare'].min(), data['Fare'].max()))
data['Embarked'].value_counts()
data['Age'].fillna(data['Age'].mean(), inplace=True)

data['Fare'].fillna(data['Fare'].median(), inplace=True)

data['Embarked'].fillna('S', inplace=True)
data.isnull().sum()
# Getting Training data from Full dataset

train = data[data['train']==1]
train['Survived'].value_counts()
# chi-square test to test independance of two categorical varaible.

def chi_test_categorical_feature(alpha, feature, target='Survived'):

    contigency_pclass = pd.crosstab(train[feature], train[target])

    stat, p, dof, expected = chi2_contingency(contigency_pclass)

    if p < alpha:

        print('There is relationship btw {} and target variable with p_value = {} and Chi-squared = {}'.format(feature, p, stat) )

    else:

        print('not good predictor with p_value = {} and Chi-squared = {}'.format( p, stat))
train['Sex'].value_counts()
sns.countplot(train['Sex'], hue=train['Survived'])
# for making contegancy table 

pd.crosstab(train['Sex'], train['Survived'], normalize='all')*100
chi_test_categorical_feature(0.01, 'Sex')
train['Pclass'].value_counts()
sns.countplot(train['Pclass'], hue=train['Survived'])
pd.crosstab(train['Pclass'], train['Survived'], normalize='all')*100
chi_test_categorical_feature(0.01, 'Pclass')
train['SibSp'].value_counts()
sns.countplot(train['SibSp'], hue=train['Survived'])
pd.crosstab(train['SibSp'], train['Survived'], normalize='all')*100
chi_test_categorical_feature(0.01, 'SibSp')
train['Parch'].value_counts()
sns.countplot(train['Parch'], hue=train['Survived'])
pd.crosstab(train['Parch'], train['Survived'], normalize='all')*100
chi_test_categorical_feature(0.01, 'Parch')
train['Embarked'].value_counts()
sns.countplot(train['Embarked'], hue=train['Survived'])
pd.crosstab(train['Embarked'], train['Survived'], normalize='all')*100
chi_test_categorical_feature(0.01, 'Embarked')
sns.boxplot(y=train['Age'], x=train['Survived'])
sns.violinplot(y=train['Age'], hue=train['Survived'], x=[""]*len(train), palette="Set2")
train[['Age', 'Survived']].corr()
sns.boxplot(y=train['Fare'], x=train['Survived'])
train[['Fare', 'Survived']].corr()
sns.violinplot(y=train['Fare'], hue=train['Survived'], x=[""]*len(train), palette="Set2")
train[train['Cabin'].isnull()]['Survived'].value_counts()
train[train['Cabin'].isnull()==False]['Survived'].value_counts()
pd.crosstab([train['Cabin'].isnull()], train['Survived'], normalize='all')*100
# creating featur fromcabin column if cabin exist then 1 else 0

data['Has_Cabin'] = ~data['Cabin'].isnull()

data['Has_Cabin'] = data['Has_Cabin'].astype(int)
train = data[['Has_Cabin', 'Survived']]

sns.countplot(train['Has_Cabin'], hue=train['Survived'])
# Creating feature FamilySize from SibSp and Parch

data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
train = data[['Family_Size', 'Survived']]

sns.countplot(train['Family_Size'], hue=train['Survived'])
data['Is_Alone'] = data['Family_Size'].apply(lambda x: 1 if x==1 else 0)

train = data[['Is_Alone', 'Survived']]

sns.countplot(train['Is_Alone'], hue=train['Survived'])
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""
data['Title'] = data['Name'].apply(get_title)
data['Title'].value_counts()
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'OTHER')
data['Title'] = data['Title'].replace('Mlle', 'Miss')

data['Title'] = data['Title'].replace('Ms', 'Miss')

data['Title'] = data['Title'].replace('Mme', 'Mrs')
train = data[['Title', 'Survived']]

sns.countplot(train['Title'], hue=train['Survived'])
data.info()
data['Age'].max()
bins = np.linspace(0, 80, 6)

data['Age_binned']= pd.cut(data['Age'], bins, labels=[1,2,3,4,5], include_lowest=True)

data['Age_binned'] = data['Age_binned'].astype(int)
train = data[['Age_binned', 'Survived']]

sns.countplot(train['Age_binned'], hue=train['Survived'])
bins = [-1,50,100,390, 520]

data['Fare_binned'] = pd.cut(data['Fare'], bins ,labels=[1,2,3,4], include_lowest=True)

data['Fare_binned'] = data['Fare_binned'].astype(int)
train = data[['Fare_binned', 'Survived']]

sns.countplot(train['Fare_binned'], hue=train['Survived'])
data['Sex'].replace({'male': 1, 'female': 0}, inplace=True)

data['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3}, inplace=True)

data['Title'].replace({'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'OTHER': 5}, inplace=True)
feature = ['Embarked','Pclass', 'Sex', 'SibSp','Parch', 'Has_Cabin', 'Family_Size', 'Is_Alone', 'Title', 'Age_binned', 'Fare_binned']
# converting feature to category so that we perform encoding on them.

data[feature] = data[feature].astype('category')

dummy_data = pd.get_dummies(data[feature])

# join with orginal dataset

data = pd.concat([data, dummy_data], axis=1)
# Separating training and testing data. 

training_data = data[data['train']==1]

testing_data = data[data['train']==0]
# All features

feature_1 = ['Embarked_1', 'Embarked_2', 'Embarked_3', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_0', 'Sex_1',

             'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1',

             'Parch_2','Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Parch_9','Has_Cabin_0', 'Has_Cabin_1', 'Family_Size_1',

             'Family_Size_2', 'Family_Size_3', 'Family_Size_4', 'Family_Size_5','Family_Size_6', 'Family_Size_7',

             'Family_Size_8', 'Family_Size_11','Is_Alone_0', 'Is_Alone_1', 'Title_1','Title_2', 'Title_3', 'Title_4',

             'Title_5','Age_binned_1', 'Age_binned_2', 'Age_binned_3', 'Age_binned_4','Age_binned_5', 'Fare_binned_1',

             'Fare_binned_2', 'Fare_binned_3','Fare_binned_4'] 
feature_set = []

chi2_selector = SelectKBest(chi2, k=30)

chi2_selector.fit_transform(training_data[feature_1], y=training_data['Survived'].astype(int))

for feature, chi_result in zip(feature_1, chi2_selector.get_support()):

    if chi_result==True:

        feature_set.append(feature)
train_accuracy = pd.DataFrame(columns=['Name of Model', 'Accuracy'])
seed = 101
# 1.Decision Tree Classifier

dt = DecisionTreeClassifier(random_state = seed)



# 2.Support Vector Machines

svc = SVC(gamma = 'auto')



# 3.Random Forest Classifier



rf = RandomForestClassifier(random_state = seed, n_estimators = 100)



#4.Gaussian Naive Bayes

gnb = GaussianNB()



#5.Gradient Boosting Classifier

gbc = GradientBoostingClassifier(random_state = seed)



#6.Adaboost Classifier

abc = AdaBoostClassifier(random_state = seed)



#7.ExtraTrees Classifier

etc = ExtraTreesClassifier(random_state = seed)



#10.Extreme Gradient Boosting

xgbc = XGBClassifier(random_state = seed)



clf_list = [dt, svc, rf, gnb, gbc, abc, etc, xgbc]

clf_list_name = ['dt', 'svc', 'rf', 'gnb', 'gbc', 'abc', 'etc', 'xgbc']
def train_accuracy_model(model):

    model.fit(X_train, y_train)

    accuracy = (model.score(X_train, y_train))*100

    return accuracy
# For Feature set 1 which countain all Feature.

X_train = training_data[feature_set]

y_train = training_data['Survived'].astype(int)

for clf, name in zip(clf_list, clf_list_name):

    accuracy = train_accuracy_model(clf)

    r = train_accuracy.shape[0]

    train_accuracy.loc[r] = [name, accuracy]
train_accuracy.sort_values(by='Accuracy', ascending=False)
cross_val_df = pd.DataFrame(columns=['Name of Model', 'Accuracy'])
def cross_val_accuracy(model):

    score = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1).mean()

    score = np.round(score*100, 2)

    return score

    
# For Feature set 1 which countain all Feature.

X_train = training_data[feature_set]

y_train = training_data['Survived'].astype(int)

for clf, name in zip(clf_list, clf_list_name):

    accuracy = cross_val_accuracy(clf)

    r = cross_val_df.shape[0]

    cross_val_df.loc[r] = [name, accuracy]
cross_val_df.sort_values(by='Accuracy', ascending=False)
#Define dataframe for Parameter tuning.

# Accuracy here is mean value of cross validation score of model with best paramters

param_df = pd.DataFrame(columns=['Name of Model', 'Accuracy', 'Parameter'])
# For GBC, the following hyperparameters are usually tunned.

gbc_params = {'learning_rate': [0.01, 0.02, 0.05, 0.01],

              'max_depth': [4, 6, 8],

              'max_features': [1.0, 0.3, 0.1], 

              'min_samples_split': [ 2, 3, 4],

              'random_state':[seed]}



# For SVC, the following hyperparameters are usually tunned.

svc_params = {'C': [6, 7, 8, 9, 10], 

              'kernel': ['linear','rbf'],

              'gamma': [0.5, 0.2, 0.1, 0.001, 0.0001]}



# For DT, the following hyperparameters are usually tunned.

dt_params = {'max_features': ['auto', 'sqrt', 'log2'],

             'min_samples_split': [2, 3, 4, 5, 6, 7, 8], 

             'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8],

             'random_state':[seed]}



# For RF, the following hyperparameters are usually tunned.

rf_params = {'criterion':['gini','entropy'],

             'n_estimators':[10, 15, 20, 25, 30],

             'min_samples_leaf':[1, 2, 3],

             'min_samples_split':[3, 4, 5, 6, 7], 

             'max_features':['sqrt', 'auto', 'log2'],

             'random_state':[44]}





# For ABC, the following hyperparameters are usually tunned.'''

abc_params = {'n_estimators':[1, 5, 10, 50, 100, 200],

              'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3, 1.5],

              'random_state':[seed]}



# For ETC, the following hyperparameters are usually tunned.

etc_params = {'max_depth':[None],

              'max_features':[1, 3, 10],

              'min_samples_split':[2, 3, 10],

              'min_samples_leaf':[1, 3, 10],

              'bootstrap':[False],

              'n_estimators':[100, 300],

              'criterion':["gini"], 

              'random_state':[seed]}



# For XGBC, the following hyperparameters are usually tunned.

xgbc_params = {'n_estimators': (150, 250, 350,450,550,650, 700, 800, 850, 1000),

              'learning_rate': (0.01, 0.6),

              'subsample': (0.3, 0.9),

              'max_depth': [3, 4, 5, 6, 7, 8, 9],

              'colsample_bytree': (0.5, 0.9),

              'min_child_weight': [1, 2, 3, 4],

              'random_state':[seed]}

clf_list = [dt, svc, rf, gbc, abc, etc, xgbc]

clf_list_name = ['dt', 'svc', 'rf', 'gbc', 'abc', 'etc', 'xgbc']

clf_param_list = [dt_params, svc_params, rf_params, gbc_params, abc_params, etc_params, xgbc_params]
# Create a function to tune hyperparameters of the selected models.'''

def tune_hyperparameters(model, params):

    from sklearn.model_selection import GridSearchCV

    # Construct grid search object with 10 fold cross validation.

    grid = GridSearchCV(model, params, verbose = 0, cv = 10, scoring = 'accuracy', n_jobs = -1)

    # Fit using grid search.

    grid.fit(X_train, y_train)

    best_params, best_score = grid.best_params_, np.round(grid.best_score_*100, 2)

    return best_params, best_score
# Tuning Parameters of all Model

X_train = training_data[feature_set]

y_train = training_data['Survived'].astype(int)

for clf, name, params in zip(clf_list, clf_list_name, clf_param_list):

    best_params, best_score = tune_hyperparameters(clf, params)

    r = param_df.shape[0]

    param_df.loc[r] = [name, best_score, best_params]
param_df.sort_values(by='Accuracy', ascending=False)

param_df.to_pickle("./dummy.pkl")
def ploting_learning_curve(model):

    # Create CV training and test scores for various training set sizes

    train_sizes, train_scores, test_scores = learning_curve(model, 

                                                        X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1, 

                                                        # 50 different sizes of the training set

                                                        train_sizes=np.linspace(0.01, 1.0, 50))





    # Create means and standard deviations of training set scores

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)



    # Create means and standard deviations of test set scores

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    # Draw lines

    plt.plot(train_sizes, train_mean, '--',  label="Training score")

    plt.plot(train_sizes, test_mean, label="Cross-validation score")



    # Draw bands

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



    # Create plot

    plt.title("Learning Curve")

    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

    plt.tight_layout()

    plt.show()

#  RandomFrostClassifier Learning curve

model = RandomForestClassifier(criterion='entropy', max_features='log2', min_samples_leaf=1, min_samples_split=7,

                      n_estimators=20, random_state = 44)

ploting_learning_curve(model)
model = XGBClassifier(colsample_bytree= 0.9, learning_rate= 0.01, max_depth= 6, min_child_weight= 2,

                      n_estimators= 1000, random_state= 101, subsample= 0.3)

ploting_learning_curve(model)
model = GradientBoostingClassifier(learning_rate= 0.02, max_depth= 4, max_features= 0.3, min_samples_split= 4,

                                   random_state= 101)

ploting_learning_curve(model)
X_train = training_data[feature_set]

y_train = training_data['Survived'].astype(int)

model = XGBClassifier(colsample_bytree= 0.9, learning_rate= 0.01, max_depth= 6, min_child_weight= 2,

                      n_estimators= 1000, random_state= 101, subsample= 0.3)





model.fit(X_train, y_train)
X_test=testing_data[feature_set]

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)