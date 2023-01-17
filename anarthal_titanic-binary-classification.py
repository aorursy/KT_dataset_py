import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score, StratifiedKFold, cross_val_predict

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, RFE

import xgboost as xgb

import pylab

import re



sns.set()

pylab.rcParams['figure.figsize'] = (15.0, 8.0)
df = pd.read_csv('/kaggle/input/titanic/train.csv')

dfresult = pd.read_csv('/kaggle/input/titanic/test.csv')

initial_df = df.copy()

initial_dfresult = dfresult.copy()

dfall = pd.concat([df, dfresult.assign(Survived=np.nan)], sort=True).set_index('PassengerId')



title_pattern = re.compile(r'.* ([a-zA-Z]{2,})\. .*')

title_classes = {

    'mister': ['Don', 'Mr', 'Rev', 'Dr', 'Capt', 'Col', 'Major', 'Sir', 'Jonkheer', 'Sir'],

    'miss': ['Mlle', 'Ms', 'Miss'],

    'mistress': ['Countess', 'Lady', 'Mme', 'Mrs', 'Dona'],

    'master': ['Master']

}



def get_title(name):

    return re.search(title_pattern, name)[1]



def classify_title(title):

    for key, values in title_classes.items():

        if title in values:

            return key

    return None

    

def classify_family_size(num):

    if num == 1:

        return 'alone'

    elif num < 5:

        return 'small'

    else:

        return 'big'

    

def group_cabin(x):

    if x in 'ABC':

        return 'ABC'

    elif x in 'DE':

        return 'DE'

    elif x in 'FG':

        return 'FG'

    else:

        return 'N'





# NaN inputation

dfall['Embarked'].fillna(dfall['Embarked'].mode()[0], inplace=True)

dfall['Fare'].fillna(dfall['Fare'].median(), inplace=True)



# Derive the Title feature

dfall['Title'] = dfall['Name'].map(get_title)



# NaN imputation for age

master_ages = dfall[dfall.Title == 'Master']

dfall.loc[dfall.Title == 'Master', 'Age'] = master_ages.fillna(master_ages.median())

dfall['Age'] = dfall.groupby('Pclass')['Age'].apply(lambda gp: gp.fillna(gp.median()))



# Feature creation

dfall['TitleGrouped'] = dfall['Title'].map(classify_title)

dfall['FamilySize'] = (dfall['SibSp'] + dfall['Parch'] + 1).map(classify_family_size)

dfall['IsChild'] = (dfall['Age'] < 14.0).astype('int')

dfall['TicketCount'] = dfall.groupby('Ticket')['Ticket'].transform('count')

dfall['Cabin'] = dfall['Cabin'].fillna('N').map(lambda x: x.split(' ')[0][0]).map(group_cabin)

dfall['Pclass'] = dfall['Pclass'].map(str)

dfall['Surname'] = dfall['Name'].map(lambda x: x.split(',')[0])

avg_survival = df['Survived'].mean()

dfall['HasFamilySurvivalRate'] = dfall.groupby('Surname')['Survived'].transform(lambda x: (x.size > 1) & ~x.isna().all()).astype('int')

dfall.loc[dfall['FamilySize'] == 'alone', 'HasFamilySurvivalRate'] = 0

test_families = dfall.loc[dfresult['PassengerId'], 'Surname'].unique()

dfall.loc[~dfall['Surname'].isin(test_families), 'HasFamilySurvivalRate'] = 0

dfall['FamilySurvivalRate'] = dfall.groupby('Surname')['Survived'].transform('mean')

dfall.loc[~dfall['HasFamilySurvivalRate'].astype('bool'), 'FamilySurvivalRate'] = avg_survival





# categorical encoding

cat_vars = ['Sex', 'Embarked', 'FamilySize', 'Cabin', 'TitleGrouped', 'Pclass']

dummies = pd.get_dummies(dfall[cat_vars])

dfall[dummies.columns] = dummies

dfall.drop(columns=cat_vars, inplace=True)



# Binning

for name, percentile in [('Age', 13), ('Fare', 10)]:

    dfall[name] = LabelEncoder().fit_transform(pd.qcut(dfall[name], percentile, duplicates='drop'))



# Feature deletion

dfall.drop(columns=['Ticket', 'Title', 'SibSp', 'Parch', 'IsChild', 'Name', 'Surname', 'Survived', 'Sex_female', 'Embarked_Q',

                    'TitleGrouped_mister'], inplace=True)



y = df.set_index('PassengerId')['Survived'].copy()

df = dfall.loc[df.PassengerId, :].copy()

dfresult = dfall.loc[dfresult.PassengerId, :].copy()
corr = df.assign(Survived=y).corr()

idx = corr['Survived'].abs().sort_values(ascending=False).index

sns.heatmap(corr.loc[idx, idx], cmap=plt.cm.BrBG, annot=True)
def split(X, y):

    return train_test_split(X, y, test_size=0.2, random_state=0)#, stratify=y)



def print_scores(X_train, X_test, y_train, y_test, model):

    print('TRAIN: {:.4f}'.format(model.score(X_train, y_train)))

    print('TEST : {:.4f}'.format(model.score(X_test, y_test)))

    

def print_cv_scores(X, y, model, name='Model', cv=None):

    if cv is None:

        cv=StratifiedKFold(5)

    score = cross_val_score(model, X, y, cv=cv)

    print_score_range(score, name)

    

    

def print_score_range(score, name='Model'):

    print('{} has {:.2f}% +- {:.2f}% accuracy'.format(name, 100*np.mean(score), 100*np.std(score) * 2))



def write_output(X_output, model):

    pred = model.predict(X_output)

    dfoutput = pd.DataFrame({'PassengerId': X_output.index, 'Survived': pred})

    dfoutput.to_csv('output.csv', index=False)

    

def plot_learning_curves(X, y, model):

    train_sizes, train_scores, cv_scores = learning_curve(model, X, y)

    mean_train_scores = np.mean(train_scores, axis=1)

    mean_cv_scores = np.mean(cv_scores, axis=1)

    std_train_scores = np.std(train_scores, axis=1)

    std_cv_scores = np.std(cv_scores, axis=1)

    plt.figure(figsize=(10,10))

    plt.plot(train_sizes, mean_train_scores, label='Train', color='r')

    plt.plot(train_sizes, mean_cv_scores, label='CV', color='b')

    plt.fill_between(train_sizes, mean_train_scores + 2*std_train_scores, mean_train_scores - 2*std_train_scores, color='r', alpha=0.2)

    plt.fill_between(train_sizes, mean_cv_scores + 2*std_cv_scores, mean_cv_scores - 2*std_cv_scores, color='b', alpha=0.2)

    plt.xlabel('Sample size')

    plt.ylabel('Accuracy')

    plt.legend()
X_train_all, X_test, y_train_all, y_test = split(df, y)

X_train, X_cv, y_train, y_cv = split(X_train_all, y_train_all)



model = RandomForestClassifier(criterion='gini',

                                           n_estimators=1750,

                                           max_depth=7,

                                           min_samples_split=6,

                                           min_samples_leaf=6,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=42,

                                           n_jobs=-1)

print_cv_scores(X_train_all, y_train_all, model, 'Ensemble')
model.fit(X_train_all, y_train_all)

print_scores(X_train_all, X_test, y_train_all, y_test, model)
# Train on entire training set and predict output

model.fit(df, y)

write_output(dfresult, model)