import numpy as np

import pandas as pd

import re as re

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



from scipy.stats import norm
train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
#missing train data



total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)


#missing test data



total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# for debug purpose

def die(error_message):

    raise Exception(error_message)





# cabin transform

def cabin_value(cabin):

    if str(cabin) == 'nan':

        return str('0')

    else:

        return ord(cabin[0]) - ord('A') + 1





# transform gender to numeric [male, female, child]

def sex_value(sex, age):

    if age < 17:

        return 2

    elif sex == 'male':

        return 0

    else:

        return 1





# Port

def embark_value(em):

    if em == 'C':

        return 3

    elif em == 'Q':

        return 2

    elif em == 'S':

        return 1

    else:

        return 0





# Could of family member siblings + parents

def family_size(sib, par):

    return sib + par + 1





# Has any family member on board

def is_alone(family_size):

    if family_size > 1:

        return 1

    else:

        return 0





# Reverse p_class

def p_class(p_class):

    return (1 - p_class) + 3





# fill nan Age by random and make groups

age_avg = train['Age'].mean()

age_std = train['Age'].std()

age_null_count = train['Age'].isnull().sum()

age_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)





def age_groups(age):

    if(np.isnan(age)):

        age = np.random.choice(age_random_list, 1)

        age = age[0]

    age = int(age)

    if(age <= 8):

        return 6

    if(age <= 18):

        return 5

    if(age <= 25):

        return 4

    elif(age <= 32):

        return 3

    elif(age <= 48):

        return 2

    elif(age <= 64):

        return 1

    else:

        return 0





# fill nan Fare by random and make groups

def fare_groups(fare):

    if(np.isnan(fare)):

        fare = 0

    fare = int(fare)

    if(fare <= 8):

        return 0

    if(fare <= 14):

        return 1

    if(fare <= 31):

        return 2

    elif(fare <= 51):

        return 4

    else:

        return 5





# get title from name and remap to have less dencity



title_map = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Dr": "Officer",

    "Rev": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir": "Royalty",

    "the Countess": "Royalty",

    "Dona": "Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr": "Mr",

    "Mrs": "Mrs",

    "Miss": "Miss",

    "Master": "Master",

    "Lady": "Royalty"

}

title_string_map = {

    'Mrs': 4,

    'Miss': 3,

    'Master': 2,

    'Mr': 1

}





def title_value(df):

    df['Title'] = df.apply(lambda row: row['Name'].split(',')[1].split('.')[0].strip(), axis=1).map(title_map)

    df['Title'] = df['Title'].map(title_map)

    df['Title'] = df['Title'].map(title_string_map)

    df['Title'] = df["Title"].fillna(0)



    return df





# features function

def titan_feature(model):

    df = model.copy()



    df['Embarked'] = df["Embarked"].fillna("S")



    df = title_value(df)



    df['AgeGroup'] = df['Age'].apply(lambda x: age_groups(x))

    df['FareGroup'] = df['Fare'].apply(lambda x: fare_groups(x))

    

    df['Pclass'] = df['Pclass'].apply(lambda x: p_class(x))

    df['Person'] = df.apply(lambda row: sex_value(row['Sex'], row['Age']), axis=1)

    df['Pclass-Person'] = df.apply(lambda row: row['Pclass'] * row['Person'], axis=1)

    

    

    df['Cabin'] = df['Cabin'].apply(lambda x: cabin_value(x))

    df['Embarked'] = df['Embarked'].apply(lambda x: embark_value(x))

    df['FamilySize'] = df.apply(lambda row: family_size(row['SibSp'], row['Parch']), axis=1)

    df['IsAlone'] = df.apply(lambda row: is_alone(row['FamilySize']), axis=1)

    return df





train_features = titan_feature(train)

train_features = train_features[['Survived', 'Pclass-Person', 'Pclass', 'AgeGroup', 'FareGroup', 'Person', 'IsAlone', 'Embarked', 'Title']]



test_features = titan_feature(test)

test_features = test_features[['Pclass', 'AgeGroup', 'FareGroup', 'Person', 'IsAlone', 'Embarked', 'Title']]
#correlation matrix

corrmat = train_features.corr()

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Survived')['Survived'].index

cm = np.corrcoef(train_features[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_features.head(10)
# TRAINING

classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=100),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()

]



classifiers_score = {}

best_acc = 0

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



# define training and testing sets

train_data_all = train_features.drop("Survived", axis=1)

train_result_all = train_features["Survived"]

for train_data_index, train_result_index in sss.split(train_data_all, train_result_all):

    train_data_splitted, test_data_splitted = train_data_all.iloc[train_data_index], train_data_all.iloc[train_result_index]

    train_results_splitted, test_results_splitted = train_result_all.iloc[train_data_index], train_result_all.iloc[train_result_index]



    for classifier in classifiers:

            name = classifier.__class__.__name__

            classifier.fit(train_data_splitted, train_results_splitted)

            train_predictions = classifier.predict(test_data_splitted)

            acc = accuracy_score(test_results_splitted.values, train_predictions) * 100

            if name in classifiers_score:

                classifiers_score[name]['score'] += acc

            else:

                classifiers_score[name] = {}

                classifiers_score[name]['score'] = acc

                classifiers_score[name]['classifier'] = classifier





models_score = pd.DataFrame(columns=('Model', 'Score'))

best_acc = 0



for classifier in classifiers_score:

    acc = classifiers_score[classifier]['score']/10

    models_score.loc[classifier] = [classifier, acc]

    if(best_acc < acc):

        best_acc = acc

        best_classifier = classifiers_score[classifier]['classifier']



print(models_score.sort_values(by='Score', ascending=False))
X_train = train_features.drop("Survived",axis=1)

Y_train = train_features["Survived"]

X_test  = test_features.copy()



best_classifier.fit(X_train, Y_train)

results = best_classifier.predict(X_test)



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": results

    })



submission.to_csv('results.csv', index=False)

print(submission)