import numpy as np 

import pandas as pd

import pandas_profiling



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer



from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score



from statistics import mean 



# Data Viz

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_train.profile_report(style={'full_width': True})
# Embarked is missing 2 data entries so let's fill those with the Mode value 

df_embarked = df_train['Embarked'].copy()

df_embarked.fillna(df_embarked.mode(), inplace=True)

df_train['EmbarkedFilled'] = df_embarked

sns.barplot(x="EmbarkedFilled", y="Survived", hue="Sex", data=df_train)
sns.barplot(x='Pclass', y='Survived', hue="Sex", data=df_train)
# Age is interesting so let's do some light transformation and look at the Age column via age groups

df_age = df_train['Age'].copy()



# Age column is missing some data approximately 20% so let's do some pre-processing and fill in the missing data with the median age.

median_age = df_age[pd.notna(df_age)].median()

df_age.fillna(median_age, inplace=True)



# Next let's decide a real life grouping for the ages:

# baby- 0-1; toddler- 1-3; preschool- 3-5; gradeschooler- 5-12; teen- 12-18; young adult- 18-21; adult- 21-54; senior- 55+ 

bins = (-1, 1, 3, 5, 12, 18, 21, 54, 100)

age_groups = pd.cut(df_age, bins=bins, labels=['baby', 'todd', 'pre', 'grade', 'teen', 'ya', 'adult', 'senior'])

# Add it to the original df so we can graph

df_train['AgeGroup'] = age_groups



# Graph the age grouping

sns.barplot(x='AgeGroup', y='Survived', hue='Sex', data=df_train)
# Given there are 248 unique fare amounts paid that may be too disperse and noisy let's group them into bins based on the quantiles.

# I was trying to think of ways to naturally group them similar to how we did with ages but there's no obvious way. For example if there

# were price tiers that we can derive from the fare amounts that would've helped but we don't have any indication of a tier. I mean who

# was setting the pricing on this trip sheeesh. 



df_fare = df_train['Fare'].copy()

bins = (-1, 8, 14, 31, 600)

df_train['FareGroup'] = pd.cut(df_fare, bins=bins)

sns.barplot(x='FareGroup', y='Survived', hue='Sex', data=df_train)

sns.pointplot(x='Parch', y='Survived', hue='Sex', data=df_train)
sns.pointplot(x='SibSp', y='Survived', hue='Sex', data=df_train)
def fillAges(df):

    '''Fill in missing data for Age column by using the median'''

    df_age = pd.DataFrame(df['Age'])

    imputer = SimpleImputer(strategy='median')

    df['Age'] = imputer.fit_transform(df_age)

    return df

    

def fillEmbarked(df):

    '''Fill in missing data for the Embarked column by using the mode'''

    df_embarked = pd.DataFrame(df['Embarked'])

    imputer = SimpleImputer(strategy='most_frequent')

    df['Embarked'] = imputer.fit_transform(df_embarked)

    return df

    

def addAgeGroup(df):

    '''Group ages in bins and create a new series for it in the dataframe'''

    # baby- 0-1; toddler- 1-3; preschool- 3-5; gradeschooler- 5-12; teen- 12-18; young adult- 18-21; adult- 21-54; senior- 55+ 

    bins = (-1, 1, 3, 5, 12, 18, 21, 54, 100)

    age_group = pd.cut(df['Age'], bins=bins, labels=['baby', 'todd', 'pre', 'grade', 'teen', 'ya', 'adult', 'senior'])

    df['AgeGroup'] = age_group

    return df



def addFareGroup(df):

    '''Group fare in bins and create a new series it in the dataframe'''

    bins = (-1, 8, 14, 31, 600)

    df['FareGroup'] = pd.cut(df_fare, bins=bins, labels=['1','2','3','4'])

    return df



def addTitle(df):

    '''Add title column based on the Name column'''

    name = df['Name'].copy()

    df['Title'] = name.apply(lambda x: x.split(', ')[1].split(' ')[0])

    return df



def dropUnused(df, cols_to_keep):

    '''Drop the unused columns'''

    return df[cols_to_keep]



def clean(df, cols_to_keep):

    '''Clean the data and include the columns we dont want to drop in the end'''

    df = fillAges(df)

    df = fillEmbarked(df)

    df = addAgeGroup(df)

    df = addFareGroup(df)

    df = addTitle(df)

    df = dropUnused(df, cols_to_keep)

    return df



df_train = clean(df_train, ['AgeGroup', 'Title', 'FareGroup', 'Embarked', 'Parch', 'SibSp', 'Sex', 'Pclass', 'Survived'])

df_test = clean(df_test, ['AgeGroup', 'Title', 'FareGroup', 'Embarked', 'Parch', 'SibSp', 'Sex', 'Pclass', 'PassengerId']) # We need to keep the PassengerId for submission
def encode(df_train, df_test, features):

    '''Encode the categorical columns'''

    

    # combine so that we can fit over the training and test data to ensure best results for transform

    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:

        lbl_encoder = LabelEncoder()

        lbl_encoder = lbl_encoder.fit(df_combined[feature])

        df_train[feature] = lbl_encoder.transform(df_train[feature])

        df_test[feature] = lbl_encoder.transform(df_test[feature])

        

    return (df_train, df_test)





categorical_features = ['Embarked', 'Sex', 'Title', 'AgeGroup', 'FareGroup']

df_train, df_test = encode(df_train, df_test, categorical_features)
df_train.profile_report(style={'full_width': True})
X_train_all = df_train.drop(['Survived'], axis=1)

y_train_all = df_train['Survived']



test_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, test_size=test_size, random_state=1)
def simple_predict(X_train, y_train, X_test, y_test, pred_models):

    models = []

    score_results = []



    for m in prediction_models:

        model = eval(m)()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        models.append(m)

        score_results.append(accuracy_score(y_test, y_pred))

        

    return pd.DataFrame({'model': models, 'result': score_results})



predict(X_train, y_train, X_test, y_test, ['RandomForestClassifier', 'SGDClassifier', 'GradientBoostingClassifier'])



def cross_validation(X_all, y_all, pred_models):

    '''Cross validate across all the different classifiers. By default use 4 folds'''

    models = []

    result_1 = []

    result_2 = []

    result_3 = []

    result_4 = []

    means = []

    

    for m in pred_models:

        model = eval(m)()

        results = cross_val_score(model, X_all, y_all, cv=4, scoring='accuracy')

        models.append(m)

        result_1.append(results[0])

        result_2.append(results[1])

        result_3.append(results[2])

        result_4.append(results[3])

        means.append(mean(results))

        

    return pd.DataFrame({'model': models, 

                         'result1': result_1,

                         'result2': result_2,

                         'result3': result_3,

                         'result4': result_4, 

                         'mean': means})



cross_validation(X_train_all, y_train_all, ['RandomForestClassifier', 'SGDClassifier', 'GradientBoostingClassifier'])

        

# Separate the passenger Ids for submission

pass_id_series = df_test['PassengerId']

df_test = df_test.drop(['PassengerId'], axis=1)



# Predict

rfc = RandomForestClassifier()

rfc.fit(X_train_all, y_train_all)

predictions = rfc.predict(df_test)



# Setup final data frame for csv output

df_output = pd.DataFrame({'PassengerId': pass_id_series, 'Survived': predictions})

print(df_output.head())
# output to csv

df_output.to_csv('submission.csv', index=False)