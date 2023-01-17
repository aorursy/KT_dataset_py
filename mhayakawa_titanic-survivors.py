# Import packages

import numpy as np  

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



print("packages imported")
# Load the training data

train_data_raw = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data_raw.head(5)
# Data types and missing numbers

train_data_raw.info()
# Pclass and survival rate

pd.DataFrame(train_data_raw.groupby(['Pclass'])['Survived'].mean())
# Get title from Name column

def get_title(name_col):

    title = name_col.str.extract(r'^[\w\s\'-]+,\s+([\w\s]+)\.\s+.*$')[0]

    title = title.replace({'Mlle': 'Miss', 'Mme': 'Mrs'})

    return title.apply(lambda nt: nt if nt in ['Mr', 'Mrs', 'Miss'] else 'Other')



# Name title and survival rate

title_survived = pd.DataFrame(train_data_raw['Survived'])

title_survived['Title'] = get_title(train_data_raw['Name'])

pd.DataFrame(title_survived.groupby(['Title'])['Survived'].mean())
# The survival rate of each gender

pd.DataFrame(train_data_raw.groupby(['Sex'])['Survived'].mean())
# The number of survivors of each age

sns.distplot(train_data_raw[train_data_raw['Survived'] == 1]['Age'].dropna())

sns.distplot(train_data_raw[train_data_raw['Survived'] == 0]['Age'].dropna())
# Categorize the Age into 4 groups

def categorize_age(age_col):

    return pd.cut(age_col, [0, 5, 65, float('inf')], labels=['child', 'adult', 'old'])



# Age groups and survival rate

age_survived = pd.DataFrame(train_data_raw['Survived'])

age_survived['Age'] = categorize_age(train_data_raw['Age'])

pd.DataFrame(age_survived.groupby(['Age'])['Survived'].mean())
# SibSp and survival rate

sibsp_survived = pd.DataFrame(train_data_raw['Survived'])

age_survived['SibSp'] = train_data_raw['SibSp'].apply(lambda f: 1 if f > 0 else f)

pd.DataFrame(age_survived.groupby(['SibSp'])['Survived'].mean())
# Family number and survival rate

def categorize_family(sibsp_col, parch_col):

    return (sibsp_col + parch_col).apply(lambda s: str(s) if s < 4 else 'more')



family_survived = pd.DataFrame(train_data_raw['Survived'])

family_survived['Family'] = categorize_family(train_data_raw['SibSp'], train_data_raw['Parch'])

pd.DataFrame(family_survived.groupby(['Family'])['Survived'].mean())
# Split ticket info into prefix and number section

def split_ticket(ticket_col):

    extracted_data = ticket_col.str.extract(r'^((.+)\s)?(\d+)$|^(LINE)$')[[1, 2, 3]]

    extracted_data.columns = ['prefix', 'number', 'isLine']

    return extracted_data

ticket_split = split_ticket(train_data_raw['Ticket'])

ticket_split.head(5)
# ticket prefix existence and survival rate

ticketpre_survived = pd.DataFrame(train_data_raw['Survived'])

ticketpre_survived['TicketPre'] = ticket_split['prefix'].notnull().astype(int)

pd.DataFrame(ticketpre_survived.groupby(['TicketPre'])['Survived'].mean())
# ticket number and survival rate

ticketnum_survived = pd.DataFrame(train_data_raw['Survived'])

ticketnum_survived['TicketNum'] = np.log10(pd.to_numeric(ticket_split['number'], errors='coerce'))

sns.distplot(ticketnum_survived[ticketnum_survived['Survived'] == 1]['TicketNum'].dropna())

sns.distplot(ticketnum_survived[ticketnum_survived['Survived'] == 0]['TicketNum'].dropna())
# LINE ticket and survival rate

ticketline_survived = pd.DataFrame(train_data_raw['Survived'])

ticketline_survived['TicketLine'] = ticket_split['isLine'].notnull().astype(int)

pd.DataFrame(ticketline_survived.groupby(['TicketLine'])['Survived'].mean())
# Log Fare amount and survival rate

sns.distplot(np.log10(train_data_raw[train_data_raw['Survived'] == 1]['Fare'] + 1))

sns.distplot(np.log10(train_data_raw[train_data_raw['Survived'] == 0]['Fare'] + 1))
# Free Fare passengers and survival rate

freefare_survived = pd.DataFrame(train_data_raw['Survived'])

freefare_survived['FreeFare'] = (train_data_raw['Fare'] == 0.).astype(int)

pd.DataFrame(freefare_survived.groupby(['FreeFare'])['Survived'].mean())
# categorize fare

def categorize_fare(fare_col):

    return pd.cut(fare_col, [0,7,float('inf')], labels=['low', 'high'])



fare_survived = pd.DataFrame(train_data_raw['Survived'])

fare_survived['Fare'] = categorize_fare(train_data_raw['Fare'])

pd.DataFrame(fare_survived.groupby(['Fare'])['Survived'].mean())

# Having Cabin number and survival rate

cabin_survived = pd.DataFrame(train_data_raw['Survived'])

cabin_survived['Cabin'] = train_data_raw['Cabin'].notnull().astype(int)

pd.DataFrame(cabin_survived.groupby(['Cabin'])['Survived'].mean())
# Cabin class and survival rate

cabin_survived['CabinClass'] = train_data_raw['Cabin'].str.extract(r'^([A-Z])').fillna('Unknown')

pd.DataFrame(cabin_survived.groupby(['CabinClass'])['Survived'].mean())
# Embarked point and survival rate

pd.DataFrame(train_data_raw.groupby(['Embarked'])['Survived'].mean())
# Extract columns



def extract_columns(df):

    x = pd.DataFrame()

    

    # Pclass

    x['Pclass'] = df['Pclass'].astype(str)

    

    # Title

    x['Title'] = get_title(df['Name'])

    

    # Sex

    x['Sex'] = df['Sex']

    

    # Age

    x['Age'] = categorize_age(df['Age'])

    

    # Family

    x['Family'] = categorize_family(df['SibSp'], df['Parch'])

    

    # Ticket number

    tikect_split = split_ticket(df['Ticket'])

    x['TicketNum'] = np.log10(pd.to_numeric(ticket_split['number'], errors='coerce'))

    

    # Ticket is LINE

    x['TicketLine'] = ticket_split['isLine'].notnull().astype(int)

    

    # Fare

    x['Fare'] = np.log10(df['Fare'] + 1)

    x['FreeFare'] = (train_data_raw['Fare'] == 0.).astype(int)

    

    # Cabin

    x['Cabin'] = df['Cabin'].notnull().astype(int)

    

    # Embarked

    x['Embarked'] = df['Embarked']

    

    return x



train_data_extracted = extract_columns(train_data_raw)

train_data_extracted.head(5)
# Replace missing values for numerical features

train_data_imputed = pd.DataFrame(train_data_extracted)

numerical_cols = ['Fare', 'TicketNum']

medians = train_data_imputed[numerical_cols].median()

train_data_imputed[numerical_cols] = train_data_imputed[numerical_cols].fillna(medians)



# For categorical features

categorical_cols = ['Pclass', 'Title', 'Sex', 'Age', 'Family', 'Embarked']

modes = train_data_imputed[categorical_cols].mode().iloc[0]

train_data_imputed[categorical_cols] = train_data_imputed[categorical_cols].fillna(modes)



train_data_imputed.info()
# Transform categorical variables to dummy variables

def transform_columns(df):

    # get dummy variables

    x = pd.DataFrame(df)

    x = pd.get_dummies(x)

    

    # drop redundunt columns

    x = x.drop(['Pclass_3', 'Title_Other', 'Sex_male', 'Age_old', 'Family_more', 'Embarked_S'], axis=1) 

    

    return x



train_data_transformed = transform_columns(train_data_imputed)



# List of all features

features = train_data_transformed.columns

features
# normalize numerical features

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

train_data_scaled = pd.DataFrame(train_data_transformed)

train_data_scaled[numerical_cols] = scaler.fit_transform(train_data_transformed[numerical_cols])
# Create multi prediction models

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV

train_x, train_y = pd.DataFrame(train_data_scaled), train_data_raw['Survived']



models = {

    'KNN': {

        'params': {

            'n_neighbors': [5, 10, 50],

            'weights': ['uniform', 'distance'],

            'algorithm': ['ball_tree', 'kd_tree', 'brute'],

            'p': [1, 2, 4],

        },

        'clf': KNeighborsClassifier(),

    },

    'SVC': {

        'params': {

            'C':[0.01, 0.1, 1, 10, 100, 1000], 

            'gamma':[0.01, 0.1, 1, 10, 100, 1000],

        },

        'clf': SVC(random_state=1, probability=True),

    },

    'RF': {

        'params': {

            'criterion': ['gini', 'entropy'], 

            'max_depth': [3, 5, 10, 20],

            'max_features': ['sqrt', None],

        },

        'clf': RandomForestClassifier(random_state=1, n_estimators=100),

    },

    'LR': {

        'params': {

            'penalty': ['l1', 'l2'],

            'C': [0.01, 0.1, 1, 10, 100, 1000],

        },

        'clf': LogisticRegression(solver='liblinear'),

    },

    'GBC': {

        'params': {

            'loss': ['deviance', 'exponential'],

            'subsample': [0.3, 0.5, 1],

            'max_depth': [3, 6, 9],

            'max_features': ['sqrt', None],

        },

        'clf': GradientBoostingClassifier(random_state=1, n_estimators=100)

    },

    'LDA': {

        'params': {

            'solver': ['svd', 'lsqr', 'eigen'],

        },

        'clf': LinearDiscriminantAnalysis(),

    }

}



for m in models:

    # Find with the best parameter

    print("### {0} ###".format(m))

    gscv = GridSearchCV(models[m]['clf'], models[m]['params'], cv=5, scoring='accuracy', iid=False)

    gscv.fit(train_x, train_y)

    print("best score: {0}".format(gscv.best_score_))

    print("best params: {0}".format(gscv.best_params_))

    models[m]['clf'] = gscv.best_estimator_
# Create 2nd level Classifier

# from sklearn.ensemble import GradientBoostingClassifier



# train_x_2nd = train_x.copy()

# for m in models:

#     train_x_2nd[m] = models[m]['clf'].predict(train_x)



# params_eclf = {

#     'loss': ['deviance', 'exponential'],

#     'subsample': [0.3, 0.5, 1],

#     'max_depth': [3, 6, 9],

#     'max_features': ['sqrt', None],

# }

# eclf = GridSearchCV(GradientBoostingClassifier(random_state=1, n_estimators=100),

#                     param_grid=params_eclf, cv=5, scoring='accuracy', iid=False)

# eclf.fit(train_x_2nd, train_y)

# pd.DataFrame(eclf.cv_results_).sort_values(by=['rank_test_score']).head()



from sklearn.ensemble import VotingClassifier

params_eclf = {

    'voting': ['hard', 'soft'],

}

eclf = GridSearchCV(VotingClassifier(estimators=[(m, models[m]['clf']) for m in models]),

                    param_grid=params_eclf, cv=5, scoring='accuracy', iid=False)

eclf.fit(train_x, train_y)

pd.DataFrame(eclf.cv_results_).sort_values(by=['rank_test_score']).head()
# Load the test data

test_data_raw = pd.read_csv("/kaggle/input/titanic/test.csv")

passenger_ids = test_data_raw['PassengerId']



# Extract columns

test_data_extracted = extract_columns(test_data_raw)



# Impute data

test_data_imputed = pd.DataFrame(test_data_extracted)

test_data_imputed[numerical_cols] = test_data_imputed[numerical_cols].fillna(medians)

test_data_imputed[categorical_cols] = test_data_imputed[categorical_cols].fillna(modes)



# Transform data

test_data_transformed = transform_columns(test_data_imputed)



# Normalize data

test_data_scaled = pd.DataFrame(test_data_transformed)

test_data_scaled[numerical_cols] = scaler.transform(test_data_transformed[numerical_cols])



# Remove additional columns

for f in test_data_scaled.columns:

    if f not in features:

        test_data_scaled = test_data_scaled.drop(f, axis=1)



# Add nescessary columns

for f in features:

    if f not in test_data_scaled.columns:

        test_data_scaled[f] = 0



# Reorganize the columns order

test_data_scaled = test_data_scaled[features]

test_data_scaled.head(5)
# predict by emsemble model

test_x = pd.DataFrame(test_data_scaled)

# test_x_2nd = test_x.copy()

# for m in models:

#     test_x_2nd[m] = models[m]['clf'].predict(test_x)

    

prediction_result = pd.DataFrame({'PassengerId': passenger_ids,'Survived': eclf.predict(test_x)})

prediction_result
# Output result

prediction_result.to_csv('submission.csv', index=False)

prediction_result['Survived'].value_counts()