import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV



from sklearn.base import BaseEstimator, TransformerMixin



from sklearn.metrics import mean_absolute_error



%matplotlib inline
# Read the data

X_full = pd.read_csv('../input/titanic/train.csv')

X_test_full = pd.read_csv('../input/titanic/test.csv')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['Survived'], inplace=True)

y = X_full.Survived

X_full.drop(['PassengerId'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.7, test_size=0.3,

                                                                random_state=2)



fig, (train, test) = plt.subplots(ncols=2, sharey=True, figsize=(15,5))

sns.heatmap(data=X_full.isnull(), ax=train)

sns.heatmap(data=X_test_full.isnull(), ax=test)
categorical_cols = ['Sex', 'Embarked']

numerical_cols = ['Pclass', 'SibSp', 'Parch']

age_col = ['Age']

fare_col = ['Fare']

complex_cols = ['Name','Ticket','Cabin']



# Keep selected columns only

my_cols = categorical_cols + numerical_cols + age_col + fare_col + complex_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



X_train.head()
avg_survival_chance = X_full['Survived'].mean()

avg_survival_chance_male = X_full[X_full['Sex']=='male']['Survived'].mean()

avg_survival_chance_female = X_full[X_full['Sex']=='female']['Survived'].mean()

print('all: {}, male {}, female {}'.format(avg_survival_chance, avg_survival_chance_male, avg_survival_chance_female))
ticket_survival = X_full[['Ticket','Survived']].copy()

ticket_survival['TicketCode'] = ticket_survival['Ticket'].map(lambda c : c.split()[0])

ticket_survival['TicketCode'] = ticket_survival['TicketCode'].map(lambda c : 'NUM' if c.isdigit() else c)

ticket_survival = ticket_survival.groupby('TicketCode').mean().sort_values(by='Survived')['Survived']

ticket_survival
code_dict = {}

def split_ticket_codes(surv):

    if surv < 0.05:

        return 1

    elif surv < 0.15:

        return 2

    elif surv < 0.3:

        return 3

    elif surv < 0.4:

        return 4

    elif surv < 0.48:

        return 5

    elif surv < 0.6:

        return 6

    elif surv < 0.8:

        return 7

    return 8



for i in ticket_survival.index:

    code_dict[i] = split_ticket_codes(ticket_survival[i])

code_dict['Y'] = split_ticket_codes(avg_survival_chance)
cabin_survival = X_full[['Cabin','Survived']].copy()

cabin_survival['Code'] = cabin_survival['Cabin'].map(lambda c : str(c)[0])

cabin_survival = cabin_survival.groupby('Code').mean().sort_values(by='Survived')

cabin_survival
cabin_dict = {'T':1, 'n':2, 'A': 3, 'G':4, 'C':5, 'F':6, 'B':7, 'E':8, 'D':9}
title_survival = X_full[['Name','Survived']].copy()

title_survival['Title'] = title_survival['Name'].map(lambda name : name.split(',')[1].split('.')[0].strip())

title_survival = title_survival.groupby('Title').mean().sort_values(by='Survived')

title_survival
title_dict = {'Capt':0, 'Don':1, 'Jonkheer':2, 'Rev':3,

                'Mr':4,

                'Dr':7,

                'Col':8, 'Major':9, 'Master':10,

                'Miss':11,

                'Mrs':15, 

                'Mme':16, 'Sir':17, 'Ms':18, 'Lady':19, 'Mlle':20, 'the Countess':21,

              #only in test data - avg survival rate for male/female

              #male 0.18890814558058924, female 0.7420382165605095:

                  'Don':5, 'Jonkheer':6, 

                  'Dona':12, 'Lady':13}
train_age = X_train[['Age','Name']].append(X_test[['Age','Name']])
def get_title(name):

    return name.split(',')[1].split('.')[0].strip()

    

def get_title_id(title):

    return title_dict[title] if title in title_dict else 9

    

def get_age_per_title(x):

    x['Title'] = x.Name.map(lambda x : get_title(x))

    return x.groupby('Title').Age.mean()

    

avg_age_per_title = get_age_per_title(train_age) 
train_fare = X_train[['Fare','Pclass']].append(X_test[['Fare','Pclass']])
def get_fare_per_pclass(x):

    avg_fare = []

    

    for i in range(1,4):

        avg_fare.append(x[x['Pclass']==i]['Fare'].mean())



    print('avg_fare: {}'.format(avg_fare))

    

    return avg_fare



avg_fare = get_fare_per_pclass(train_fare)
class NameAgeFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

        

    def fit(self, x, y=None):

        return self



    def omit_garbage(self, name):

        #make sure there is space before and after parenthes

        name = name.replace('(', ' (')

        name = name.replace(')', ' )')

        parts = name.split()

        

        idx_start = -1

        idx_end = -1

        has_nick = 0

        for p in parts:

            i = p.find('(')

            j = p.find(')')

            if i >= 0:

                idx_start = i

            if j >= 0:

                idx_end = j

                

            i = p.find('"')

            if i >= 0:

                has_nick = 1

        comp_len = idx_end - idx_start

        return len(parts) - comp_len - has_nick

            

        

    def transform(self, x):

        x['Title'] = x['Name'].map(lambda name : get_title(name))

        #x['Company'] = x['Name'].map(lambda name : 0 if name.find('(') >= 0 else 1)

        x['NameLen'] = x['Name'].map(lambda name : self.omit_garbage(name))

        

        x['Age'] = x.apply(lambda x : avg_age_per_title[x['Title']] if pd.isnull(x['Age']) else x['Age'], axis=1)

        x['Title'] = x['Title'].map(lambda title : title_dict[title])

        

        x.drop('Name', axis=1, inplace=True)

    

        return x.values
class CabinFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

        

    def fit(self, x, y=None):

        return self



    def transform(self, x):

        x.Cabin.fillna('n0', inplace=True)

        

        x['CabinCode'] = x['Cabin'].map(lambda c : str(c)[0])

        x['CabinCode'] = x['CabinCode'].map(lambda c : int(cabin_dict[c]))

        x.drop('Cabin', axis=1, inplace=True)

    

        return x.values
class FareFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

        

    def fit(self, x, y=None):

        return self



    def fillEmpty(self, fare, pclass):

        if pd.isnull(fare):

            return avg_fare[int(pclass)-1]

        return fare

    

    def transform(self, x):

        x['Fare'] = x.apply(lambda c :  self.fillEmpty(c['Fare'], c['Pclass']), axis=1)

        x.drop('Pclass', axis=1, inplace=True)

        

        return x.values
class TicketFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

        

    def fit(self, x, y=None):

        return self



    def transform(self, x):

        x.Ticket.fillna('NUM 0', inplace=True)

        

        x['TicketCode'] = x['Ticket'].map(lambda c : c.split()[0])

        x['TicketCode'] = x['TicketCode'].map(lambda c : c if not(c.isdigit()) else 'NUM')

        x['TicketCode'] = x['TicketCode'].map(lambda c : int(code_dict[c]) if c in code_dict else code_dict['Y'])

        x['TicketNum'] = x['Ticket'].map(lambda c : c.split()[-1])

        x['TicketNum'] = x['TicketNum'].map(lambda c : int(c) if c.isdigit() else 0)

        x.drop('Ticket', axis=1, inplace=True)

    

        return x.values
# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Define model

model = RandomForestClassifier(random_state=2)



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('ticket', TicketFeaturesTransformer(), ['Ticket']),

        ('names', NameAgeFeaturesTransformer(), ['Name','Age']),

        ('cabin', CabinFeaturesTransformer(), ['Cabin']),

        ('fare', FareFeaturesTransformer(), ['Fare','Pclass']),

        ('cat', categorical_transformer, ['Embarked'])#categorical_cols)

    ])





# Bundle preprocessing and modeling code in a pipeline

pipeline = Pipeline(steps=[('preprocessor', preprocessor)

                            ,('model', model)

                          ])



pipeline.fit(X_train, y_train)
trans_train = pd.DataFrame(preprocessor.fit_transform(X_train, y_train))

trans_train.columns = ['TicketCode', 'TicketNum', 

                       'Age', 'Title', #'Company',

                       'NameLen', 

                       'CabinCode', 'Fare',

                       #'SexA', 'SexB', 

                       'EmbarkedA', 'EmbarkedB', 'EmbarkedC']

trans_train['Survived'] = y_train

sns.heatmap(trans_train.corr())
parameters = { 

    'model__n_estimators': [560, 580, 600, 620, 640],

}



grid_search = GridSearchCV(pipeline, parameters, scoring = 'neg_mean_absolute_error', n_jobs= 1, cv=3)

#grid_search.fit(X_train, y_train)

grid_search.fit(X_full, y)
print(grid_search.best_params_) #450

print(-1 * grid_search.best_score_) #16722783389450058#0.16610549943883277
model = RandomForestClassifier(random_state=2, n_estimators=580)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)

                            ,('model', model)

                          ])



pipeline.fit(X_full.drop(['Survived'], axis=1), y)#train, y_train)

predict = pipeline.predict(X_test_full.drop(['PassengerId'], axis=1))
out = pd.DataFrame({'PassengerId': X_test_full['PassengerId'], 'Survived': predict})

out['Survived'] = out.apply(lambda x : int(x['Survived']), axis=1)

out.to_csv('titanicOut.csv', index=False)
out = pd.read_csv('titanicOut.csv')

out