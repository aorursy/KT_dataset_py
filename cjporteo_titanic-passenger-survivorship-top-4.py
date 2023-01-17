import warnings  

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np



pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 5000)



from collections import Counter, defaultdict



from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.preprocessing import LabelEncoder, RobustScaler



from sklearn.model_selection import GridSearchCV, ShuffleSplit



#Common Model Algorithms

from sklearn import svm, ensemble

from xgboost import XGBClassifier



X_train = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

X_test = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

X_train.dropna(axis=0, subset=['Survived'], inplace=True)



# This dictionary will store all the family members and their survival outcomes for each family. 

# Two passengers are said to be in the same "family" if they have the same last name and family size.

# Consider, for instance, the scenario where there are multiple Brown families on board, but one is 

# a family of 6 and the other is a lone bachelor.

# Our system will be able to discriminate between these two groups and not associate them.

#

# { key = (last name, family size), type=tuple : value = list of tuples for all family members in training set (full name, outcome) }

family_survival_details = defaultdict(list)



# This dictionary will function similarly to the family dictionary, but instead just looks for matching tickets and isn't concerned with group size.

# { key = ticket : value = list of tuples for all ticket group members in training set (full name, outcome) }

ticket_survival_details = defaultdict(list)



# This set stores all the passengers who do not have any other family members in the training set.

# More formally, this set stores all passengers satisfying len(family_survival_details[lastname, familysize]) = 1

unknown_passengers_by_family = set([])



# Again, this functions similiarly to the "unknown by family" set.

# This set stores all passengers satisfying len(ticket_survival_details[ticket]) = 1

unknown_passengers_by_ticket = set([])



# Trims the cardinality of the Title feature that we are about to engineer from the Name field

def adjust_title(title):

    if title in ['Mr', 'Miss', 'Mrs', 'Master']:

        return title

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title in ['Mme', 'Dona']:

        return 'Mrs'

    return 'Other'



# Builds the FamilyName, FamilySize and Title features

for df in [X_train, X_test]:

    df['FamilyName'] = df['Name'].map(lambda x : x.split(',')[0])

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df['Title'].map(adjust_title)



# This function populates our family survival details dictionary

def fill_fsd(row):

    last_name = row['FamilyName']

    size = row['FamilySize']

    full_name = row['Name']

    survived = row['Survived']

    family_survival_details[(last_name, size)].append((full_name, survived))



# This function populates our ticket survival details dictionary

def fill_tsd(row):

    ticket = row['Ticket']

    full_name = row['Name']

    survived = row['Survived']

    ticket_survival_details[ticket].append((full_name, survived))



# Call the above two functions

X_train.apply(fill_fsd, axis=1)

X_train.apply(fill_tsd, axis=1)



# Establish a base survival rates dictionary based on class, gender, and title.

# When we lack any family or ticket survival rate information about a passenger, we will use this base rate.

# Even when we have family data, this base rate will still be useful. See below for details.

base_survival_rates = dict(X_train.groupby(by=['Pclass', 'Sex', 'Title'])['Survived'].mean())



# This will give us what we are looking for, our crucial feature: family specific survival rates.

# For each passenger in the training set and the testing set, we inquire about the known outcomes for 

# all OTHER seen members of that family group.

# It is very important that we don't include the outcome for the passenger in question in this calculation,

# since this would lead to data leakage and would tarnish the usefulness of this predictor.

#

# Comments have been added to this function for clarity.

def get_fsr(row):

    last_name = row['FamilyName']

    size = row['FamilySize']

    full_name = row['Name']

    # Where we are storing the known outcomes for other passengers in the family.

    outcomes = []

    for passenger, outcome in family_survival_details[(last_name, size)]:

        if passenger != full_name:

            # We only care about the outcome for OTHER passengers in the family.

            # Ex: When building the family survival rate for John Smith, we don't

            # care about whether he survived or not, only his family. (Sorry John)

            outcomes.append(outcome)

    if not outcomes:

        # If we don't have any known outcomes for other family members, add this passenger

        # to the unknown set and return 0 as a survival rate (we will adjust this later)

        unknown_passengers_by_family.add(full_name)

        return 0

    

    # Return the average of all the outcomes to get a probility to estimate survival.

    return np.mean(outcomes)



# This is simply the ticket counterpart to the above function. The inner workings are very similar.

def get_tsr(row):

    ticket = row['Ticket']

    full_name = row['Name']

    outcomes = []

    for passenger, outcome in ticket_survival_details[ticket]:

        if passenger != full_name:

            outcomes.append(outcome)

    if not outcomes:

        unknown_passengers_by_ticket.add(full_name)

        return 0

    

    return np.mean(outcomes)



for df in [X_train, X_test]:

    df['FamilySurvival'] = df.apply(get_fsr, axis=1)

    df['TicketSurvival'] = df.apply(get_tsr, axis=1)

for df in [X_train, X_test]:

    df['KnownFamily?'] = df['Name'].map(lambda x: 0 if x in unknown_passengers_by_family else 1)

    df['KnownTicket?'] = df['Name'].map(lambda x: 0 if x in unknown_passengers_by_ticket else 1)



unknown_passengers = set([])



# This function amalgamates every result we have so far involving base, family, and ticket survival rate.

# The resulting overall survival rate will be a weighted average of these three rates.

def get_osr(row):

    base_rate = base_survival_rates[(row['Pclass'], row['Sex'], row['Title'])]

    if row['KnownFamily?'] and row['KnownTicket?']:

        # The passenger can be identified by family and ticket group.

        return 0.25*row['FamilySurvival'] + 0.25*row['TicketSurvival'] + 0.5*base_rate

    elif row['KnownFamily?']:

         # The passenger can be identified by family group only.

        return 0.5*row['FamilySurvival'] + 0.5*base_rate

    elif row['KnownTicket?']:

        # The passenger can be identified by ticket group only.

        return 0.5*row['TicketSurvival'] + 0.5*base_rate

    else:

        # The passenger can't be identified by family or ticket group.

        unknown_passengers.add(row['Name'])

        return base_rate



for df in [X_train, X_test]:

    df['GroupRate'] = df.apply(get_osr, axis=1)

    df['GroupRate'] = df.apply(get_osr, axis=1)

for df in [X_train, X_test]:

    df['KnownGroup?'] = df['Name'].map(lambda x: 0 if x in unknown_passengers else 1)

    df.drop(['FamilySurvival', 'TicketSurvival', 'KnownFamily?', 'KnownTicket?'], axis=1, inplace=True)

    

y = X_train['Survived']

X_train.drop(['Survived'], axis=1, inplace=True)

train_size = len(X_train)



'''

# Which features have missing values and what are is their datatype?

def investigate_missing(df):

    for col in df:

        missing = df[col].isnull().sum()

        if missing > 0:

            print("{}: {} missing --- type: {}".format(col, missing, df[col].dtype))

            

investigate_missing(pd.concat([X_train, X_test]))

'''



def featureProcessing(df):

    

    # Change class from numerical to categorical, since class is ordinal.

    df['Pclass'] = df['Pclass'].astype(str)

    

    # Impute missing fares by the average fare across corresponding class, gender, and title.

    df['Fare'] = df.groupby(by=['Pclass', 'Sex', 'Title'])['Fare'].transform(lambda x : x.fillna(x.median()))

    

    # Impute missing ages by the average age across corresponding class, gender, and title. 

    df['Age'] = df.groupby(by=['Pclass', 'Sex', 'Title'])['Age'].transform(lambda x : x.fillna(x.median()))

    

    # Fill missing embarking locations with S, the mode.

    df['Embarked'].fillna('S', inplace=True)

      

    # Passengers travelling together might not have matching last names (friends?) but could have matching tickets.

    # Create a feature to represent the size of the "ticket group".

    df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')

    

    # Condense family size and ticket group size into the maximum between the two.

    # Split this feature into five bins.

    df['GroupSize'] = df[["FamilySize", "TicketGroupSize"]].max(axis=1)

    group_bins = [0, 1, 2, 4, 6, df['GroupSize'].max()]

    group_labels = ['Alone', 'Small', 'Medium', 'Large', 'XLarge']

    df['GroupSize'] = pd.cut(df['GroupSize'], group_bins, labels=group_labels).astype(str)

    

    # "Women and children only"

    df['Female?'] = np.where(df['Sex'] == 'female', 1, 0)

    adults = ['Mr', 'Mrs', 'Other']

    df['Parent?'] = np.where((df['Title'].isin(adults)) & (df['Parch'] > 0), 1, 0)

    

    # Cabin has lot of missing values, but of these missing values, the vast majority lie in 2nd and 3rd class.

    # Doing some simple research on the deck layout of Titanic and the likely cabin assignments. We can make

    # relatively safe assumptions to categorize passengers into their respective deck categories.

    # The three deck categories I will use here are ABC, DE, and FG.

    # These very strongly correlate to 1st, 2nd and 3rd class, respectively, but there is enough variation to warrant

    # the inclusion of this feature.

    #

    # Ex: a 3rd class passenger whose cabin happens to be located in D or E deck will stand a better chance of survival

    # than a 3rd class passenger in F or G

    def get_deck(row):  

        if pd.isnull(row['Cabin']):

            if row['Pclass'] == '1':

                return 'ABC'

            elif row['Pclass'] == '2':

                return 'DE'

            return 'FG'

        deck = row['Cabin'][0]

        if deck in 'ABCT':

            return 'ABC'

        elif deck in 'DE':

            return 'DE'

        return 'FG'

    

    df['DeckClass'] = df.apply(get_deck, axis=1) 

    

    df.drop(['Ticket', 'Cabin', 'Name', 'Parch', 'SibSp', 'Sex', 'FamilyName', 'TicketGroupSize', 'FamilySize'], axis=1, inplace=True)



    # Our numerical features

    numericals = ['Fare', 'Age']

    

    # Our categorical features

    categoricals = df.select_dtypes(include='object').columns.values

    

    # Process the numerical features

    skewness = df[numericals].apply(lambda x: skew(x))

    skew_index = skewness[abs(skewness) >= 0.5].index

    # Get features with high skew

    for col in skew_index:

        # Apply boxcox transformation to attempt to reduce their skewness

        df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))       

    for feat in numericals:

        # Reshape using RobustScaler

        df[feat] = RobustScaler().fit_transform(df[feat].apply(float).values.reshape(-1,1))

    

    # One-hot encode the categorical features

    for feat in categoricals:

        dummies = pd.get_dummies(df[feat])

        dummies.columns = [feat + ": " + col for col in dummies.columns.values]

        df.drop(feat, axis=1, inplace=True)

        df = df.join(dummies)

    

    return df



X_full = featureProcessing(pd.concat([X_train, X_test]))



X_train = X_full[:train_size]

X_test = X_full[train_size:]



# We will use five different models, tune them, and then hold a vote.

vote_est = [

    ('for', ensemble.RandomForestClassifier()),

    ('svc', svm.SVC(probability=True)),

    ('xgb', XGBClassifier()),

    ('ada', ensemble.AdaBoostClassifier()),

    ('gb', ensemble.GradientBoostingClassifier())           

]



# The tuned hyperparameters for each of the models. Obtained using GridSearchCV to cross-validate

# on five shuffle splits for each hyperparameter combination with train = 0.6, test = 0.3, drop = 0.1.

tuned_parameters=[

    {'criterion': 'entropy', 'max_depth': 12, 'n_estimators': 550, 'oob_score': True, 'random_state': 0},

    {'C': 10, 'decision_function_shape': 'ovo', 'gamma': 0.01, 'kernel': 'rbf', 'probability': True, 'random_state': 0},

    {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100, 'random_state': 0},

    {'algorithm': 'SAMME', 'learning_rate': 0.15, 'n_estimators': 250, 'random_state': 0},

    {'learning_rate': 0.01, 'loss': 'deviance', 'max_depth': 4, 'n_estimators': 100, 'random_state': 0}

]



#tuned_parameters = []



#The hyperparameter search space for our models.

grid_n_estimator = [100, 250, 400, 550, 700, 850, 1000]

grid_ratio = [.1, .25, .5, .75, 0.9]

grid_learn = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15]

grid_max_depth = [2, 4, 6, 8, 10, 12, 15, 20, None]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]



grid_param = [

            [{

            #RandomForestClassifier

            'n_estimators': grid_n_estimator,

            'criterion': grid_criterion,

            'max_depth': grid_max_depth,

            'oob_score': [True],

            'random_state': grid_seed

             }],

    

            [{

            #SVC

            'kernel': ['rbf'],

            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],

            'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],

            'decision_function_shape': ['ovo', 'ovr'],

            'probability': [True],

            'random_state': grid_seed

             }],

    

            [{

            #XGBClassifier

            'learning_rate': grid_learn, #default: .3

            'max_depth': [3, 4, 5, 6, 8, 10], #default 2

            'n_estimators': grid_n_estimator,

            'random_state': grid_seed

             }],

            

            [{

            #AdaBoostClassifier

            'algorithm': ['SAMME','SAMME.R'],

            'n_estimators' : grid_n_estimator,

            'learning_rate':  grid_learn,

            'random_state': grid_seed

            }],

    

            [{

            #GradientBoostingClassifier

            'loss' : ['deviance'],

            'n_estimators' : grid_n_estimator,

            'learning_rate': grid_learn,

            'max_depth': [3, 4, 5, 6, 8, 10],

            'random_state': grid_seed

            }]

]



cv_split = ShuffleSplit(n_splits = 5, test_size = .3, train_size = .6, random_state = 0) 



# This code is only executed if the list of tuned parameters is empty. It will find the optimal hyperparameters.

if not tuned_parameters:

    for clf, param in zip(vote_est, grid_param):  

        

        print(clf[0])



        best_search = GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'accuracy', n_jobs=-1, verbose=True)

        best_search.fit(X_train, y)



        print("Best Parameters:\n{}".format(best_search.best_params_))



        print("Best Score: {}\n".format(best_search.best_score_))



        best_param = best_search.best_params_

        clf[1].set_params(**best_param)

else:

    # Set the model parameters to the tuned values.

    for clf, tuned_param in zip(vote_est, tuned_parameters):

        clf[1].set_params(**tuned_param)



# Conduct the hard vote.

vote_hard_tuned = ensemble.VotingClassifier(estimators = vote_est, voting = 'hard')



vote_hard_tuned.fit(X_train, y)



preds = vote_hard_tuned.predict(X_test)



output = pd.DataFrame({'PassengerId' : X_test.index,

                       'Survived' : preds})



output.to_csv('submission.csv', index=False)