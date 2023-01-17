%reset

import numpy as np

import pandas as pd

import re

from sklearn.ensemble import ( AdaBoostClassifier,

                              BaggingClassifier,

                              GradientBoostingClassifier,

                              RandomForestClassifier,

                              ExtraTreesClassifier,                                                           

                              VotingClassifier)

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV





train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



train_data.head(1)
def prepare_data(train_data, test_data):

    

    data = train_data.append(test_data)

    data.index = range(0, len(data))    

    

    #SAME SURNAME SURVIVORS

    data['Surname'] = data['Name'].map(lambda name:name.split(',')[0].strip())

    survivors_by_surname = data.groupby(['Surname'])['Survived'].sum()

    surname_count = data.groupby(['Surname'])['Survived'].count()

    survivors_by_surname_dict = dict(zip(survivors_by_surname.index, survivors_by_surname.values))

    surname_count_dict = dict(zip(surname_count.index, surname_count.values))

    data['Surname_Survivors'] = data.apply( lambda x: max(0, survivors_by_surname_dict.get(x.Surname) - 1), axis = 1)

    data['Surname_Count'] = data.apply( lambda x:  surname_count_dict.get(x.Surname), axis = 1)    

    

    #TITLE

    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    titles = list(enumerate(np.unique(data.Title.dropna())))

    title_dict = {

                        "Mr" :        "1",

                        "Capt":       "2",

                        "Col":        "2",

                        "Major":      "2",

                        "Jonkheer":   "2",

                        "Don":        "2",

                        "Rev":        "2",

                        "Sir" :       "2",

                        "Dr":         "2",

                        "Master" :    "3",

                        "Ms":         "4",                        

                        "Miss" :      "4",

                        "Mrs" :       "5",

                        "Mlle":       "5" ,

                        "the Countess":"6",

                        "Dona":       "6",

                        "Lady" :      "6",

                        "Mme":        "6"                                                                                                                     

                        }

    data['Title_Class'] = data.apply(lambda row: title_dict.get(row.Title), axis = 1).astype(int)

            

    #AGE

    median_age_dict = {}    

    for title in range(1,6):       

        median_age_dict[title] = np.median(data.loc[data.Title_Class == title, ('Age')].dropna())

    data.loc[data.Age.isnull(), ('Age')] = data[data.Age.isnull()].apply( lambda x: median_age_dict.get(x.Title_Class), axis = 1)    

    data['Infant'] = data.apply( lambda row: int(row['Age'] <= 8), axis = 1)

    data['Elderly'] = data.apply( lambda row: int(row['Age'] >= 65), axis = 1)

    data['No_Parents'] = data['Parch'].map(lambda parch: int(parch == 0))

    data['Infant_No_Parents'] = data['Infant'] & data['No_Parents']

    

    #SEX

    data['Sex'] = data['Sex'].map({'female':0, 'male': 1}).astype(int)



    #FAMILY SIZE

    data['Family_Size'] = data['SibSp'] + data['Parch'] + 1 

    data['Singleton'] = data.apply( lambda row: int(row['Family_Size'] <= 1), axis = 1)

    data['Pair_Family'] = data.apply( lambda row: int(row['Family_Size'] == 2), axis = 1)

    data['Big_Family'] = data.apply( lambda row: int(row['Family_Size'] >=3), axis = 1)

 

    #TICKET

    data.loc[data['Ticket'] == "LINE", ('Ticket')] = 'LINE 00000'

    data['Ticket_Number'] = data.apply(lambda row: max(list(map(int, re.findall(r'\d+', row['Ticket'])))) , axis = 1)

    data['Ticket_String'] = data.apply(lambda row: re.sub("[0-9]", "", row['Ticket']) , axis = 1)

    data['Ticket_String'] = data.apply(lambda row: re.sub("/", "", row['Ticket_String']) , axis = 1)

    data['Ticket_String'] = data.apply(lambda row: re.sub("\.", "", row['Ticket_String']) , axis = 1)

    Ticket_Strings = list(enumerate(np.unique(data.Ticket_String)))

    Ticket_Strings_dict = { name : i for i, name in Ticket_Strings }

    data['Ticket_String_Id'] = data.Ticket_String.map(lambda x: Ticket_Strings_dict[x]).astype(int)  

    

    #EMBARKED

    data.loc[data.Embarked.isnull(), ('Embarked')] = train_data.Embarked.mode()[0]

    Ports = list(enumerate(np.unique(data.Embarked)))

    Ports_dict = { name : i for i, name in Ports }

    data.Embarked = data.Embarked.map(lambda x: Ports_dict[x]).astype(int)  

    

    #FARE

    Classes = np.unique(data.Pclass)

    data.loc[data.Fare.isnull(), ('Fare')] = 0

    median_fare_dict = {}

    for embarked in range(0,3):

        for pass_class in Classes:

                median_fare_dict[(embarked, pass_class)] = np.median(data.Fare[(data.Embarked == embarked) & (data.Pclass == pass_class)].dropna())        

    data.loc[data.Fare == 0, ('Fare')] = data.loc[data.Fare.isnull()].apply( lambda x: median_fare_dict.get(x.Embarked, x.Pclass), axis = 1)

    data.loc[data.Fare.isnull(), ('Fare')] = data.loc[data.Fare.isnull()].apply( lambda x: median_fare_dict.get(x.Embarked, x.Pclass), axis = 1)

    

    #DECK

    data['Deck'] = data.Cabin.dropna().apply(lambda row: row[0])

    Decks = list(enumerate(np.unique(data.Deck.dropna())))

    Decks_dict = { name : i for i, name in Decks }

    data.Deck = data.Deck.dropna().map(lambda x: Decks_dict[x]).astype(int)

    family_deck = data.groupby(['Surname'])['Deck'].median()

    family_deck_dict = dict(zip(family_deck.index, family_deck.values))

    data['Deck'] = data.apply( lambda x: family_deck_dict.get(x.Surname), axis = 1)

    data.loc[data.Deck.isnull(), ('Deck')] = -1

         

    #CABINMATE SURVIVORS

    cabinmates_survived = train_data.groupby(['Cabin'])['Survived'].sum() - 1

    cabinmates_survived [ cabinmates_survived == -1.0] = 0.0

    cabinmates_survived_dict = dict(zip(cabinmates_survived.index, cabinmates_survived.values))    

    data['Cabinmates_Survived'] = data.apply( lambda x: cabinmates_survived_dict.get(x.Cabin), axis = 1)    

    data.loc[data.Cabinmates_Survived.isnull(), ('Cabinmates_Survived')] = 0.0            

    data['Family_Size_Surname'] = data.apply(lambda row: min(row.Surname_Count, row.Family_Size), axis = 1).astype(int)    

    return data[:len(train_data)], data[len(train_data):]



train_data, test_data = prepare_data(train_data, test_data)
#FEATURE SELECTION

features = ['Pclass', 

            'Sex',

            'Age',   

            'Parch',

            'SibSp',

            'Ticket_Number', 

            'Ticket_String_Id',

            'Fare',

            'Embarked',            

            'Family_Size',

            'Deck',

            'Cabinmates_Survived',                                             

            'Surname_Survivors',

            'Family_Size_Surname',

            'Title_Class',

            'Surname_Count',

            'Infant',

            'Elderly',

            'Infant_No_Parents',

            'Singleton',

            'Pair_Family',

            'Big_Family'

            ]
#GRID SEARCH FOR HYPERPARAMETER OPTIMIZATION



nontree_classifiers = (

                AdaBoostClassifier(),

                BaggingClassifier(),

                GradientBoostingClassifier()               

              )



tree_classifiers = (

                RandomForestClassifier(), 

                ExtraTreesClassifier()                                      

            )



nontree_parameter_grid = {                            

                 'n_estimators': [50, 100, 150, 200, 400]                 

                 }



tree_parameter_grid = {            

                 'max_depth' : [3, 5, 7, 9],

                 'n_estimators': [50, 100, 150, 200, 400]                 

                 }



cross_validation = StratifiedKFold(n_splits = 5)



for clf in nontree_classifiers:

    grid_search = GridSearchCV(clf, param_grid=nontree_parameter_grid, cv = cross_validation)

    grid_search.fit(train_data[features], train_data['Survived'])

    print(clf)

    print('Best score: {}'.format(grid_search.best_score_))

    print('Best parameters: {}'.format(grid_search.best_params_))

    

for clf in tree_classifiers:

    grid_search = GridSearchCV(clf, param_grid=tree_parameter_grid, cv = cross_validation)

    grid_search.fit(train_data[features], train_data['Survived'])

    print(clf)

    print('Best score: {}'.format(grid_search.best_score_))

    print('Best parameters: {}'.format(grid_search.best_params_))  

    print('\n')
#VOTING CLASSIFIER



adac = AdaBoostClassifier(n_estimators = 50).fit(train_data[features], train_data['Survived'])

bagc = BaggingClassifier(n_estimators= 150).fit(train_data[features], train_data['Survived']) 

gbc = GradientBoostingClassifier(n_estimators = 50).fit(train_data[features], train_data['Survived'])   

rfc = RandomForestClassifier(n_estimators = 50, max_depth = 9).fit(train_data[features], train_data['Survived']) 

etc = ExtraTreesClassifier(n_estimators = 100, max_depth = 9).fit(train_data[features], train_data['Survived'])                





voting_clf = VotingClassifier(estimators = [('adac', adac),

                                            ('bag', bagc),

                                            ('gbc', gbc),

                                            ('rfc', rfc),

                                            ('etc', etc),                                           

                                            ],

                              voting = 'soft',

                              weights=[1,1,1,1,1])



voting_clf = voting_clf.fit(train_data[features], train_data['Survived'])    

test_data[['Survived']] = voting_clf.predict(test_data[features])

test_data[['Survived']] = test_data[['Survived']].astype(int)

data_submission = test_data[['PassengerId', 'Survived']]

data_submission.to_csv('submission.csv', index = False )

print(features, "\n", voting_clf)