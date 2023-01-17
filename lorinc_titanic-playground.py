# fixing seaborn warnings

# !pip install --upgrade pip

# !pip install --upgrade scipy

# !pip install featuretools
# imports

import numpy as np

import pandas as pd



import missingno as msno

import featuretools as ft

from fancyimpute import KNN



import seaborn as sns

from matplotlib import rcParams

from matplotlib import pyplot as plt



# plotting config

rcParams['figure.figsize'] = 20,15

sns.set(style="whitegrid", palette="hls", color_codes=True)
# reading data

train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)

full = train.append(test, ignore_index=True, sort=False)

train = test = None



# making names a bit more useful

pattern = r'^([\w\' -]+),[ ]*(\w+)\.[ ]*([\w\' -]+)?[ ]*?(.*)'

full = (

    full.merge(

        full['Name'].str.extract(pattern),

        left_index=True, right_index=True)

    .rename(columns={0: 'Name_Family', 

                     1: 'Name_Title', 

                     2: 'Name_First', 

                     3: 'Name_Misc'}))



# looking at the only person, whose name did not match our regex

# and fixing it manually, instead of complicating it

# >>> 'Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)'

full.loc[759,'Name_Misc'] = '(Lucy Noel Martha Dyer-Edwards)'

full.loc[759,'Name_Title'] = 'Countess'

full.loc[759,'Name_Family'] = 'Rothes'

full.loc[759,'Name_First'] = ''



# trying to make any sense of the ticket data

full = (

    full.join(

        full['Ticket']

            .str.split(r'[ ]*([0-9]{4,})', expand=True)

            .rename(columns={0: 'Ticket_str', 1: 'Ticket_number'}))

    .drop(columns=[2]))



# cabin to int

full['Cabin_known'] = full['Cabin'].isna().astype(int)



# C = Cherbourg; Q = Queenstown; S = Southampton

full['Embarked'].replace(

    {'C':'Cherbourg', 

     'Q':'Queenstown', 

     'S':'Southampton'},

    inplace=True)



# normalizing names

title_map = {

    'mr': ['Mr', 'Don', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer', 'Rev' ],

    'mrs': ['Mrs', 'Mme', 'Lady', 'Mlle', 'Dona', 'Countess'],

    'ms': ['Miss', 'Ms'],

    'mister': ['Master'] # boys were called mister at this time

}



for key in title_map.keys():

    full['Name_Title'].replace(title_map[key], key, inplace=True)

    

# encoding Name_Misc

full['Name_Has_Nickname'] = full['Name_Misc'].fillna('').str.contains('\"').astype(int)

full['Name_Has_Birthname'] = full['Name_Misc'].fillna('').str.contains('\(').astype(int)



# 1=male, 0=female

full.loc[:,'Sex'].replace({'male':1, 'female':0}, inplace=True)



# encoding embarked and name titles - not yet, it oversaturates exploratory plots

full = full.join(pd.get_dummies(full[['Embarked', 'Name_Title']]))

# checking the structure of missing data to guarantee, imputation does not introduce bias

ax = msno.matrix(full)
# based on the result, we can say, missing values are largery independent

ax = msno.heatmap(full)
# creating a list for the categorical and one for the numeric columns

dropcols = ['Name', 'Ticket', 'Cabin', 'Embarked', 'Name_Family',

'Name_Title', 'Name_First', 'Name_Misc', 'Ticket_str']

corecols = [column for column in full.columns if column not in dropcols]



# KNN imputation

X_KNN_impute = pd.DataFrame(KNN(k=3).fit_transform(full.drop(columns=dropcols)), columns=corecols)

X_KNN_impute.loc[:,'Survived'] = full['Survived']



# saying good-bye to the original table

full = None
X_KNN_impute['ix'] = X_KNN_impute.index



es = ft.EntitySet(id='entity_set')



es.entity_from_dataframe(

    entity_id='social_status', 

    dataframe=X_KNN_impute.iloc[:,1::],

    index='ix')



feature_matrix, feature_names = ft.dfs(entityset=es, 

                                       target_entity = 'social_status', 

                                       max_depth = 3, 

                                       verbose = 1, 

                                       n_jobs = 3)

feature_matrix.columns



#TODO: create multiple dimensions, so that featuretool can permutate. e.g. social status, ship-stuff, family-stuff
# only showing the strong correlations



corr = (

    X_KNN_impute.corr()

        .applymap(

            lambda x: 

                x if (-.4 >= x or x >=.4) 

                else 0

        )

)



# plotting

fig, ax = plt.subplots(figsize=(20,15))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

plt.xticks(range(len(corr.columns)), corr.columns);

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()
# dropping useless attributes - not yet, I need them for imputation

# full.drop(columns=['Ticket', 'Name_Family', 'Name_First', 'Name_Misc', 'Ticket_str', 2 ], inplace=True)



# split gender to roles: orphan_girl, single_man, married_men, etc