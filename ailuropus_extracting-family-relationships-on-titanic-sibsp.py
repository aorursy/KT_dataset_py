import pandas as pd

import numpy as np

X_train_initial = pd.read_csv('../input/train.csv', index_col='PassengerId')

X_test_initial = pd.read_csv('../input/test.csv', index_col='PassengerId')

y_train = X_train_initial['Survived']

# In further data transformations it would be easier to deal with combined train and test sets

X_total = pd.concat([X_train_initial, X_test_initial], axis=0)
X_total['PassengerId'] = X_total.index

X_total['FirstName'] = X_total['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)', expand=False)[1]

X_total['Title'] = X_total['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

X_total['LastName'] = X_total['Name'].str.extract('([A-Za-z]+),', expand=False)
extract_columns = ['PassengerId','Sex','Age','SibSp','FirstName','Title','LastName','Ticket','Cabin']

extract_lastnames = ['Hickman','Laroche','Renouf','Andersson']

print(X_total[extract_columns].loc[X_total['LastName'].isin(extract_lastnames) & (X_total['SibSp'] > 0)].sort_values(by=['LastName','SibSp','Age']))
# Returns next SibSpID to be added.

def get_next_SibSpID(SibSpData):

    return np.nan_to_num(SibSpData['SibSpID'].max()) + 1



# Returns a dataframe with mutual differences between rows within one column col in dataframe X

def gen_diff_table(X, col):

    AgeDiff = pd.DataFrame(columns=['x','y','val'])

    for i in range(0,X.shape[0]):

        for j in range(i+1,X.shape[0]):

            AgeDiff = AgeDiff.append(pd.DataFrame([[X.index[i], X.index[j], abs(X[col].iloc[i] - X[col].iloc[j])]], columns=['x','y','val']),ignore_index=True)

    return AgeDiff



# Checks if any of passengers from PassengerIds list are already in SibSp relation

def check_SibSp_relation(PassengerIds, SibSpData):

    groups = []

    for i in range(0,len(PassengerIds)):

        groups.extend((SibSpData['SibSpID'][SibSpData['PassengerId'] == PassengerIds[i]]).tolist())

    if len(set(groups)) == len(groups):

        return True

    else:

        return False



# Adds data on SibSp connection to the table

def add_SibSp(SibSpData, SibSpID, sibsp_pas_id):

    return SibSpData.append(pd.DataFrame([[SibSpID, sibsp_pas_id]], columns=['SibSpID','PassengerId']),ignore_index=True)



# Adds one-to-one SibSp connections iteratively within SibSp_extract group, starting from min age difference up till max_SibSp_agediff

def add_SibSp_from_group_by_age(X_total, SibSp_extract, SibSpData, max_SibSp_agediff=30):

    AgeDiff = gen_diff_table(SibSp_extract, 'Age')

    AgeDiff.sort_values(by='val', inplace=True)

    for index, row in AgeDiff.iterrows():

        if AgeDiff['val'].loc[index] <= max_SibSp_agediff and check_SibSp_relation([AgeDiff['x'].loc[index], AgeDiff['y'].loc[index]], SibSpData) and X_total['SibSpUnassigned'].loc[AgeDiff['x'].loc[index]] > 0 and X_total['SibSpUnassigned'].loc[AgeDiff['y'].loc[index]] > 0:

            next_SibSpID = get_next_SibSpID(SibSpData)

            SibSpData = add_SibSp(SibSpData, next_SibSpID, AgeDiff['x'].loc[index])

            SibSpData = add_SibSp(SibSpData, next_SibSpID, AgeDiff['y'].loc[index])

            X_total['SibSpUnassigned'].loc[AgeDiff['x'].loc[index]] -= 1

            X_total['SibSpUnassigned'].loc[AgeDiff['y'].loc[index]] -= 1

    return X_total, SibSpData



# Identifies if the SibSp_extract consists of several SibSp groups equal by size and adds this data to SibSpData

def add_SibSp_from_group_total(X_total, SibSp_extract, SibSpData, index):

    SibSpCount = X_total['SibSpUnassigned'].loc[index].item()

    if SibSp_extract.shape[0] % (SibSpCount + 1) == 0:

        if SibSp_extract['Age'].isnull().any() and SibSpCount>1 and SibSp_extract.shape[0] > (X_total['SibSpUnassigned'].loc[index] + 1):

            print('Warning! No age data')

        SibSp_extract['AgeRank'] = SibSp_extract['Age'].rank(ascending=False, na_option='top')

        for i in range(0,SibSp_extract.shape[0]//(SibSpCount + 1)):

            next_SibSpID = get_next_SibSpID(SibSpData)

            group_to_add = SibSp_extract.loc[(SibSp_extract['AgeRank'] >= i*(SibSpCount + 1) + 1) & (SibSp_extract['AgeRank'] < (i + 1)*(row['SibSp'] + 1) + 1)]

            if not check_SibSp_relation(group_to_add.index.tolist(), SibSpData):

                continue

            for sibsp_pas_id in list(group_to_add.index.values):

                SibSpData = add_SibSp(SibSpData, next_SibSpID, sibsp_pas_id)

                X_total['SibSpUnassigned'].loc[sibsp_pas_id] -= SibSpCount

    return X_total, SibSpData
SibSpData = pd.DataFrame(columns=['SibSpID','PassengerId'])



# Adding new columns that will represent number of SibSp connections not yet identified

# In the start it will be the same as SibSp, every time we find a new connection we decrease it

# If we found all the connections it should be zero

X_total['SibSpUnassigned'] = X_total['SibSp']



extract_columns = ['PassengerId','Sex','Age','SibSp','SibSpUnassigned','FirstName','Title','LastName','Ticket']

# We iterate thruogh rows of X_total and try different heuristic rules to find SibSp for a given passenger

for index, row in X_total.iterrows():

    if X_total['SibSpUnassigned'].loc[index] > 0:

        # Selected by Last name and unassigned SibSp count as for current row

        SibSp_extract = X_total[extract_columns].loc[(X_total['LastName'] == row['LastName']) & (X_total['SibSpUnassigned'] == X_total['SibSpUnassigned'].loc[index])]

        if len(SibSp_extract.index) > 1:

            X_total, SibSpData = add_SibSp_from_group_total(X_total, SibSp_extract, SibSpData, index)

    if X_total['SibSpUnassigned'].loc[index] > 0 and pd.notnull(X_total['Ticket'].loc[index]):

        # Selected by Last name, Ticket number, and unassigned SibSp count as for current row

        SibSp_extract = X_total[extract_columns].loc[(X_total['LastName'] == row['LastName']) & (X_total['SibSpUnassigned'] == X_total['SibSpUnassigned'].loc[index]) & (X_total['Ticket'] == row['Ticket'])]

        if len(SibSp_extract.index) > 1:

            X_total, SibSpData = add_SibSp_from_group_total(X_total, SibSp_extract, SibSpData, index)

    if X_total['SibSpUnassigned'].loc[index] > 0 and pd.notnull(X_total['Ticket'].loc[index]) and pd.notnull(X_total['Age'].loc[index]):

        # Selected by Last name and Ticket number as for current row

        SibSp_extract = X_total[extract_columns].loc[(X_total['LastName'] == row['LastName']) & (X_total['Ticket'] == row['Ticket']) & (X_total['SibSpUnassigned'] > 0) & (pd.notnull(X_total['Age']))]

        if len(SibSp_extract.index) > 1:

            X_total, SibSpData = add_SibSp_from_group_by_age(X_total, SibSp_extract, SibSpData)

    if X_total['SibSpUnassigned'].loc[index] > 0:

        # Selected by Last name as for current row

        SibSp_extract = X_total[extract_columns].loc[(X_total['LastName'] == row['LastName']) & (X_total['SibSpUnassigned'] > 0) & (pd.notnull(X_total['Age']))]

        if len(SibSp_extract.index) > 1:

            X_total, SibSpData = add_SibSp_from_group_by_age(X_total, SibSp_extract, SibSpData)

    if X_total['SibSpUnassigned'].loc[index] > 0 and pd.notnull(X_total['Ticket'].loc[index]):

        # Selected by Ticket as for current row

        SibSp_extract = X_total[extract_columns].loc[(X_total['Ticket'] == row['Ticket']) & (X_total['SibSpUnassigned'] > 0) & (pd.notnull(X_total['Age']))]

        if len(SibSp_extract.index) > 1:

            X_total, SibSpData = add_SibSp_from_group_by_age(X_total, SibSp_extract, SibSpData)

    if X_total['SibSpUnassigned'].loc[index] > 0 and pd.notnull(X_total['Cabin'].loc[index]) and pd.notnull(X_total['Ticket'].loc[index]):

        # Selected by Cabin as for current row

        SibSp_extract = X_total[extract_columns].loc[(X_total['Ticket'] == row['Ticket']) & (X_total['Cabin'] == row['Cabin']) & (X_total['SibSpUnassigned'] > 0) & (pd.notnull(X_total['Age']))]

        if len(SibSp_extract.index) > 1:

            X_total, SibSpData = add_SibSp_from_group_by_age(X_total, SibSp_extract, SibSpData)
print('Passengers with some SibSp unassigned: ' + str(X_total['PassengerId'].loc[X_total['SibSpUnassigned'] > 0].count()) + ' with a total of ' +str(X_total['SibSpUnassigned'].sum()) + ' connections.')



print(X_total[['Sex','Age','SibSp','SibSpUnassigned','FirstName','Title','LastName','Ticket','Cabin']].loc[X_total['SibSpUnassigned'] > 0].sort_values(by=['LastName','SibSp','Age']))
# Define a dataframe with all passengers and their groups (SibSpIDs)

pass_sib_table = X_total[['PassengerId','Survived','Age']].merge(right=SibSpData, left_on='PassengerId', right_on='PassengerId', suffixes=('','_r'), how='left')

pass_sib_table['IsTrainSet'] = 1

pass_sib_table['IsTrainSet'].ix[pd.isnull(pass_sib_table['Survived'])] = 0

SibSp_count = pass_sib_table[['PassengerId','SibSpID','Survived']].groupby(by='SibSpID').count()

SibSp_sum =  pass_sib_table[['SibSpID','Survived']].groupby(by='SibSpID').sum()

SibSpStats = SibSp_count.merge(right=SibSp_sum, left_index=True, right_index=True, suffixes=('','_sum'), how='left')

SibSpStats.rename(columns={"PassengerId": "PassengerCount", "Survived": "TrainPassengerCount", "Survived_sum": "TrainSurvivedPassengerCount"}, inplace=True)



pass_sib_table = pass_sib_table.merge(right=SibSpStats, left_on='SibSpID', right_index=True, suffixes=('','_sibsp'), how='left')

pass_sib_table['SurvivedFilledZero'] = pass_sib_table['Survived']

pass_sib_table['SurvivedFilledZero'].fillna(0, inplace=True)

pass_sib_table['OtherTrainPassengerCount'] = pass_sib_table['TrainPassengerCount'] - pass_sib_table['IsTrainSet']

pass_sib_table['OtherTrainSurvivedPassengerCount'] = pass_sib_table['TrainSurvivedPassengerCount'] - pass_sib_table['IsTrainSet']*pass_sib_table['SurvivedFilledZero']

pass_sibsp_surv = pass_sib_table[['PassengerId','OtherTrainPassengerCount','OtherTrainSurvivedPassengerCount']].groupby(by='PassengerId').sum()



# Now we have in pass_sibsp_surv for every passenger the number of other passengers from SibSp groups that are

# in the train set and number of survived passengers out of them

# Now we define the final features

pass_sibsp_surv['OneSibSpSurvived'] = 0

pass_sibsp_surv['OneSibSpSurvived'].ix[(pass_sibsp_surv['OtherTrainPassengerCount'] > 0) & (pass_sibsp_surv['OtherTrainSurvivedPassengerCount'] > 0)] = 1

pass_sibsp_surv['OneSibSpNotSurvived'] = 0

pass_sibsp_surv['OneSibSpNotSurvived'].ix[(pass_sibsp_surv['OtherTrainPassengerCount'] > 0) & (pass_sibsp_surv['OtherTrainSurvivedPassengerCount'] < pass_sibsp_surv['OtherTrainPassengerCount'])] = 1

pass_sibsp_surv.drop(labels = ['OtherTrainPassengerCount','OtherTrainSurvivedPassengerCount'], axis = 1, inplace=True)



# Joining the new features and corrected age with initial data

X_total = X_total.merge(right=pass_sibsp_surv, left_index=True, right_index=True, suffixes=('','_sibsp'), how='left')



# Let's calculate mean age of other passengers within a SibSp group to then use it for filling missing data

SibSp_mean = pass_sib_table[['SibSpID','Age']].groupby(by='SibSpID').mean()

pass_sib_table = pass_sib_table.merge(right=SibSp_mean, left_on='SibSpID', right_index=True, suffixes=('','_sibsp'), how='left')

pass_sibsp_age = pass_sib_table[['PassengerId','Age_sibsp']].groupby(by='PassengerId').mean()

X_total = X_total.merge(right=pass_sibsp_age, left_index=True, right_index=True, suffixes=('','_sibspage'), how='left')
print('# of passengers in total sample')

print(pd.crosstab(index=X_total['OneSibSpSurvived'], columns=X_total['OneSibSpNotSurvived']))

print('\n# of passengers in train sample')

print(pd.crosstab(index=X_total['OneSibSpSurvived'], columns=X_total['OneSibSpNotSurvived'], values=X_total['Survived'], aggfunc='count'))

print('\n% of survived passengers')

print(pd.crosstab(index=X_total['OneSibSpSurvived'], columns=X_total['OneSibSpNotSurvived'], values=X_total['Survived'], aggfunc='mean'))
print('Null values before:')

print(X_total[X_total.columns[X_total.isnull().any()].tolist()].isnull().sum())



X_train = X_total.loc[pd.notnull(X_total['Survived'])]

X_test = X_total.loc[pd.isnull(X_total['Survived'])]



# Filling missing age. First priority is mean age within SibSp groups, second is mean age within title

title_mapping = pd.DataFrame.from_dict({'Capt':'Mr', 'Col':'Mr', 'Countess':'Mrs', 'Don':'Mr', 'Dona':'Mrs', 'Dr':'Dr', 'Jonkheer':'Mr', 'Lady':'Mrs', 'Major':'Mr', 'Master':'Master', 'Miss':'Miss', 'Mlle':'Miss', 'Mme':'Miss', 'Mr':'Mr', 'Mrs':'Mrs', 'Ms':'Miss', 'Rev':'Mr', 'Sir':'Mr'}, orient='index')

title_mapping.columns = ['TitleGroup']



X_train = X_train.merge(right=title_mapping, left_on='Title', right_index=True, suffixes=('','_title'), how='left')

X_total = X_total.merge(right=title_mapping, left_on='Title', right_index=True, suffixes=('','_title'), how='left')

title_age = X_train[['Age','TitleGroup']].groupby('TitleGroup').mean()

X_total = X_total.merge(right=title_age, left_on='TitleGroup', right_index=True, suffixes=('','_title'), how='left')



X_total['Age'].fillna(X_total['Age_sibsp'], inplace=True)

X_total['Age'].fillna(X_total['Age_title'], inplace=True)



# Filling one row of missing Fare by mean fare within Plcass

Pclass_Fare = X_train[['Pclass','Fare']].groupby('Pclass').mean()

X_total = X_total.merge(right=Pclass_Fare, left_on='Pclass', right_index=True, suffixes=('','_pclass'), how='left')



X_total['Fare'].fillna(X_total['Fare_pclass'], inplace=True)



# Filling two missing values of Embarked with most frequent

X_total['Embarked'].loc[X_total['Embarked'].isnull()] = X_train['Embarked'].value_counts().index[0]



print('Null values left (check):')

print(X_total[X_total.columns[X_total.isnull().any()].tolist()].isnull().sum())
X_total['IsAlone'] = 0

X_total['IsAlone'].loc[X_total['SibSp'] + X_total['Parch'] == 0] = 1



X_total['IsSmallFamily'] = 0

X_total['IsSmallFamily'].loc[X_total['SibSp'] + X_total['Parch'] < 5] = 1



X_total['IsLargeFamily'] = 0

X_total['IsLargeFamily'].loc[X_total['SibSp'] + X_total['Parch'] >= 5] = 1



# Dropping columns that I'm not going to use in my model further

X_total.drop(labels = ['SibSpUnassigned','TitleGroup','SibSp','Parch','Survived','Ticket','Cabin','Age_sibsp','Fare_pclass','Age_title','Title','Name','FirstName','LastName','PassengerId'], axis = 1, inplace=True)



# Pclass is not a number but a category

X_total['Pclass'] = X_total['Pclass'].astype('category')



# Dummy coding categorical variables

X_total = pd.get_dummies(X_total)



# Splitting train and test

X_train = X_total.loc[X_total.index < 892]

X_test = X_total.loc[X_total.index >= 892]



# X without the created variables to see impact from adding them

X_train_wo_sibsp = X_train.drop(labels = ['OneSibSpSurvived','OneSibSpNotSurvived'], axis = 1)

X_test_wo_sibsp = X_test.drop(labels = ['OneSibSpSurvived','OneSibSpNotSurvived'], axis = 1)
from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

import time

import datetime

cv = KFold(n_splits=5, shuffle=True, random_state=91)

grid = {'n_estimators': np.asarray([10, 20, 40, 60]), 'max_depth': np.asarray([2, 3, 4, 5])}

clf = GradientBoostingClassifier(random_state=91)

gs = GridSearchCV(estimator=clf, param_grid=grid, scoring='accuracy', cv=cv)



# Training two models with and without varialbes obtained above

gs.fit(X_train, y_train)

print('Accuracy with SibSp survaval data\n%.2f' % (gs.best_score_*100))

clf_best_withsibsp = gs.best_estimator_



gs.fit(X_train_wo_sibsp, y_train)

print('Accuracy without SibSp survaval data\n%.2f' % (gs.best_score_*100))

clf_best_wosibsp = gs.best_estimator_
np.savetxt(fname='titanic_predict_withsibsp.csv', X=np.vstack((X_test.index.values,clf_best_withsibsp.predict(X_test))).T, delimiter=',', fmt=['%d','%d'], header='PassengerId,Survived', comments='')

np.savetxt(fname='titanic_predict_wosibsp.csv', X=np.vstack((X_test.index.values,clf_best_wosibsp.predict(X_test_wo_sibsp))).T, delimiter=',', fmt=['%d','%d'], header='PassengerId,Survived', comments='')