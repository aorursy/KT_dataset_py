import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest, chi2





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')

dataset.head()
dataset.info()
dataset.describe()
dataset.shape
# rename invalid column names

dataset = dataset.rename(columns={'CRIMINAL\nCASES': 'CRIMINAL_CASES', 'GENERAL\nVOTES': 'GENERAL_VOTES', 'POSTAL\nVOTES': 'POSTAL_VOTES', 'TOTAL\nVOTES': 'TOTAL_VOTES', 'OVER TOTAL ELECTORS \nIN CONSTITUENCY': 'OVER_TOTAL_ELECTORS_IN_CONSTITUENCY', 'OVER TOTAL VOTES POLLED \nIN CONSTITUENCY': 'OVER_TOTAL_VOTES_POLLED_IN_CONSTITUENCY', 'TOTAL ELECTORS': 'TOTAL_ELECTORS'})



# drop rows with NA values

dataset = dataset[dataset['GENDER'].notna()]



# replace Nil values with 0

dataset['ASSETS'] = dataset['ASSETS'].replace(['Nil', '`', 'Not Available'], '0')

dataset['LIABILITIES'] = dataset['LIABILITIES'].replace(['NIL', '`', 'Not Available'], '0')

dataset['CRIMINAL_CASES'] = dataset['CRIMINAL_CASES'].replace(['Not Available'], '0')



# clean ASSETS and LIABILITIES column values

dataset['ASSETS'] = dataset['ASSETS'].map(lambda x: x.lstrip('Rs ').split('\n')[0].replace(',', ''))

dataset['LIABILITIES'] = dataset['LIABILITIES'].map(lambda x: x.lstrip('Rs ').split('\n')[0].replace(',', ''))



# convert ASSETS, LIABILITIES and CRIMINAL_CASES column values into numeric

dataset['ASSETS'] = dataset['ASSETS'].astype(str).astype(float)

dataset['LIABILITIES'] = dataset['LIABILITIES'].astype(str).astype(float)

dataset['CRIMINAL_CASES'] = dataset['CRIMINAL_CASES'].astype(str).astype(int)



# reorder columns

cols = dataset.columns.tolist()

cols = cols[0:3] + cols[4:] + cols[3:4]

dataset = dataset[cols]
dataset.head()



dataset.isna().sum()
dataset['PARTY'].value_counts()
# change party of the less frequent parties as Other

# 'BJP','INC','IND','BSP', 'CPI(M)', 'AITC', 'MNM': high frequent

# 'TDP', 'VSRCP', 'SP', 'DMK', 'BJD': medium frequent



dataset.loc[~dataset["PARTY"].isin(['BJP','INC','IND','BSP', 'CPI(M)', 'AITC', 'MNM', 'TDP', 'VSRCP', 'SP', 'DMK', 'BJD']), "PARTY"] = "Other"

dataset['PARTY'].value_counts()

dataset['CATEGORY'].value_counts()
dataset['EDUCATION'].value_counts()
# encode education column

encoded_edu = []



# iterate through each row in the dataset

for row in dataset.itertuples():

    education = row.EDUCATION



    if education == "Illiterate":

        encoded_edu.append(0)

    elif education == "Literate":

        encoded_edu.append(1)

    elif education == "5th Pass":

        encoded_edu.append(2)

    elif education == "8th Pass":

        encoded_edu.append(3)

    elif education == "10th Pass":

        encoded_edu.append(4)

    elif education == "12th Pass":

        encoded_edu.append(7)

    elif education == "Graduate":

        encoded_edu.append(8)

    elif education == "Post Graduate":

        encoded_edu.append(9)

    elif education == "Graduate Professional":

        encoded_edu.append(10)

    elif education == "Doctorate":

        encoded_edu.append(11)

    else:

        encoded_edu.append(5)



dataset['EDUCATION'] = encoded_edu

# Preparing feature values



cons_per_state = {}

voters_per_state = {}



party_winningSeats = {}

party_criminal = {}

party_education = {}



party_totalCandidates_per_cons = {}

party_winningSeats_per_cons = {}

party_criminal_per_cons = {}

party_education_per_cons = {}



voters_per_cons = {}





# group by state

subset = dataset[['STATE', 'CONSTITUENCY', 'TOTAL_ELECTORS']]

gk = subset.groupby('STATE')



# for each state

for name,group in gk:

    # total constituencies per state

    cons_per_state[name] = len(group)

    

    # total voters per state

    voters_per_state[name] = group['TOTAL_ELECTORS'].sum()





# group by party

subset = dataset[['PARTY', 'CONSTITUENCY', 'CRIMINAL_CASES', 'EDUCATION', 'WINNER']]

gk = subset.groupby('PARTY')



# for each party

for name,group in gk:

    # winning seats by party

    party_winningSeats[name] = group[group['WINNER'] == 1.0].shape[0]

    

    # criminal cases by party

    party_criminal[name] = group['CRIMINAL_CASES'].sum()

    

    # education qualification by party (sum of candidates)

    party_education[name] = group['EDUCATION'].sum()

    

    # group by constituency

    gk2 = group.groupby('CONSTITUENCY')

    

    # for each constituency

    for name2, group2 in gk2:

        key = name2 + '_' + name    # cons_party

        

        # total candidates by party in constituency

        party_totalCandidates_per_cons[key] = len(group2)

        

        # party winning seats in the constituency

        party_winningSeats_per_cons[key] = group2[group2['WINNER'] == 1.0].shape[0]

        

        # criminal cases by party in the constituency

        party_criminal_per_cons[key] = group2['CRIMINAL_CASES'].sum()



        # education qualification by party in constituency (sum of candidates)

        party_education_per_cons[key] = group2['EDUCATION'].sum()





# Total voters per constituency

subset = dataset[['CONSTITUENCY', 'TOTAL_ELECTORS']]

gk = subset.groupby('CONSTITUENCY')



# for each constituency

for name,group in gk:

    voters_per_cons[name] = len(group)

# Applying feature values



# new feature columns

total_cons_per_state = []

total_voters_per_state = []

total_voters_per_cons = []



winning_seats_by_party = []

criminal_by_party = []

education_by_party = []



total_candidates_by_party_per_cons = []

winning_seats_by_party_per_cons = []

criminal_by_party_per_cons = []

education_by_party_per_cons = []





# iterate through each row in the dataset

for row in dataset.itertuples():

    subkey = row.CONSTITUENCY + '_' + row.PARTY



    total_cons_per_state.append(cons_per_state.get(row.STATE))

    total_voters_per_state.append(voters_per_state.get(row.STATE))

    total_voters_per_cons.append(voters_per_cons.get(row.CONSTITUENCY))

    winning_seats_by_party.append(party_winningSeats.get(row.PARTY))

    criminal_by_party.append(party_criminal.get(row.PARTY))

    education_by_party.append(party_education.get(row.PARTY))

    total_candidates_by_party_per_cons.append(party_totalCandidates_per_cons.get(subkey))

    winning_seats_by_party_per_cons.append(party_winningSeats_per_cons.get(subkey))

    criminal_by_party_per_cons.append(party_criminal_per_cons.get(subkey))

    education_by_party_per_cons.append(party_education_per_cons.get(subkey))





# append columns to dataset

dataset['total_cons_per_state'] = total_cons_per_state

dataset['total_voters_per_state'] = total_voters_per_state

dataset['total_voters_per_cons'] = total_voters_per_cons

dataset['winning_seats_by_party'] = winning_seats_by_party

dataset['criminal_by_party'] = criminal_by_party

dataset['education_by_party'] = education_by_party

dataset['total_candidates_by_party_per_cons'] = total_candidates_by_party_per_cons

dataset['winning_seats_by_party_per_cons'] = winning_seats_by_party_per_cons

dataset['criminal_by_party_per_cons'] = criminal_by_party_per_cons

dataset['education_by_party_per_cons'] = education_by_party_per_cons

# label encode categorical columns



lblEncoder_state = LabelEncoder()

lblEncoder_state.fit(dataset['STATE'])

dataset['STATE'] = lblEncoder_state.transform(dataset['STATE'])



lblEncoder_cons = LabelEncoder()

lblEncoder_cons.fit(dataset['CONSTITUENCY'])

dataset['CONSTITUENCY'] = lblEncoder_cons.transform(dataset['CONSTITUENCY'])



lblEncoder_name = LabelEncoder()

lblEncoder_name.fit(dataset['NAME'])

dataset['NAME'] = lblEncoder_name.transform(dataset['NAME'])



lblEncoder_party = LabelEncoder()

lblEncoder_party.fit(dataset['PARTY'])

dataset['PARTY'] = lblEncoder_party.transform(dataset['PARTY'])



lblEncoder_symbol = LabelEncoder()

lblEncoder_symbol.fit(dataset['SYMBOL'])

dataset['SYMBOL'] = lblEncoder_symbol.transform(dataset['SYMBOL'])



lblEncoder_gender = LabelEncoder()

lblEncoder_gender.fit(dataset['GENDER'])

dataset['GENDER'] = lblEncoder_gender.transform(dataset['GENDER'])



lblEncoder_category = LabelEncoder()

lblEncoder_category.fit(dataset['CATEGORY'])

dataset['CATEGORY'] = lblEncoder_category.transform(dataset['CATEGORY'])

# scaling values into 0-1 range



scaler = MinMaxScaler(feature_range=(0, 1))

features = [

    'STATE', 'CONSTITUENCY', 'NAME', 'PARTY', 'SYMBOL', 'GENDER', 'CRIMINAL_CASES', 'AGE', 'CATEGORY', 'EDUCATION', 'ASSETS', 'LIABILITIES', 'GENERAL_VOTES', 'POSTAL_VOTES', 'TOTAL_VOTES', 'OVER_TOTAL_ELECTORS_IN_CONSTITUENCY', 'OVER_TOTAL_VOTES_POLLED_IN_CONSTITUENCY', 'TOTAL_ELECTORS',

     'total_cons_per_state', 'total_voters_per_state', 'total_voters_per_cons', 'winning_seats_by_party', 'criminal_by_party', 'education_by_party', 'total_candidates_by_party_per_cons', 'winning_seats_by_party_per_cons', 'criminal_by_party_per_cons', 'education_by_party_per_cons'

]



dataset[features] = scaler.fit_transform(dataset[features])

# separate train features and label

y = dataset["WINNER"]

X = dataset.drop(labels=["WINNER"], axis=1)

# apply SelectKBest class to extract top most features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X, y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



# concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns, dfscores], axis=1)

featureScores.columns = ['Specs', 'Score']

print(featureScores.nlargest(30, 'Score'))

# remove unnecessary columns



X.drop(labels=["NAME"], axis=1, inplace=True)

X.drop(labels=["SYMBOL"], axis=1, inplace=True)

X.drop(labels=["POSTAL_VOTES"], axis=1, inplace=True)

X.drop(labels=["GENERAL_VOTES"], axis=1, inplace=True)



X.drop(labels=["TOTAL_ELECTORS"], axis=1, inplace=True)

X.drop(labels=["STATE"], axis=1, inplace=True)

X.drop(labels=["CONSTITUENCY"], axis=1, inplace=True)

X.drop(labels=["GENDER"], axis=1, inplace=True)

X.drop(labels=["criminal_by_party_per_cons"], axis=1, inplace=True)

X.drop(labels=["total_voters_per_state"], axis=1, inplace=True)

X.drop(labels=["CRIMINAL_CASES"], axis=1, inplace=True)

X.drop(labels=["total_cons_per_state"], axis=1, inplace=True)

X.drop(labels=["EDUCATION"], axis=1, inplace=True)

X.drop(labels=["education_by_party_per_cons"], axis=1, inplace=True)

X.drop(labels=["AGE"], axis=1, inplace=True)

# split dataset into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
knn.predict(X_test)

print("Testing Accuracy is: ", knn.score(X_test, y_test)*100, "%")
figsize=(18,14)

fig, ax = plt.subplots(figsize=figsize)

sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)



## just to visualize correlated features