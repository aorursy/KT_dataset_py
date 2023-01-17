import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('dark')

sns.set_palette('Set2')

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

data = pd.concat([train_data, test_data]).reset_index().drop(['index'], axis=1)
data['Surname'] = data['Name'].apply(lambda x: x.split(',')[0])
# New Ticket_id column

data['Ticket_id'] = 'new_col'

# Initialize Ticket_id = Pclass + Ticket + Fare + Embarked

def ticket_id(row):

    row['Ticket_id'] = str(row.Pclass) + '-' + str(row.Ticket)[:-1] + '-' + str(row.Fare) + '-' + str(row.Embarked)

    return row



data = data.apply(ticket_id, axis='columns')
# New Group_id column

data['Group_id'] = 'new_col2'

# Initialize Group_id = Surname + Ticket_id

def group_id(row):

    row['Group_id'] = str(row.Surname) + '-' + str(row.Ticket_id)

    return row



data = data.apply(group_id, axis='columns')
# creation of the Title feature

data['Title'] = 'man'

data.loc[data.Sex == 'female', 'Title'] = 'woman'

data.loc[data['Name'].str.contains('Master'), 'Title'] = 'boy'
data.loc[data.Title == 'man', 'Group_id'] = 'noGroup'

# New column with WC frequency

data['WC_count'] = data.loc[data.Title != 'man'].groupby('Group_id')['Group_id'].transform('count')

# assign noGroup to every unique value

data.loc[data.WC_count <=1, 'Group_id'] = 'noGroup'
cols = ['PassengerId', 'Survived', 'Name', 'Title', 'Ticket_id','Group_id']

data.loc[(data.Ticket_id == '1-1696-134.5-C') & (data.Title != 'man'), cols]
indices = []

count = 0

for i in range(0,1309):

    if (data.loc[i,'Title'] != 'man') & (data.loc[i,'Group_id'] == 'noGroup'):

        data.loc[i,'Group_id'] = data.loc[(data['Ticket_id'] == data.loc[i, 'Ticket_id']) & (data.Title != 'man'), 'Group_id'].iloc[0]

        if (data.loc[i, 'Group_id'] != 'noGroup'):

            indices.append(i)

            count += 1

print('{:d} passengers were added to an existing group'.format(count))
cols = ['PassengerId', 'Survived', 'Name', 'Title', 'Group_id']

data.loc[indices, cols]
number_of_groups = data.loc[data.Group_id != 'noGroup', 'Group_id'].nunique()

print('Number of groups found: {:d}'.format(number_of_groups))

number_of_WCG_passengers = data.loc[data.Group_id != 'noGroup', 'Group_id'].count()

print('\nNumber of passengers in a group: {:d}'.format(number_of_WCG_passengers))

composition = data.loc[data.Group_id != 'noGroup','Title'].value_counts()

print('\nComposition of the groups:')

print(composition.to_string())
data['WCSurvived'] = data.loc[(data.Title != 'man') & (data.Group_id != 'noGroup')].groupby('Group_id').Survived.transform('mean')
cols = ['PassengerId', 'Survived', 'WCSurvived', 'Name', 'Title', 'Group_id']

data.loc[data.Group_id == 'Sage-3-CA. 234-69.55-S', cols]
print('WCSurvived all data values:')

print(data.WCSurvived.value_counts().to_string())

plt.figure(figsize=(7,5))

f = sns.countplot(y=data.WCSurvived)
data.loc[(data.WCSurvived==0.75) | (data.WCSurvived==0.5), cols].sort_values(by='Group_id')
# Get the family names using set difference

test_groups = set(data[891:1309].Group_id.unique()) - set(data[0:891].Group_id.unique())

data.loc[data.Group_id.isin(test_groups), cols].sort_values(by='Group_id')
fig, ax = plt.subplots(1,2,figsize=(12,6))

fig.suptitle('Woman-child-groups analysis', fontsize=14)

a = sns.barplot(x='Pclass', y='Survived', data=data[data.Group_id != 'noGroup'], ax=ax[0]).set_ylabel('Survival rate')

b = sns.barplot(x='Embarked', y='Survived', data=data[data.Group_id != 'noGroup'], ax=ax[1]).set_ylabel('Survival rate')
# Assign WCSurvived = 0 to 3rd class test families, else 1

data.loc[data.Group_id.isin(test_groups), 'WCSurvived'] = 0

data.loc[(data.Group_id.isin(test_groups)) & (data.Pclass != 3), 'WCSurvived'] = 1
print('WCSurvived test values:')

print(data[891:1309].WCSurvived.value_counts().to_string())
# Set everyone to 0

data.loc[891:1308, 'Predict'] = 0

# Set women to 1, completing the gender model

data.loc[891:1308, 'Predict'][(data.Sex == 'female')] = 1

# Change WCG women with WCSurvived=0 to 0

data.loc[891:1308,'Predict'][(data.Sex == 'female') & (data['WCSurvived'] == 0)] = 0

# Change WCG boys with WCSurvived=1 to 1, completing the WCG + gender model

data.loc[891:1308, 'Predict'][(data.Title == 'boy') & (data['WCSurvived'] == 1)] = 1

# With this, the three group members with non-integer WCSurvived are not changed from the gender model
print('The following 8 males are predicted to live:')

cols = ['PassengerId', 'Name', 'Title', 'Group_id']

data[891:1309][cols].loc[(data.Title == 'boy') & (data.Predict == 1)]
print('The following 15 females are predicted to die:')

data[891:1309][cols].loc[(data.Title == 'woman') & (data.Predict == 0)]
print('The remaining 258 males are predicted to die')

print('and the remaining 137 females are predicted to live')
output = pd.DataFrame({'PassengerId': data[891:1309].PassengerId, 'Survived': data[891:1309].Predict.astype('int')})

output.to_csv('WCG_gender.csv', index=False)

print('WCG_gender submission was successfully saved!')

print('Submission is loading... you scored 81,6%!')
# Assign np.NaN to zero-fares

def fix_fare(row):

    if row.Fare == 0:

        row.Fare = np.NaN

    return row

print('The following {:d} passengers have a zero Fare:'.format(data[data.Fare==0].shape[0]))

cols = ['PassengerId', 'Survived', 'Pclass','Fare', 'Name']

data.loc[data.Fare==0, cols]
fig, ax = plt.subplots(1,2,figsize=(12,8))

fig.suptitle('Removing zero fares: before and after', fontsize=14)

a = sns.swarmplot(x='Pclass', y='Fare', data=data, ax=ax[0])

ax[0].axhline(y=2, color='r')

# Apply the fix_fare function 

data = data.apply(fix_fare, axis='columns')

ax[1].axhline(y=2, color='r')

b = sns.swarmplot(x='Pclass', y='Fare', data=data, ax=ax[1])
# Calculate Ticket frequency and divide Fare by it

data['Ticket_freq'] = data.groupby('Ticket')['Ticket'].transform('count')

data['Pfare'] = data['Fare'] / data['Ticket_freq']
fig, ax = plt.subplots(1,2,figsize=(12,8))

fig.suptitle('Fare and Pfare compared', fontsize=14)

a = sns.swarmplot(x='Pclass', y='Fare', data=data, ax=ax[0])

b = sns.swarmplot(x='Pclass', y='Pfare', data=data, ax=ax[1])
# Isolating adult males in train and test set

train_male = data[0:891].loc[(data.Sex=='male') & (data.WCSurvived.isnull())]

test_male = data[891:1309].loc[(data.Sex=='male') & (data.WCSurvived.isnull())]
fig, ax = plt.subplots(2,2,figsize=(12,12))

fig.suptitle('Adult males EDA', fontsize=14)

sns.barplot(x='Pclass', y='Survived', data=train_male, ax=ax[0][0])

ax[0][0].axhline(y=train_male.Survived.mean(), color='r')

sns.barplot(x='Embarked', y='Survived', data=train_male, ax=ax[0][1])

ax[0][1].axhline(y=train_male.Survived.mean(), color='r')

sns.swarmplot(x='Pclass', y='Pfare', hue='Survived', data=train_male, ax=ax[1][0])

ax[1][0].axhline(y=25, color='y')

ax[1][0].axhline(y=32, color='y')

a = sns.swarmplot(y='Age', x='Pclass', hue='Survived', data=train_male, ax=ax[1][1])
x1 = train_male.loc[train_male['Survived']==1, 'Pfare']

x0 = train_male.loc[train_male['Survived']==0, 'Pfare']

y1 = train_male.loc[train_male['Survived']==1, 'Age']

y0 = train_male.loc[train_male['Survived']==0, 'Age']



fig, ax = plt.subplots(1,2,figsize=(12,6))

fig.suptitle('Age and Pfare distributions with hue Survived', fontsize=14)

sns.distplot(x1, bins=30, label = 'Survived', ax = ax[0], color = 'c')

sns.distplot(x0, bins=25, label = 'Not survived', ax = ax[0], color = 'y')

ax[0].set_xlim(-5, 70)

ax[0].legend()

sns.distplot(y1, bins=20, label = 'Survived', ax = ax[1], color = 'g')

sns.distplot(y0, bins=20, label = 'Not survived', ax = ax[1], color = 'r')

ax[1].legend()

fig.show()
cols = ['PassengerId', 'Name', 'Pfare', 'Pclass', 'Embarked']

y_m = train_male['Survived']

features = ['Pfare', 'Pclass', 'Embarked']

X_m = train_male[features]



numerical_cols = ['Pfare']

categorical_cols = ['Pclass', 'Embarked']



numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer()),

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, numerical_cols),

    ('cat', categorical_transformer, categorical_cols)

])



precision_m = []

recall_m = []



for k in range(1,18):

    pipeline1 = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', KNeighborsClassifier(n_neighbors=k))

    ])

    precision_m.append(cross_val_score(pipeline1, X_m, y_m, cv=15, n_jobs=-1, scoring='precision').mean())

    recall_m.append(cross_val_score(pipeline1, X_m, y_m, cv=15, n_jobs=-1, scoring='recall').mean())

    

k_range = range(1,18)

plt.figure(figsize=(7,5))

plt.plot(k_range, precision_m, label='15-fold precision')

plt.plot(k_range, recall_m, label='15-fold recall')

plt.axhline(y=0.5, color='r')

plt.xlabel('Value of k for KNN')

plt.title('Precision and recall by number of neighbors', fontsize=14)

plt.legend()

plt.show()
m1 = KNeighborsClassifier(n_neighbors=1)

m2 = KNeighborsClassifier(n_neighbors=3)

m3 = KNeighborsClassifier(n_neighbors=7)

# Preprocessing is the same as before

male_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('voting',VotingClassifier([

        ('m1', m1), ('m2', m2), ('m3', m3)]))

])

print('15-fold precision of the ensemble: {:.3f}'.format(

    cross_val_score(male_pipeline, X_m, y_m, cv=15, n_jobs=-1, scoring='precision').mean()))

print('15-fold recall of the ensemble: {:.3f}'.format(

    cross_val_score(male_pipeline, X_m, y_m, cv=15, n_jobs=-1, scoring='recall').mean()))

print('15-fold accuracy of the ensemble: {:.3f}'.format(

    cross_val_score(male_pipeline, X_m, y_m, cv=15, n_jobs=-1).mean()))

# Fit model and make predictions

male_pipeline.fit(X_m, y_m)

learn_train_m = male_pipeline.predict(X_m)

X_test_m = test_male[features]

predictions_m = male_pipeline.predict(X_test_m)

print('\nThe following 9 adult males are predicted to live:')

test_male.loc[(predictions_m==1), cols]
fig, ax = plt.subplots(1,3,figsize=(15,8))

fig.suptitle('Fun comparison of train set vs test set', fontsize=14)

ax[0].set_title('Real train set')

ax[0].set_ylim(top=60)

sns.swarmplot(x=X_m.Pclass, y=X_m.Pfare, hue=y_m, ax=ax[0])

ax[1].set_title('Ensemble learns the train set')

ax[1].set_ylim(top=60)

sns.swarmplot(x=X_m.Pclass, y=X_m.Pfare, hue=learn_train_m,  ax=ax[1])

ax[2].set_title('Ensemble predicts the test set')

ax[2].set_ylim(top=60)

a = sns.swarmplot(x=test_male.Pclass, y=test_male.Pfare, hue=predictions_m,  ax=ax[2])
data.loc[891:1308, 'Predict'][(data.Sex=='male') & (data.WCSurvived.isnull())] = predictions_m

output = pd.DataFrame({'PassengerId': data[891:1309].PassengerId, 'Survived': data[891:1309].Predict.astype('int')})

output.to_csv('WCG_male.csv', index=False)

print('WCG_male submission was successfully saved!')

print('Submission is loading... you scored 82,3%!')
train_female = data[0:891].loc[(data.Sex=='female')  & (data.WCSurvived.isnull())]

test_female = data[891:1309].loc[(data.Sex=='female') & (data.WCSurvived.isnull())]
fig, ax = plt.subplots(2,2,figsize=(12,12))

fig.suptitle('Non-WCG females EDA', fontsize=14)

sns.barplot(x='Pclass', y='Survived', data=train_female, ax=ax[0][0])

ax[0][0].axhline(y=train_female.Survived.mean(), color='r')

sns.barplot(x='Embarked', y='Survived', data=train_female, ax=ax[0][1])

ax[0][1].axhline(y=train_female.Survived.mean(), color='r')

sns.swarmplot(x='Pclass', y='Pfare', hue='Survived', data=train_female, ax=ax[1][0])

ax[1][0].set_ylim(top=70)

ax[1][0].axhline(y=7, color='y')

ax[1][0].axhline(y=10, color='y')

a = sns.swarmplot(y='Age', x='Pclass', hue='Survived', data=train_female, ax=ax[1][1])
w1 = train_female.loc[train_female['Survived']==1, 'Pfare']

w0 = train_female.loc[train_female['Survived']==0, 'Pfare']

z1 = train_female.loc[train_female['Survived']==1, 'Age']

z0 = train_female.loc[train_female['Survived']==0, 'Age']



fig, ax = plt.subplots(1,2,figsize=(12,6))

fig.suptitle('Age and Pfare distributions with hue Survived', fontsize=14)

sns.distplot(w1, bins=35, label = 'Survived', ax = ax[0], color = 'c')

sns.distplot(w0, bins=15, label = 'Not survived', ax = ax[0], color = 'y')

ax[0].set_xlim(-5, 60)

ax[0].legend()

sns.distplot(z1, bins=12, label = 'Survived', ax = ax[1], color = 'g')

sns.distplot(z0, bins=10, label = 'Not survived', ax = ax[1], color = 'r')

ax[1].legend()

fig.show()
from sklearn.metrics import make_scorer, precision_score, recall_score

# We set zero_division=0 to avoid raising errors

custom_precision = make_scorer(precision_score, pos_label=0, zero_division=0)

custom_recall = make_scorer(recall_score, pos_label=0)
y_f = train_female['Survived']

X_f = train_female[features]

precision_f = []

recall_f = []

# Preprocessing is always the same...

for k in range(1,18):

    pipeline2 = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', KNeighborsClassifier(n_neighbors=k))

    ])

    # We use 9-fold because the train size is smaller

    # and 198/9 = integer

    precision_f.append(cross_val_score(pipeline2, X_f, y_f, cv=9, n_jobs=-1, scoring=custom_precision).mean())

    recall_f.append(cross_val_score(pipeline2, X_f, y_f, cv=9, n_jobs=-1, scoring=custom_recall).mean())

    

plt.figure(figsize=(7,5))

plt.plot(k_range, precision_f, label='9-fold precision')

plt.plot(k_range, recall_f, label='9-fold recall')

plt.axhline(y=0.5, color='r')

plt.xlabel('Value of k for KNN')

plt.title('Precision and recall by number of neighbors', fontsize=14)

plt.legend()

plt.show()
f1 = KNeighborsClassifier(n_neighbors=4)

f2 = KNeighborsClassifier(n_neighbors=9)

f3 = KNeighborsClassifier(n_neighbors=11)

# Preprocessing pipelines are the same as before

female_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('voting', VotingClassifier([

        ('f1', f1), ('f2', f2), ('f3', f3)]))

])

print('9-fold precision of the ensemble: {:.3f}'.format(

    cross_val_score(female_pipeline, X_f, y_f, cv=9, scoring=custom_precision).mean()))

print('9-fold recall of the ensemble: {:.3f}'.format(

    cross_val_score(female_pipeline, X_f, y_f, cv=9, scoring=custom_recall).mean()))

print('9-fold accuracy of the ensemble: {:.3f}'.format(

    cross_val_score(female_pipeline, X_f, y_f, cv=9).mean()))

# Preprocessing of training data, fit model

female_pipeline.fit(X_f, y_f)

learn_train_f = female_pipeline.predict(X_f)

X_test_f = test_female[features]

predictions_f = female_pipeline.predict(X_test_f)

print('\nThe following 6 non-WCG females are predicted to die:')

test_female.loc[(predictions_f==0), cols]
fig, ax = plt.subplots(1,3,figsize=(15,8))

fig.suptitle('Fun comparison of train set vs test set', fontsize=14)

ax[0].set_title('Real train set')

ax[0].set_ylim(top=55)

sns.swarmplot(x=X_f.Pclass, y=X_f.Pfare, hue=y_f, ax=ax[0])

ax[1].set_title('Ensemble learns the train set')

ax[1].set_ylim(top=55)

sns.swarmplot(x=X_f.Pclass, y=X_f.Pfare, hue=learn_train_f,  ax=ax[1])

ax[2].set_title('Ensemble predicts the test set')

ax[2].set_ylim(top=55)

a = sns.swarmplot(x=test_female.Pclass, y=test_female.Pfare, hue=predictions_f,  ax=ax[2])
data.loc[891:1308, 'Predict'][(data.Sex=='female') & (data.WCSurvived.isnull())] = predictions_f

output = pd.DataFrame({'PassengerId': data[891:1309].PassengerId, 'Survived': data[891:1309].Predict.astype('int')})

output.to_csv('WCG_male_female.csv', index=False)

print('WCG_male_female was successfully saved!')

print('Submission is loading... you scored 82,8%!')