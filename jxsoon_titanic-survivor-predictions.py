# import the necessary libraries for data preprocessing

import numpy as np

import pandas as pd



# import libraries for viz

import matplotlib.pyplot as plt

import seaborn as sns



# Set preferred style for visualization and dataframe display

%matplotlib inline

sns.set(style='darkgrid')

pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', None)



# ignore warnings library

import warnings

warnings.filterwarnings("ignore")
# Load the datasets

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.skew()
# Preview of the train dataset

train.head(20)
# Statistics summary of the train dataset

train.describe()
# Display the brief info of the dataset (showing shape, null values, and types)

train.info()
test.info()
# Check for the survival rate of the passengers and total number of survivors

survived_rate = train['Survived'].mean()

survived_count = train['Survived'].value_counts().values[1]



print('The survival rate of titanic passengers: {:.2f}%'.format(survived_rate*100))

print('The number of survivors: {}'.format(survived_count))
# Quick look at  male and female survival rates

male_df = train[train['Sex']=='male']

female_df= train[train['Sex']=='female']

male_survived = male_df['Survived'].mean()

female_survived = female_df['Survived'].mean()

# We can classify children as passengers with age <7

child_df = train[train['Age'] <7]

child_survived = child_df['Survived'].mean()

# We can classify old people as passengers with age >60

senior_df = train[train['Age'] >60]

senior_survived = senior_df['Survived'].mean()

print('Male survival rate: {:.2f}%'.format(male_survived*100))

print('Female survival rate: {:.2f}%'.format(female_survived*100))

print('Children (<7) survival rate: {:.2f}%'.format(child_survived*100))

print('Senior (>60) survival rate: {:.2f}%'.format(senior_survived*100))
# A quick look at the survival distribution between males and females

sns.violinplot('Sex', 'Survived', data=train)

plt.show()
class1 = train[train['Pclass']==1]

class2 = train[train['Pclass']==2]

class3 = train[train['Pclass']==3]

class1_survived = class1['Survived'].mean()

class2_survived = class2['Survived'].mean()

class3_survived = class3['Survived'].mean()

print('Class 1 survival rate: {:.2f}%'.format(class1_survived*100))

print('Class 2 survival rate: {:.2f}%'.format(class2_survived*100))

print('Class 3 survival rate: {:.2f}%'.format(class3_survived*100))
# A quick look at the survival distribution between males and females

sns.violinplot('Pclass', 'Survived', data=train)

plt.show()
# A quick look at the distribution of survivors and non-survivors across the 3 ports of embarkation

pd.crosstab(train['Embarked'], train['Survived'])
port_c = train[train['Embarked']=='C']

port_q = train[train['Embarked']=='Q']

port_s = train[train['Embarked']=='S']

port_c_survived = port_c['Survived'].mean()

port_q_survived = port_q['Survived'].mean()

port_s_survived = port_s['Survived'].mean()

print('Passengers embarked from Cherbourg survival rate: {:.2f}%'.format(port_c_survived*100))

print('Passengers embarked from Queenstown survival rate: {:.2f}%'.format(port_q_survived*100))

print('Passengers embarked from Southampton survival rate: {:.2f}%'.format(port_s_survived*100))
pd.crosstab(train['Pclass'], train['Embarked'])
# Plot box and whisker plots to visualize the continuous variables



plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

sns.boxplot(train['Age'], y=train['Sex'])



plt.subplot(1,2,2)

sns.boxplot(train['Fare'], y=train['Sex'])





plt.show()
# Plot histograms to visualize the continuous variables



plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

sns.distplot(train['Age'])



plt.subplot(1,2,2)

sns.distplot(train['Fare'])



plt.show()
# Check the skewness present in the dataset

skewness = train.skew()

skewness
# Using Isolation Forest to detect/predict the outliers



train_no_age_NA = train[train['Age'].notnull()]



from sklearn.ensemble import IsolationForest

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)

model.fit(train_no_age_NA[['Age']])

model.fit(train_no_age_NA[['Fare']])
# Adding new columns to show which observation is an outlier (-1)



train_no_age_NA['scores_age'] = model.decision_function(train_no_age_NA[['Age']])

train_no_age_NA['scores_fare'] = model.decision_function(train_no_age_NA[['Fare']])

train_no_age_NA['outlier_age'] = model.predict(train_no_age_NA[['Age']])

train_no_age_NA['outlier_fare'] = model.predict(train_no_age_NA[['Fare']])

train_no_age_NA.head(20)
# The number of outliers detected using machine learning

fare_out = len(train_no_age_NA[train_no_age_NA['outlier_fare'] == -1])

age_out = len(train_no_age_NA[train_no_age_NA['outlier_age'] == -1])



print('The number of outliers found in the age column: {}'.format(age_out))

print('The number of outliers found in the fare column: {}'.format(fare_out))
train_no_age_NA[train_no_age_NA['outlier_fare'] == -1].head(20)
# We will manually encode the Sex and Embarked column to obtain the correlation

# as we sort of know there are some relationship that lie within these variables

train['Sex'] = train['Sex'].replace({'male': 1, 'female': 2})

test['Sex'] = test['Sex'].replace({'male': 1, 'female': 2})

train['Embarked'] = train['Embarked'].replace({'C': 1, 'Q': 2, 'S': 3})

test['Embarked'] = test['Embarked'].replace({'C': 1, 'Q': 2, 'S': 3})
# Correlation check

correlation = train.corr()

correlation
# Visualize the correlations

plt.figure(figsize=(10,8))

sns.heatmap(correlation, cmap='RdYlGn', annot=True)

plt.title('Heatmap for the correlations found in training dataset')

plt.show()
# Check the null rows

null_train = train.isnull().sum()

null_test = test.isnull().sum()
null_train.sort_values(ascending=False)
null_test.sort_values(ascending=False)
# Combine both datasets

df = train.append(test).reset_index(drop=True)
# Fill the missing cells in Embarked column with the mode

mode_embarked = df['Embarked'].mode().values[0]

df['Embarked'] = df['Embarked'].fillna(mode_embarked)
# Fill the missing cell in Fare column with the median value by the class the passenger was in

df['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(x.median()))
# Extract the title of the passengers from their names

df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



# Grouping the titles together to form a smaller group

title_dict = {'Mr': 'Mr',

              'Miss': 'Miss',

              'Mrs': 'Mrs',

              'Master': 'Master',

              'Dr': 'Other',

              'Rev': 'Other',

              'Col': 'Other',

              'Major': 'Other',

              'Ms': 'Miss',

              'Mlle': 'Miss',

              'Dona': 'Royal',

              'Sir': 'Royal',

              'Lady': 'Royal',

              'Don': 'Royal',

              'Mme': 'Mrs',

              'Jonkheer': 'Royal',

              'Capt': 'Other',

              'the Countess': 'Royal'    

}



df['Title'] = df['Title'].map(title_dict)
# Survival rates by the passenger's title

df.groupby('Title')['Survived'].mean()
# Create a new feature call 'Deck' by extracting the first character from the 'Cabin' column

df['Deck'] = df['Cabin'].str[0]



# Replace the crew 'T' deck with 'A' as the *passenger* is in Pclass = 1

df['Deck'] = df['Deck'].replace('T', 'A')



# Group decks together to form a smaller group

df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')

df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')

df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')
# Create a new feature call 'Family Size' from the 'SibSp' and 'Parch' columns

df['Family Size'] = df['SibSp'] + df['Parch'] + 1



# Grouping Family size together to form a smaller group

df['Family Size'] = df['Family Size'].replace(1, 'No Family')

df['Family Size'] = df['Family Size'].replace([2,3,4], 'Small')

df['Family Size'] = df['Family Size'].replace([5,6,7,8,11], 'Large')

#df['Family Size'] = df['Family Size'].replace([7,8,11], 'Large')
df['Family Size'].value_counts()
# Obtain family name to analyze the dataset deeper (it will be dropped before fitting any models)

df['Surname'] = df['Name'].str.split(',', expand=True)[0]
# Create new feature to distinguish passengers with no family

#df['Is Alone'] = df.apply(lambda x: 1 if (x['SibSp'] + x['Parch'] + 1 == 1) else 0, axis=1)
# Create a new feature to distinguish passengers who are in a family with kids/parents/guardian

#df['Travel w/ Kids'] = df.apply(lambda x: 1 if x['Parch']>0 else 0, axis=1)
df.head()
# For this analysis, we have to split our dataset into the train and test dataset again

train2 = df.iloc[:891].copy()

test2 = df.iloc[891:].copy().drop('Survived', axis=1)

missing_cabin_df = train2[train2['Deck'].isnull()]
missing_cabin_df.head()
missing_survived_rate = missing_cabin_df['Survived'].mean()

missing_survived = len(missing_cabin_df[missing_cabin_df['Survived'] == 1])

print('The survival rate for passengers with missing decks: {:.2f}%'.format(missing_survived_rate*100))

print('The number of survivors: {}'.format(missing_survived))
class1m = missing_cabin_df[missing_cabin_df['Pclass']==1]

class2m = missing_cabin_df[missing_cabin_df['Pclass']==2]

class3m = missing_cabin_df[missing_cabin_df['Pclass']==3]

class1m_survived = class1m['Survived'].mean()

class2m_survived = class2m['Survived'].mean()

class3m_survived = class3m['Survived'].mean()

print('For passengers with missing decks,')

print('Class 1 survival rate: {:.2f}%'.format(class1m_survived*100))

print('Class 2 survival rate: {:.2f}%'.format(class2m_survived*100))

print('Class 3 survival rate: {:.2f}%'.format(class3m_survived*100))
train2.head()
drop_col = ['Name', 'Ticket', 'Cabin', 'Surname', 'Embarked', 'SibSp', 'Parch', 'PassengerId']

train2 = train2.drop(drop_col, axis=1)

test2 = test2.drop(drop_col, axis=1)
from fancyimpute import KNN

from sklearn.preprocessing import OrdinalEncoder



#instantiate both packages to use

encoder = OrdinalEncoder()

imputer = KNN()

# create a list of categorical columns to iterate over

cat_cols = ['Title', 'Deck', 'Family Size']

df_all = [train2, test2]



def encode(data):

    '''function to encode non-null data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode date

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data



#create a for loop to iterate through each column in the data

for df in df_all:

    for columns in cat_cols:

        encode(df[columns])
# impute data and convert

encode_data_train = pd.DataFrame(imputer.fit_transform(train2),columns = train2.columns)

encode_data_test = pd.DataFrame(imputer.fit_transform(test2),columns = test2.columns)
# Since KNN produces floats, we will have to round up the deck and age columns

encode_data_train.loc[:, ['Age', 'Deck']] = round(encode_data_train.loc[:, ['Age', 'Deck']])

encode_data_test.loc[:, ['Age', 'Deck']] = round(encode_data_test.loc[:, ['Age', 'Deck']])
# Create dummy variables

#encode_dfs = [encode_data_train, encode_data_test]

#for df in encode_dfs:

    #df['Is Infant'] = df['Age'].apply(lambda x: 1 if x<3 else 0)

    #df['Is Child'] = df['Age'].apply(lambda x: 1 if (x<11 and x>=3) else 0)

    #df['Is Teen'] = df['Age'].apply(lambda x: 1 if (x<20 and x>=11) else 0)

    #df['Is Adult'] = df['Age'].apply(lambda x: 1 if (x<60 and x>=20) else 0)

    #df['Is Senior'] = df['Age'].apply(lambda x: 1 if x>=60 else 0)
encode_data_train.head()
encode_data_test.head()
# Now we can perform binning towards the Age and Fare columns 

encode_dfs = [encode_data_train, encode_data_test]



for df in encode_dfs:

    df['Age'] = pd.qcut(df['Age'], 8)

    df['Fare'] = pd.qcut(df['Fare'], 12)
# Label encode the binned columns

from sklearn.preprocessing import LabelEncoder



encode_cols = ['Age', 'Fare']

for df in encode_dfs:

    for col in encode_cols:

        df[col] = LabelEncoder().fit_transform(df[col])
encode_data_train.skew().abs().sort_values(ascending=False)
encode_data_test.skew().abs().sort_values(ascending=False)
# One hot encode the nominal categories

from sklearn.preprocessing import OneHotEncoder

nominal_col = ['Pclass', 'Family Size', 'Sex', 'Title', 'Deck']



dfs = [encode_data_train, encode_data_test]



encoded_features = []

for df in dfs:

    for col in nominal_col:

        encoded_feat = OneHotEncoder().fit_transform(df[col].values.reshape(-1, 1)).toarray()

        n = df[col].nunique()

        cols = ['{}_{}'.format(col, n) for n in range(1, n + 1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df.index

        encoded_features.append(encoded_df)



df_train = pd.concat([encode_data_train, *encoded_features[:5]], axis=1).drop(nominal_col, axis=1)

df_test = pd.concat([encode_data_test, *encoded_features[5:]], axis=1).drop(nominal_col, axis=1)
df_train.head()
df_test.head()
# Now we fit and run a random forest model

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler



y_train = train['Survived']

X_train = df_train.drop('Survived', axis=1)

X_test = df_test



random_forest = RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 4, 

                                       min_samples_split = 10,

                                       n_estimators = 1300, 

                                       max_features = 'auto', 

                                       oob_score = True, 

                                       random_state = 0, 

                                       n_jobs = -1)



scores1 = cross_val_score(random_forest, X_train, y_train, cv=5)

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)





print('Mean CV-5 score:', scores1.mean())



rf_score = round(random_forest.score(X_train, y_train) * 100, 2)

print('Accuracy:', rf_score, '%')
#area for feature importance visual

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances
output = pd.DataFrame(columns=['PassengerId', 'Survived'])

output['PassengerId'] = test['PassengerId']

output['Survived'] = rf_pred.astype(int)

output.to_csv('my_submission.csv', header=True, index=False)
output.head(20)