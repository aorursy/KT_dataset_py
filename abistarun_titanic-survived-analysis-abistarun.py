%matplotlib inline

import re

import numpy as np

import scipy.stats as sp

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder 

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import f_classif

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv", index_col = 'PassengerId')

test = pd.read_csv("../input/test.csv", index_col = 'PassengerId')
def get_combined_dataframe():

    combined = train.append(test)

    combined.drop('Survived', axis=1, inplace=True)

    return combined



class DataType:

    CONTINUOUS = 1

    CATEGORICAL = 2
train.head()
train.info()
get_combined_dataframe().isnull().sum()/len(get_combined_dataframe()) * 100
median = train['Age'].median()

train['Age'].fillna(median, inplace=True)

test['Age'].fillna(median, inplace=True)



train['Cabin'].fillna('U0', inplace=True)

test['Cabin'].fillna('U0', inplace=True)



mode = train['Embarked'].mode().iloc[0]

train['Embarked'].fillna(mode, inplace=True)

test['Embarked'].fillna(mode, inplace=True)



median = train['Fare'].median()

train['Fare'].fillna(median, inplace=True)

test['Fare'].fillna(median, inplace=True)



get_combined_dataframe().isnull().sum()/len(get_combined_dataframe()) * 100
def extract_title(name):

    name = name.split(',')[1].split('.')[0].strip()

    if name == 'Jonkheer':

        return 'Master'

    elif name in ['Ms', 'Mlle']:

        return 'Miss'

    elif name == 'Mme':

        return 'Mrs'

    elif name in ['Capt', 'Don', 'Major', 'Col', 'Sir', 'Rev']:

        return 'Sir'

    elif name in ['Dona', 'Lady', 'the Countess']:

        return 'Lady'

    else:

        return name



def extract_last_name(name):

    return name.split(',')[0].strip()



train['Title'] = train['Name'].map(extract_title)

test['Title'] = test['Name'].map(extract_title)



train['LastName'] = train['Name'].map(extract_last_name)

test['LastName'] = test['Name'].map(extract_last_name)



train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
plt.figure(figsize=(15,8))

sns.countplot(x='Title', hue='Survived', data=train, palette=["#ff0000", "#00ff00"])
plt.figure(figsize=(15,8))

sns.countplot(x='Sex', hue='Survived', data=train, palette=["#ff0000", "#00ff00"])
train.drop('Ticket', axis=1, inplace=True)

test.drop('Ticket', axis=1, inplace=True)
train['Deck'] = train['Cabin'].map(lambda cabin: cabin[:1])

test['Deck'] = test['Cabin'].map(lambda cabin: cabin[:1])



def extract_room_no(cabin):

    m = re.search("\d", cabin)

    if m is None:

        return 0

    i = m.start()

    return int(cabin[i:i+3].strip())



train['RoomNo'] = train['Cabin'].map(extract_room_no)

test['RoomNo'] = test['Cabin'].map(extract_room_no)



train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)
plt.figure(figsize=(15,8))

sns.countplot(x='Deck', hue='Survived', data=train, palette=["#ff0000", "#00ff00"])
plt.figure(figsize=(15,8))

sns.countplot(x='RoomNo', hue='Survived', data=train, palette=["#ff0000", "#00ff00"])
train.drop('RoomNo', axis=1, inplace=True)

test.drop('RoomNo', axis=1, inplace=True)
plt.figure(figsize=(15,8))

sns.countplot(x='Embarked', hue='Survived', data=train, palette=["#ff0000", "#00ff00"])
plt.figure(figsize=(15,8))

sns.countplot(x='Pclass', hue='Survived', data=train, palette=["#ff0000", "#00ff00"])
plt.figure(figsize=(15,8))

sns.kdeplot(train[train['Survived']==1]['Age'], color='#00ff00')

sns.kdeplot(train[train['Survived']==0]['Age'], color='#ff0000')
plt.figure(figsize=(15,8))

sns.kdeplot(train[train['Survived']==1]['SibSp'], color='#00ff00')

sns.kdeplot(train[train['Survived']==0]['SibSp'], color='#ff0000')
plt.figure(figsize=(15,8))

sns.kdeplot(train[train['Survived']==1]['Parch'], color='#00ff00')

sns.kdeplot(train[train['Survived']==0]['Parch'], color='#ff0000')
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1

test['FamilySize'] = test['Parch'] + test['SibSp'] + 1



plt.figure(figsize=(15,8))

sns.kdeplot(train[train['Survived']==1]['FamilySize'], color='#00ff00')

sns.kdeplot(train[train['Survived']==0]['FamilySize'], color='#ff0000')
plt.figure(figsize=(15,8))

sns.kdeplot(train[train['Survived']==1]['Fare'], color='#00ff00')

sns.kdeplot(train[train['Survived']==0]['Fare'], color='#ff0000')
le = LabelEncoder()

le.fit(train['Deck'])

train['Deck'] = le.transform(train['Deck'])

test['Deck'] = le.transform(test['Deck'])



le.fit(train['Sex'])

train['Sex'] = le.transform(train['Sex'])

test['Sex'] = le.transform(test['Sex'])
def corr(dataframe, datatypes):

    columns = dataframe.columns

    p_vals = pd.DataFrame(columns=columns, index=columns)

    for c1 in columns:

        for c2 in columns:

            corr=None

            p_val = None

            if c1==c2:

                continue

            elif datatypes[c1]==DataType.CONTINUOUS and datatypes[c2]==DataType.CONTINUOUS:

                # Spearman

                corr, p_val = sp.spearmanr(dataframe[c1], dataframe[c2])

            elif datatypes[c1]==DataType.CATEGORICAL and datatypes[c2]==DataType.CATEGORICAL:

                # Chi2

                corr, p_val = chi2(dataframe[c1], dataframe[c2])

            elif datatypes[c1]==DataType.CONTINUOUS and datatypes[c2]==DataType.CATEGORICAL:

                # Anova

                corr, p_val = f_classif(dataframe[c1].to_frame(), dataframe[c2].to_frame())

                corr = corr[0]

                p_val = p_val[0]

            elif datatypes[c1]==DataType.CATEGORICAL and datatypes[c2]==DataType.CONTINUOUS:

                # Anova

                corr, p_val = f_classif(dataframe[c2].to_frame(), dataframe[c1].to_frame())

                corr = corr[0]

                p_val = p_val[0]

            p_vals[c1][c2] = p_val

    return p_vals

                

def chi2(feature1, feature2):

    f1_uniq = feature1.unique()

    f2_uniq = feature2.unique()

    observed_matrix = prepare_observed_matrix(feature1, feature2, f1_uniq, f2_uniq)

    expected_matrix = prepare_expected_matrix(observed_matrix)    

    deg_of_freedom = get_degree_of_freedom_chi2(observed_matrix)

    observed_matrix = observed_matrix.ravel()

    expected_matrix = expected_matrix.ravel()

    return sp.chisquare(observed_matrix, expected_matrix, ddof=deg_of_freedom)

    

def get_degree_of_freedom_chi2(observed_matrix):

    # Note : This is not degree of freedom but the adjustment required in dof.

    # Default is : k-1 (k is noOFObservations)

    # So (r-1)*(c-1) = k-1-dof

    # Hence, dof = k-1-(r-1)*(c-1)

    no_of_rows, no_of_cols = observed_matrix.shape

    deg_of_freedom = len(observed_matrix.ravel()) - 1 - ((no_of_rows - 1) * (no_of_cols - 1))

    return deg_of_freedom





def prepare_expected_matrix(observed_matrix):

    row_sum = observed_matrix.sum(axis=1)

    col_sum = observed_matrix.sum(axis=0)

    total = row_sum.sum()

    no_of_rows, no_of_cols = observed_matrix.shape



    expected_matrix = []

    for i in range(0, no_of_rows):

        curr_row = []

        for j in range(0, no_of_cols):

            curr_row.append(float((row_sum[i] * col_sum[j])) / total)

        expected_matrix.append(curr_row)

    expected_matrix = np.array(expected_matrix)

    return expected_matrix





def prepare_observed_matrix(data_frame_column, target_column, column_unique_values, target_unique_values):

    contengency_table_observed = pd.crosstab(index=target_column,

                                                 columns=data_frame_column)

    contengency_table_observed.columns = column_unique_values

    contengency_table_observed.index = target_unique_values

    observed_matrix = contengency_table_observed.as_matrix()

    return observed_matrix



datatypes = { 

    'Age':DataType.CONTINUOUS, 

    'Embarked':DataType.CATEGORICAL, 

    'FamilySize':DataType.CONTINUOUS, 

    'Fare':DataType.CONTINUOUS, 

    'LastName':DataType.CATEGORICAL, 

    'Pclass':DataType.CONTINUOUS, 

    'Sex':DataType.CATEGORICAL, 

    'Title':DataType.CATEGORICAL, 

    'Deck':DataType.CATEGORICAL, 

    'Parch':DataType.CONTINUOUS, 

    'SibSp':DataType.CONTINUOUS, 

    'Survived':DataType.CATEGORICAL 

} 



def color_corr_features(val):

    if val is not None:

        if val<10**-100:

            return 'background-color: red'

    return 'background-color: green'

    

p_values = corr(train, datatypes)

p_values.style.applymap(color_corr_features)
train.drop(['Parch', 'SibSp', 'Deck', 'Sex'], axis=1, inplace=True)

test.drop(['Parch', 'SibSp', 'Deck', 'Sex'], axis=1, inplace=True)
get_combined_dataframe().info()
def one_hot_encoding(dataframe, features):

    for feature in features:

        if feature in dataframe.columns:

            dummies = pd.get_dummies(dataframe[feature], prefix=feature + "_")

            dataframe = pd.concat([dataframe,dummies], axis=1)

    return dataframe.drop(features, axis=1)



train = one_hot_encoding(train, ['Embarked', 'LastName', 'Title'])

test = one_hot_encoding(test, ['Embarked', 'LastName', 'Title'])



# Get missing columns in the training test

# Add a missing column in test set with default value equal to 0

train_cols = train.drop("Survived", axis=1).columns

missing_cols = set(train_cols) - set(test.columns)

for c in missing_cols:

    test[c] = 0



# Ensure the order of column in the test set is in the same order than in train set

test = test[train_cols]
scaler = StandardScaler()

features_to_scale = ['Age', 'Fare', 'FamilySize']

scaler.fit(train[features_to_scale])

train[features_to_scale] = scaler.transform(train[features_to_scale])

test[features_to_scale] = scaler.transform(test[features_to_scale])
x = train.drop("Survived", axis=1)

y = train["Survived"]
model = LinearSVC(loss='squared_hinge', dual=False, C=100)
cross_val_score(model, x, y, scoring='accuracy', cv=5)    
cross_val_score(model, x, y, scoring='roc_auc', cv=5)    
model = RandomForestClassifier(n_estimators=100, criterion="gini")
cross_val_score(model, x, y, scoring='accuracy', cv=5)    
cross_val_score(model, x, y, scoring='roc_auc', cv=5)
model = RandomForestClassifier(n_estimators=100, criterion="gini")

model.fit(x, y)

y_predict = model.predict(test)

data = {"Survived":y_predict}

predicted_df = pd.DataFrame(index=test.index, data=data)
predicted_df.to_csv("PredictedResult.csv")