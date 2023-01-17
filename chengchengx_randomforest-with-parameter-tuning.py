import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from pandas.api.types import is_string_dtype, is_numeric_dtype

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt
# convert non-numeric columns to numeric columns

def convert2numeric(cols, df):

    for col in cols:

        if is_string_dtype(df[col]):

            df[col] = df[col].astype('category').cat.as_ordered()

        df[col] = df[col].cat.codes + 1
# col: a column name with missing values; 

# df : dataframe

# add_na: a boolean value to indicate whether to add a col_na column

# Note: This currently only works for handling contiuous variables

def fix_missing(col, add_na, df):

    if is_numeric_dtype(df[col]):

        is_na = pd.isnull(df[col])

        if add_na: df[col+'_na'] = is_na

        df[col][is_na] = df[col].median()
# Write prediction result to local file

def writeTOfile(nums1, nums2, fname):

    # nums is id list, and nums2 is probabilities of preidiction

    if len(nums1)!=len(nums2): return

    s = "PassengerId,Survived\n"

    for i in range(0, len(nums1)):

        s += str(nums1[i]) + "," + str(nums2[i]) + "\n"

    f = open(fname, 'w')

    f.write(s)
df_raw = pd.read_csv('../input/train.csv', low_memory=False)
df_raw.shape
df_raw
df_raw.dtypes
is_string_dtype(df_raw["Name"]), is_string_dtype(df_raw["Sex"]),\

is_string_dtype(df_raw["Ticket"]), is_string_dtype(df_raw["Cabin"]), \

is_string_dtype(df_raw["Embarked"])
df_raw.Name.unique().shape, df_raw.Ticket.unique().shape,\

df_raw.Cabin.unique().shape, df_raw.Sex.unique().shape, \

df_raw.Embarked.unique().shape
df = df_raw.copy()

# PassengerID is also needed to be dropped

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
# convert Sex and Embarked to numeric type

convert2numeric(["Sex", "Embarked"], df)
df.dtypes
df.isnull().sum().sort_index()/len(df)
# For Age, create a new column Age_na to indicate which data is missing, and 

# replace na with median Age.

fix_missing("Age", True, df)
df.isnull().sum().sort_index()/len(df)
# Convert 'Survided' to categorical variable

df['Survived'] = df['Survived'].astype('category')

X = df.drop('Survived', axis = 1)

y = df["Survived"]
# Tune three parameters: n_estimators, min_samepls_leaf, and max_features

# It might take some to run

numOfestimators = [1, 10, 20, 30, 40, 50]

numOfleafs = [1, 3, 5, 10, 25]

numOffeatures = np.arange(0.1, 1.1, 0.1)

best_result = []

for numOfestimator in numOfestimators:

    for numOfleaf in numOfleafs:

        for numOffeature in numOffeatures:  

            result = [numOfestimator, numOfleaf, numOffeature]

            m = RandomForestClassifier(n_jobs=-1, n_estimators=numOfestimator,\

                                    min_samples_leaf=numOfleaf,\

                                    max_features=numOffeature)

            m.fit(X, y)

            result.append(m.score(X, y))

            result.append(m.oob_score_)

            if len(best_result) == 0: best_result = result

            elif best_result[4] < result[4]: 

                best_result = result

print(best_result)
# Chose parameters

# [30, 3, 0.7]

m_final = RandomForestClassifier(n_jobs=-1, n_estimators=30,\

                                    min_samples_leaf=3,\

                                    max_features=0.7, oob_score=True)

m_final.fit(X, y)
test_df = pd.read_csv('../input/test.csv', low_memory=False)
test_df.shape
# Drop some columns

# PassengerId is extracted to construct submission file

pId_list = test_df.PassengerId.tolist()

test_df = test_df.drop(['PassengerId', 'Name', \

                        'Ticket', 'Cabin'], axis = 1)
# convert Sex and Embarked to numeric columns

convert2numeric(["Sex", "Embarked"], test_df)
test_df.shape
test_df.isnull().sum().sort_index()/len(df)
fix_missing("Age", True, test_df)

fix_missing("Fare", False, test_df)
m_final.oob_score_
result = m_final.predict(test_df)

predictions = [row for row in result]

writeTOfile(pId_list, predictions, "submission.csv")