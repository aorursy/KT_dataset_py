import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from pandas.api.types import is_string_dtype, is_numeric_dtype
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

X_train, X_test, y_train, y_test = train_test_split(X, y, \

                                                    test_size = 0.30,\

                                                    random_state = 99)
X_train.shape, X_test.shape, y_test.shape, y_test.shape
# First Random Forest model

m = RandomForestClassifier(n_jobs=-1)

m.fit(X_train, y_train)
m.score(X_train, y_train)
m.score(X_test, y_test)
test_df = pd.read_csv('../input/test.csv', low_memory=False)
test_df.shape
# Drop some columns

# PassengerId is extracted to construct submission file

pId_list = test_df.PassengerId.tolist()

test_df = test_df.drop(['PassengerId', 'Name', \

                        'Ticket', 'Cabin'], axis = 1)
# convert Sex and Embarked to numeric columns

convert2numeric(["Sex", "Embarked"], test_df)
test_df.isnull().sum().sort_index()/len(df)
fix_missing("Age", True, test_df)

fix_missing("Fare", False, test_df)
result = m.predict(test_df)

predictions = [row for row in result]

writeTOfile(pId_list, predictions, "submission.csv")