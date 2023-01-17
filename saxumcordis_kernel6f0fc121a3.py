import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.head()

df.info()
df.isnull().sum() * 100 / len(df)

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if big_string is np.NaN or big_string.find(substring) != -1:

            return substring

    print(big_string)

    return np.nan
#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
from sklearn.base import BaseEstimator, TransformerMixin





class CusttomAttribTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self 

    

    def transform(self, X):

        title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr','Ms', 

            'Mlle','Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']

        X['Title'] = X['Name'].map(lambda x: substrings_in_string(x, title_list))

        X['Title'] = X.apply(replace_titles, axis=1)

        

        cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

        X['Deck'] = X['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

        X['Family_Size'] = X['SibSp'] + X['Parch']

        X = X.drop(columns=['Cabin', 'Name', 'Parch', 'SibSp', 'Sex'])

        X[['Deck', 'Title']] = OrdinalEncoder().fit_transform(X[['Deck', 'Title']])

        imp = SimpleImputer(missing_values=np.nan, strategy='median')

        X["Age"] = imp.fit_transform(X[["Age"]])

        X["Age"] = pd.cut(X["Age"], bins=[0., 18, 30, 40, 50, 90], labels=[1, 2, 3, 4, 5])



        return X
from sklearn.model_selection import train_test_split



y = df['Survived']

X = df.drop(['Survived'], axis=1)





# Divide data into training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler





categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('ordinal', OrdinalEncoder())

])



numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaling', MinMaxScaler())

])





# custom_columns = ['Name', 'SibSp', 'Parch', 'Cabin', 'Sex', 'Age']

# categorical_cols = [c for c in categorical_cols if c not in custom_columns]

# numerical_cols = [c for c in numerical_cols if c not in custom_columns]



my_cols = numerical_cols + categorical_cols



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

#         ('custom', CusttomAttribTransformer(), custom_columns),

        ('num', SimpleImputer(strategy='median'), numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

# cols = ['Age', 'Title', 'Deck', 'Family_Size'] + my_cols
X_train_full = pd.DataFrame(preprocessor.fit_transform(X_train_full), columns=my_cols)

X_valid_full = pd.DataFrame(preprocessor.fit_transform(X_valid_full), columns=my_cols)
X_train_full
X_train_full.corr()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data = pd.DataFrame(preprocessor.transform(test_data), columns=my_cols)

test_data.head()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error
print(X_valid_full.shape, y_valid.shape)
my_model = LogisticRegression()

my_model.fit(X_train_full, y_train)

predictions = my_model.predict(X_valid_full)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
my_model = RandomForestClassifier()

my_model.fit(X_train_full.append(X_valid_full), y_train.append(y_valid)) 

predictions = my_model.predict(test_data)
output = pd.DataFrame({'PassengerId': test_data.PassengerId.astype(int), 'Survived': predictions.round().astype(int)})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")