import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", index_col='PassengerId')

test = pd.read_csv("../input/test.csv", index_col='PassengerId')

test.head()
train.tail()
train_results = train["Survived"].copy()

train.drop("Survived", axis=1, inplace=True, errors="ignore")

full_df = pd.concat([train, test])

traindex = train.index

testdex = test.index
full_df.drop("Ticket", axis=1, inplace=True, errors="ignore")

full_df["Fare"] = full_df["Fare"].fillna(full_df["Fare"].mean())

full_df["Age"] = full_df["Age"].fillna(full_df["Age"].mean())

full_df["Embarked"] = full_df["Embarked"].fillna(full_df["Embarked"].mode().iloc[0])

full_df["Cabin_Data"] = full_df["Cabin"].isnull().apply(lambda x: not x)
full_df["Deck"] = full_df["Cabin"].str.slice(0,1)

full_df["Room"] = full_df["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")

full_df[full_df["Cabin_Data"]]


full_df["Deck"] = full_df["Deck"].fillna("N")

full_df["Room"] = full_df["Room"].fillna(full_df["Room"].mean())
full_df.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")
def one_hot_column(df, label, drop_col=False):

    '''

    This function will one hot encode the chosen column.

    Args:

        df: Pandas dataframe

        label: Label of the column to encode

        drop_col: boolean to decide if the chosen column should be dropped

    Returns:

        pandas dataframe with the given encoding

    '''

    one_hot = pd.get_dummies(df[label], prefix=label)

    if drop_col:

        df = df.drop(label, axis=1)

    df = df.join(one_hot)

    return df





def one_hot(df, labels, drop_col=False):

    '''

    This function will one hot encode a list of columns.

    Args:

        df: Pandas dataframe

        labels: list of the columns to encode

        drop_col: boolean to decide if the chosen column should be dropped

    Returns:

        pandas dataframe with the given encoding

    '''

    for label in labels:

        df = one_hot_column(df, label, drop_col)

    return df
one_hot_df = one_hot(full_df, ["Embarked","Deck"])
one_hot_df.info()
one_hot_df.drop(["Embarked", "Deck"], axis=1, inplace=True, errors="ignore")
one_hot_df.head()
full_df["Title"] = full_df["Name"].str.extract("([A-Za-z]+\.)", expand=False)
full_df["Title"] = full_df["Title"].fillna("None")
full_df["Title"].value_counts()
one_hot_df["Title"] = full_df["Title"]

one_hot_df = one_hot_column(one_hot_df, "Title")
one_hot_df.drop("Name", axis=1, inplace=True, errors="ignore")
one_hot_df["Sex"] = one_hot_df["Sex"].map({"male": 0, "female":1}).astype(int)
one_hot_df.drop("Title_Dona.", axis=1, inplace=True, errors="ignore")

one_hot_df.drop("Title", axis=1, inplace=True, errors="ignore")
# Train

train_df = one_hot_df.loc[traindex, :]

train_df['Survived'] = train_results



# Test

test_df = one_hot_df.loc[testdex, :]
corr = train_df.corr()
corr["Survived"].sort_values(ascending=False)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron, SGDClassifier

from sklearn.neural_network import MLPClassifier



from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn import metrics

from sklearn.model_selection import cross_val_score



import scipy.stats as st
rfc = RandomForestClassifier()
rfc.get_params().keys()
X = train_df.drop("Survived", axis=1).copy()

y = train_df["Survived"]
param_grid ={'max_depth': st.randint(6, 11),

             'n_estimators':st.randint(300, 500),

             'max_features':np.arange(0.5,.81, 0.05),

            'max_leaf_nodes':st.randint(6, 10)}



grid = RandomizedSearchCV(rfc,

                    param_grid, cv=10,

                    scoring='accuracy',

                    verbose=1,n_iter=80)



grid.fit(X, y)
grid.best_estimator_
grid.cv_results_
predictions = grid.best_estimator_.predict(test_df)
predictions
results_df =pd. DataFrame()

results_df["PassngerId"] = test_df.index

results_df["Predictions"] = predictions
results_df
results_df.to_csv("Predictions", index=False)