import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
titanic_train = pd.read_csv("../input/train.csv", index_col='PassengerId')
titanic_test = pd.read_csv("../input/test.csv", index_col = "PassengerId")
train_results = titanic_train["Survived"].copy()
titanic_train.drop("Survived", axis=1, inplace=True, errors="ignore")
titanic = pd.concat([titanic_train, titanic_test])
traindex = titanic_train.index
testdex = titanic_test.index
titanic.shape[0]
titanic[titanic["Cabin"].isnull()==True].shape[0]
titanic["Cabin"].value_counts()
cabin_only = titanic[["Cabin"]].copy()
cabin_only["Cabin_Data"] = cabin_only["Cabin"].isnull().apply(lambda x: not x)
cabin_only["Deck"] = cabin_only["Cabin"].str.slice(0,1)
cabin_only["Room"] = cabin_only["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
cabin_only[cabin_only["Cabin_Data"]]
cabin_only[cabin_only["Deck"]=="F"]
cabin_only.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")
cabin_only["Deck"] = cabin_only["Deck"].fillna("N")
cabin_only["Room"] = cabin_only["Room"].fillna(cabin_only["Room"].mean())
cabin_only.info()
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
cabin_only = one_hot(cabin_only, ["Deck"],drop_col=True)
cabin_only.head()
for column in cabin_only.columns.values[1:]:
    titanic[column] = cabin_only[column]
titanic.drop(["Ticket","Cabin"], axis=1, inplace=True)
corr = titanic.corr()
corr["Pclass"].sort_values(ascending=False)
corr["Fare"].sort_values(ascending=False)
# Train
train_df = cabin_only.loc[traindex, :]
train_df['Survived'] = train_results

# Test
test_df = cabin_only.loc[testdex, :]
test_df.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import scipy.stats as st
rfc = RandomForestClassifier()
X = train_df.drop("Survived", axis=1).copy()
y = train_df["Survived"]
param_grid ={'max_depth': st.randint(6, 11),
             'n_estimators':st.randint(300, 500),
             'max_features':np.arange(0.5,.81, 0.05),
            'max_leaf_nodes':st.randint(6, 10)}

grid = RandomizedSearchCV(rfc,
                    param_grid, cv=10,
                    scoring='accuracy',
                    verbose=1,n_iter=20)

grid.fit(X, y)
grid.best_estimator_
grid.best_score_
predictions = grid.best_estimator_.predict(test_df)
results_df = pd.DataFrame()
results_df["PassngerId"] = test_df.index
results_df["Predictions"] = predictions
results_df.head()
results_df.to_csv("Predictions", index=False)