import pandas as pd 



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]

train_df.info()
train_df.describe(include=['O'])
train_df[['MSZoning', 'SalePrice']].groupby(['MSZoning'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)


MSZoning_mapping = {"FV": 1, "RL": 2, "RH": 3, "RM": 4, "C (all)": 5}

for dataset in combine:

    dataset['MSZoning'] = dataset['MSZoning'].map(MSZoning_mapping)

    dataset['MSZoning'] = dataset['MSZoning'].fillna(0)

    

train_df.head()
M = {item:index for item,index in enumerate(set(train_df['Street']))}

print(M)



for dataset in combine:

    dataset['Street'] = dataset['Street'].map(M)

    dataset['Street'] = dataset['Street'].fillna(0)
train_df[['GarageQual', 'SalePrice']].groupby(['GarageQual'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
GarageQual_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}

for dataset in combine:

    dataset['GarageQual'] = dataset['GarageQual'].map(GarageQual_mapping)

    dataset['GarageQual'] = dataset['GarageQual'].fillna(0)

    

train_df.head()
train_df[['GarageType', 'SalePrice']].groupby(['GarageType'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
GarageType_mapping = {"BuiltIn": 1, "Attchd": 2, "Basment": 3, "2Types": 4, "Detchd": 5, "CarPort": 6}

for dataset in combine:

    dataset['GarageType'] = dataset['GarageQual'].map(GarageType_mapping)

    dataset['GarageType'] = dataset['GarageQual'].fillna(0)

    

train_df.head()
train_df[['GarageCond', 'SalePrice']].groupby(['GarageCond'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
GarageCond_mapping = {"TA": 1, "Gd": 2, "Ex": 3, "Fa": 4, "Po": 5}

for dataset in combine:

    dataset['GarageCond'] = dataset['GarageCond'].map(GarageType_mapping)

    dataset['GarageCond'] = dataset['GarageCond'].fillna(0)

    

train_df.head()
train_df[['PoolArea', 'SalePrice']].groupby(['PoolArea'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
for dataset in combine:

    dataset['PoolArea'] = dataset['PoolArea'].map(lambda x: 1 if x>0 else 0)
X_train = train_df[["MSZoning", "YearBuilt", "YrSold", "OverallQual", "OverallCond", "GarageQual", "GarageType", "GarageCond", "PoolArea"]]

Y_train = train_df["SalePrice"]

X_test  = test_df[["MSZoning", "YearBuilt", "YrSold", "OverallQual", "OverallCond", "GarageQual", "GarageType", "GarageCond", "PoolArea"]]

X_train.shape, Y_train.shape, X_test.shape



# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)



submission = pd.DataFrame({

        "Id": test_df["Id"],

        "SalePrice": Y_pred

    })

submission.to_csv("output.csv")