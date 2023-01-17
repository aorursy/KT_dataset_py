import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt



from matplotlib import rcParams



rcParams['figure.figsize'] = (4, 2.7)

rcParams['figure.dpi'] = 150

#rcParams['axes.color_cycle'] = dark2_colors

rcParams['lines.linewidth'] = 2

rcParams['axes.grid'] = False

rcParams['axes.facecolor'] = '#eeeeee'

rcParams['font.size'] = 9

rcParams['patch.edgecolor'] = 'none'
train_original = pd.read_csv("../input/train.csv")



# X = features, y = labels



X = train_original[train_original.columns.drop("Survived")]

y = train_original["Survived"]

X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

train = pd.concat([y_train, X_train], axis=1)

test = pd.concat([y_test, X_test], axis=1)

train.head()
train.info()
feat = "Sex"

train[["Survived", feat]].groupby(by=feat).mean().plot.bar()

plt.title("Survival rate vs. Sex", )
feat = ["Sex", "SibSp", "Parch"]

df_perc = train[["Survived", "Sex", "SibSp", "Parch"]].groupby(by=feat).mean().reset_index()

pd.crosstab(df_perc.Sex, df_perc.SibSp, margins=True, 

            normalize=True).style.background_gradient(cmap="summer_r", axis=1)
pd.crosstab(df_perc.Sex, df_perc.Parch, margins=True, 

            normalize=True).style.background_gradient(cmap="summer_r", axis=1)
feat = ["Age", "Parch"]

df_perc = train[["Survived", "Age", "Parch"]].groupby(by=feat).mean().reset_index()

# group the ages at every 10 years

df_perc["age_group"] = np.where(df_perc["Age"] < 10, "0-10",

                               np.where(df_perc["Age"] < 20, "11-20",

                                       np.where(df_perc["Age"] < 30, "21-30",

                                               np.where(df_perc["Age"] > 60, "61+",

                                                       "31-60"))))



pd.crosstab(df_perc.age_group, df_perc.Parch, margins=True, 

            normalize=True).style.background_gradient(cmap="summer_r", axis=0)
feat = "Pclass"

#df_perc = train[["Survived", "Pclass"]].groupby(by=feat).mean().reset_index()

pd.crosstab(train.Survived, train[feat], margins=True, 

            normalize=True).style.background_gradient(cmap="summer_r", axis=1)
feat = "Fare"

df_perc = train[["Survived", "Fare"]].groupby(by=feat).mean().reset_index()

# group the fare at every 100

df_perc["fare_group"] = np.where(df_perc[feat] < 100, "0-100",

                               np.where(df_perc[feat] < 200, "101-200",

                                       np.where(df_perc[feat] < 300, "201-300",

                                                np.where(df_perc[feat] < 400, "301-400",

                                                         np.where(df_perc[feat] < 500, "401-500",

                                                                    "500+")))))

df_perc[["Survived", "fare_group"]].groupby(by=["fare_group"]).mean().plot()
feat = "Cabin"

#df_perc = train[["Survived", "Pclass"]].groupby(by=feat).mean().reset_index()

train["Cabin_group"] = train[feat].str[0]

feat = "Cabin_group"

pd.crosstab(train.Survived, train[feat], margins=True, 

            normalize=True).style.background_gradient(cmap="summer_r", axis=0)
feat = "Embarked"

pd.crosstab(train.Survived, train[feat], margins=True, 

            normalize=True).style.background_gradient(cmap="summer_r", axis=1)
def create_missing_df(df):

    missing_df = df.isnull().sum(axis=0).reset_index()

    missing_df.columns = ["feature", "missing_count"]

    missing_df = missing_df[missing_df.missing_count > 0]

    missing_df["missing_perc"] = 100 * missing_df["missing_count"] / train.shape[0]

    missing_df = missing_df.sort_values(by="missing_perc", ascending=False)

    return missing_df
missing_df = create_missing_df(train)

missing_df.head()
from sklearn.metrics import accuracy_score



np.random.seed(0)

y_pred_bench = np.random.randint(0, 2, size=y_test.size)



acc_bench = accuracy_score(y_pred=y_pred_bench, y_true=y_test)

print("Benchmark: %.2f" % acc_bench)
### function to get Cabin Class

def cabin_class(df, col_cabin="Cabin"):

    df[col_cabin] = df[col_cabin].str[0].fillna("No Cabin Assign")

    return df
#X_train = cabin_class(X_train.drop("PassengerId", axis=1))

#X_test = cabin_class(X_test.drop("PassengerId", axis=1))



X_train_2 = pd.get_dummies(X_train.drop(["Cabin", "Ticket", "PassengerId", "Name"], axis=1), prefix_sep="_")

X_test_2 = pd.get_dummies(X_test.drop(["Cabin", "Ticket", "PassengerId", "Name"], axis=1), prefix_sep="_")
### Age nan imputer

from sklearn.preprocessing import Imputer



imp = Imputer(strategy='mean')

imp.fit(X_train_2["Age"].reshape(-1, 1))

X_train_2["Age"] = imp.transform(X_train_2["Age"].reshape(-1, 1))

X_test_2["Age"] = imp.transform(X_test_2["Age"].reshape(-1, 1))



#X_train = X_train.dropna()

#X_test = X_test.dropna()
from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()

dtree.fit(X_train_2, y_train)

y_pred_dtree = dtree.predict(X_test_2)



acc_dtree = accuracy_score(y_pred=y_pred_dtree, y_true=y_test)

print("Decision Tree: %.2f" % acc_dtree)
from sklearn.ensemble import RandomForestClassifier



rforest = RandomForestClassifier(n_estimators=300, random_state=0)

rforest.fit(X_train_2, y_train)

y_pred_rforest = rforest.predict(X_test_2)



acc_rforest = accuracy_score(y_pred=y_pred_rforest, y_true=y_test)

print("Random Forest: %.2f" % acc_rforest)
from sklearn.linear_model import LogisticRegressionCV



lreg = LogisticRegressionCV(cv=10, random_state=0)

lreg.fit(X_train_2, y_train)

y_pred_lreg = lreg.predict(X_test_2)



acc_lreg = accuracy_score(y_pred=y_pred_lreg, y_true=y_test)

print("Logistic Regression: %.2f" % acc_lreg)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=20, weights='distance', p=1)

knn.fit(X_train_2, y_train)

y_pred_knn = knn.predict(X_test_2)



acc_knn = accuracy_score(y_pred=y_pred_knn, y_true=y_test)

print("KNN: %.2f" % acc_knn)
from sklearn.ensemble import VotingClassifier



rforest = RandomForestClassifier(n_estimators=500, random_state=0)

lreg = LogisticRegressionCV(cv=10, random_state=0)

knn = KNeighborsClassifier(n_neighbors=20, weights='distance', p=1)



voting = VotingClassifier(estimators=[('rforest', rforest), ('lreg', lreg), ('knn', knn)], 

                         voting='soft')

voting.fit(X_train_2, y_train)

y_pred_voting = voting.predict(X_test_2)



acc_voting = accuracy_score(y_pred=y_pred_voting, y_true=y_test)

print("Voting: %.2f" % acc_voting)
test_data = pd.read_csv("../input/test.csv")

X_test_data = pd.get_dummies(test_data.drop(["Cabin", "Ticket", "PassengerId", "Name"], axis=1), prefix_sep="_")

X_test_data["Age"] = imp.transform(X_test_data["Age"].reshape(-1, 1))



out = pd.DataFrame({"PassengerId": test_data["PassengerId"].values, 

                    "Survived": voting.predict(X_test_data.fillna(0))})
out.to_csv("voting.csv", index=False)