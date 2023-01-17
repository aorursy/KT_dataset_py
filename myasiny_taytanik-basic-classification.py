import os

import warnings

import numpy as np

import pandas as pd



# Ignore warnings



warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None



# Show input data files



for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Train set



df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.shape
# Test set



df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.shape
# Combine features of test and train sets



df_both = df_train.loc[:, df_train.columns != "Survived"].append(df_test, ignore_index=True)

df_both.shape
df_both.head()
df_both.describe()
# Replace null values



def fill_na(df, val):

    return df.fillna(val, inplace=True)
# Drop rows containing null values



def drop_na(df):

    return df.dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)
import re



# Check if a value exists in given columns



def val_exist(df, cols, val):

    for col in cols:

        if isinstance(val, str) and df[col].str.contains(val, na=False, flags=re.IGNORECASE).any():

            return True

        elif (isinstance(val, int) or isinstance(val, float)) and any(df[col] == val):

            return True

    return False
# Get title info from passenger name



df_train["Title"] = df_train["Name"].apply(lambda x: x.split(".")[0].split(",")[1].strip())
df_train[["Name", "Title"]].head()
# Get family size



df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1
df_train[["SibSp", "Parch", "FamilySize"]].head()
# Drop passenger name and id columns



df_train = df_train.drop(["Name", "PassengerId"], axis=1)
# Columns containing text values

str_cols = ["Sex", "Ticket", "Cabin", "Embarked", "Title"]



val_exist(df_train, str_cols, "Bilinmiyor")
# Replace null values with a unique value



for col in str_cols:

    fill_na(df_train[col], "Bilinmiyor")
# Columns containing numeric values

int_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]



for col in int_cols:

    print(val_exist(df_train, [col], df_train[col].mean(skipna=True)))
# Replace null values with column mean



for col in int_cols:

    fill_na(df_train[col], df_train[col].mean(skipna=True))
import seaborn as sns



# Visualization of survival by age



graph = sns.FacetGrid(df_train, col="Survived") 

graph = graph.map(sns.distplot, "Age")
from sklearn.preprocessing import LabelBinarizer



# Convert text values to binary lists



def binarize(df):

    return LabelBinarizer().fit_transform(df).tolist()
# Convert binary lists to base-2 numbers



def base_two(df):

    return df.apply(lambda x: int("".join("{0}".format(i) for i in x), 2))
from sklearn.preprocessing import StandardScaler



# Scale values column-wise



def scaler(df):

    return StandardScaler().fit(df)
# Replace outliers with null values



def outlier(df):

    return df[np.abs(df - df.mean()) <= (3 * df.std())]
for col in str_cols:

    df_train[col] = binarize(df_train[col])

    df_train[col] = base_two(df_train[col])
# Feature columns



df_feature = df_train.loc[:, df_train.columns != "Survived"]



# Scale values column-wise



std_scaler = scaler(df_feature)

df_feature.iloc[:,:] = std_scaler.transform(df_feature)



# Append labels



df_feature["Survived"] = df_train["Survived"]



df_feature.shape
# Columns containing features



all_cols = ["Sex", "Ticket", "Cabin", "Embarked", "Title", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]



# Detect outliers



for col in all_cols:

    df_feature[col] = outlier(df_feature[col])

    

# Remove rows containing outliers



df_feature = drop_na(df_feature)



df_feature.shape
from sklearn.model_selection import train_test_split



# Split train set into X and y



X, y = df_feature.loc[:, df_feature.columns != "Survived"], df_feature["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train.shape
X_train.head()
from sklearn.model_selection import GridSearchCV



# Find best parameters for given data and classifier



def best_params(classifier, parameters, X, y):

    clf = GridSearchCV(classifier, parameters)

    clf.fit(X, y)

    return clf.best_params_
from sklearn.metrics import accuracy_score



# Calculate accuracy of predictions by expected label set



def accuracy(name, y, predictions):

    return "Accuracy of {0}: {1}".format(name, accuracy_score(y, predictions))
from sklearn.svm import SVC



params_svc = {"C":[1, 10], "gamma":("scale", "auto"), "kernel":("linear", "rbf")}

best_params(SVC(), params_svc, X_train, y_train)
# Train SVC



clf_svc = SVC(C=1, gamma="auto", kernel="rbf")

clf_svc.fit(X_train, y_train)
print(accuracy("SVC", y_test, clf_svc.predict(X_test)))
from sklearn.neighbors import KNeighborsClassifier



params_kn = {"algorithm":("auto", "brute", "kd_tree"), "n_neighbors":[1, 3, 5]}

best_params(KNeighborsClassifier(), params_kn, X_train, y_train)
# Train K-Neighbors



clf_kn = KNeighborsClassifier(algorithm="brute", n_neighbors=5)

clf_kn.fit(X_train, y_train)
print(accuracy("K-Neighbors", y_test, clf_kn.predict(X_test)))
from sklearn.ensemble import RandomForestClassifier



params_rf = {"class_weight":("balanced", "balanced_subsample"), "max_features":("auto", "log2"), "n_estimators":[10, 100]}

best_params(RandomForestClassifier(), params_rf, X_train, y_train)
# Train Random Forest



clf_rf = RandomForestClassifier(class_weight="balanced_subsample", max_features="log2", n_estimators=100)

clf_rf.fit(X_train, y_train)
print(accuracy("Random Forest", y_test, clf_rf.predict(X_test)))
from sklearn.ensemble import GradientBoostingClassifier



params_gb = {"criterion":("friedman_mse", "mse", "mae"), "loss":("deviance", "exponential"), "max_features":("auto", "log2")}

best_params(GradientBoostingClassifier(), params_gb, X_train, y_train)
# Train Gradient Boosting



clf_gb = GradientBoostingClassifier(criterion="friedman_mse", loss="deviance", max_features="auto")

clf_gb.fit(X_train, y_train)
print(accuracy("Gradient Boosting", y_test, clf_gb.predict(X_test)))
from sklearn.naive_bayes import GaussianNB



# Train Gaussian Naive Bayes



clf_nb = GaussianNB()

clf_nb.fit(X_train, y_train)
print(accuracy("Naive Bayes", y_test, clf_nb.predict(X_test)))
# Go on with the most accurate estimator



clf = clf_kn
df_test["Title"] = df_test["Name"].apply(lambda x: x.split(".")[0].split(",")[1].strip())
df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1
for col in str_cols:

    fill_na(df_test[col], "Bilinmiyor")
for col in int_cols:

    fill_na(df_test[col], df_test[col].mean(skipna=True))
for col in str_cols:

    df_test[col] = binarize(df_test[col])

    df_test[col] = base_two(df_test[col])
data = df_test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "Title", "FamilySize"]]

data.iloc[:,:] = std_scaler.transform(data)
data.shape
data.head()
# Survival predictions by main estimator



pred = clf.predict(data)
# Match survival info with passenger id



result = {}

for i in range(len(pred)):

    result[df_test.iloc[i]["PassengerId"]] = pred[i]
import csv



# Export the result as csv file



with open("result.csv", "w") as f:

    writer = csv.DictWriter(f, fieldnames=["PassengerId", "Survived"])

    writer.writeheader()

    for key, val in result.items():

        writer.writerow({"PassengerId": key, "Survived": val})
df_result = pd.read_csv("result.csv")

df_result.head()
# Distribution of survival info



df_result.groupby(["Survived"]).count().plot(kind="bar")