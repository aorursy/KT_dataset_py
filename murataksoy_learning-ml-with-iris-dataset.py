# required libraies



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split #for splitting dataset into train,test 

from sklearn.impute import SimpleImputer #for missing values

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# reading dataset

iris = pd.read_csv("../input/iris-nan/data_with_nans.csv")
# exploring dataset

iris.head()
iris.columns
#dropping unnecessary "Unnamed: 0" column

iris.drop(columns="Unnamed: 0", inplace=True)
for c in iris.columns[1:-1]:

    plt.figure(figsize=(12,8))

    sns.scatterplot(data=iris, x="Id", y=c, hue="Species")

    plt.show();
#clean outliers: 3 sigma method

"""

for c in iris.columns[1:-1]:

    for s in iris["Species"].unique():

        spec = iris[iris["Species"] == s]

        col = spec[c]

        std = col.std()

        avg = col.mean()

        three_sig_plus = avg + 3*std

        three_sig_minus = avg - 3*std

        outlier = col[((spec[c] > three_sig_plus) | (spec[c] < three_sig_minus))].index

        iris.drop(index=outlier, inplace=True)

"""
#clean outliers: IQR method (alternative)

for c in iris.columns[1:-1]:

    for s in iris["Species"].unique():

        spec = iris[iris["Species"] == s]

        col = spec[c]

        q1 = col.quantile(0.25)

        q3 = col.quantile(0.75)

        iqr = q3 - q1

        minimum = q1 - (1.5*iqr)

        maximum = q3 + (1.5*iqr)

        outlier = col[((spec[c] > maximum) | (spec[c] < minimum))].index

        iris.drop(index=outlier, inplace=True)

        print(outlier)
#cleaned dataset

for c in iris.columns[1:-1]:

    plt.figure(figsize=(12,8))

    sns.scatterplot(data=iris, x="Id", y=c, hue="Species")

    plt.show();
#set index to Id column

iris.set_index("Id", inplace=True)
iris.describe()
iris.groupby("Species").describe().T
iris.info()
#setting X and y 

X = iris.select_dtypes(include=["float64"])

y = iris.select_dtypes(include=["object"])
X.head()
y.head()
#label encoding

le = LabelEncoder()

y=le.fit_transform(y)

y
#we have some nan values

X.isna().sum()
# fill na in in X data using SimpleImputer

x_column_names = X.columns

imputer = SimpleImputer(strategy="most_frequent")

imputer = imputer.fit(X)

X = imputer.transform(X) 

X = pd.DataFrame(X)

X.columns = x_column_names # put back column names
X.head()
#now, no nan values

X.isna().sum()
#spliting dataset for training and test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=61)
#RandomForestClassifier model

model = RandomForestClassifier(random_state=61)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(accuracy_score(y_test, preds))

print(confusion_matrix(y_test,preds))
"""

trying with different hyperparameters

params = {"min_samples_split":[1,3,5,8], "max_depth":[1,3,5,8], "n_estimators":[100,200,500,1000], "max_features":[1,3,5,8]}

tuned = GridSearchCV(model, params, cv=5, n_jobs=-1 ,verbose=2)

tuned.fit(X_train,y_train)

preds=tuned.predict(X_test)

print(accuracy_score(y_test, preds))

print(tuned.best_params_)

"""
#KNeighborsClassifier model

model = KNeighborsClassifier()

model.fit(X_train, y_train)

preds=model.predict(X_test)

print(accuracy_score(y_test, preds))

print(confusion_matrix(y_test,preds))
"""

trying with different hyperparameters

params = {"leaf_size":[2,3,5,10,20], "n_neighbors":[3,5,7]}

tuned = GridSearchCV(model, params, cv=10, n_jobs=-1 ,verbose=2)

tuned.fit(X_train,y_train)

preds=tuned.predict(X_test)

print(accuracy_score(y_test, preds))

print(tuned.best_params_)

"""