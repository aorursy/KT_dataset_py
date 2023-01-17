import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import RFECV, RFE

from sklearn.model_selection import ParameterGrid, GridSearchCV, KFold, cross_val_score

from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import catboost

random_state = 101
data_train = pd.read_csv("../input/titanic/train.csv").set_index("PassengerId")

data_test = pd.read_csv("../input/titanic/test.csv").set_index("PassengerId")

data = pd.concat([data_train, data_test])
data["Sex"] = OneHotEncoder(drop='if_binary').fit_transform(data[["Sex"]]).toarray().astype("int")
fig, [ax1, ax2] = plt.subplots(1, 2,sharey=False)

sns.countplot(x="Sex", hue="Survived", data=data, ax=ax1)

sns.countplot(x="Pclass", hue="Survived", data=data, ax=ax2)

sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="point", data=data)
FEATURES_BASELINE = ["Sex", "Pclass"]
def title(name):

    surname_split = name.split(", ")

    title_split = surname_split[1].split(". ")

    title = title_split[0]

    return title
data["Title"] = data.Name.apply(title)

data['Title'] = data['Title'].replace(['Dona', 'Mlle', 'Ms'], 'Miss')

data['Title'] = data['Title'].replace(['Lady', 'the Countess', 'Mme'], 'Mrs')

data['Title'] = data['Title'].replace(['Jonkheer', 'Don', 'Sir', 'Capt', 'Major', 'Col'], 'Mr')

data['Title'].value_counts()
def neighbors(tickets):

    tickets = tickets.to_frame()

    neighbors = (tickets.value_counts() - 1)

    result = tickets.replace({"Ticket": neighbors})["Ticket"]

    return result
data["Neighbors"] = neighbors(data["Ticket"])
sns.catplot(x="Neighbors", y="Survived", kind="point", data=data)
print("Passengers with Embarked NaNs")

print(data[pd.isnull(data.Embarked)])

embarked_nans_fare = data.loc[(data.Pclass == 1) & (data.Neighbors == 1), "Fare"].median()

print(f"Median fare for this type of passengers: {embarked_nans_fare}")
sns.catplot(x="Embarked", y="Fare", kind="box", data=data)

plt.axhline(embarked_nans_fare, linestyle='dashed', c='black',alpha = .3)
data.loc[pd.isnull(data.Embarked), "Embarked"] = "C"
print(data[pd.isnull(data.Age)])

data.loc[(data["Title"] == "Mr") & (data["Age"] < 12), "Title"] = "Master"

data.Age.fillna(data.groupby(['Title', 'Pclass', 'Sex']).transform('median').Age, inplace=True) 
print(data[(pd.isnull(data["Fare"])) | (data["Fare"] == 0)])

data.loc[data["Fare"] == 0, "Fare"] = np.NaN

data["Fare"].fillna(data.groupby(['Embarked', 'Pclass', 'Neighbors']).transform('median').Fare, inplace=True)
encoder = LabelEncoder()

data["Embarked"] = encoder.fit_transform(data["Embarked"])
sns.countplot(x="Embarked", hue="Survived", data=data)
FEATURES_1 = ['Sex', 'Pclass', 'Embarked']
data['Group_Status'] = 0

ticket_grouping = data.groupby("Ticket")

for _, group in ticket_grouping:

    if (len(group) > 1):

        for i, row in group.iterrows():

            s_max = group.drop(i)['Survived'].max()

            pass_id = row.name

            if s_max == 1.0:

                data.loc[pass_id, 'Group_Status'] = 1

            elif s_max == 0.0:

                data.loc[pass_id, 'Group_Status'] = -1
sns.countplot(x="Group_Status", hue="Survived", data=data)
FEATURES_2 = ['Sex', 'Pclass', 'Embarked', 'Group_Status']
sns.kdeplot(data.loc[(data["Survived"] == 0) & (data["Sex"] == 1), "Age"])

sns.kdeplot(data.loc[(data["Survived"] == 1) & (data["Sex"] == 1), "Age"])

# is_baby, is_boy and others

bins = np.append((data["Age"].min(), 1.0), np.linspace(12.0, data["Age"].max(), 6))

plt.vlines(bins, 0, 0.05, linestyles="dotted")

data["Age_Bin"] = np.digitize(data["Age"], bins)
FEATURES_3 = ['Sex', 'Pclass', 'Embarked', 'Group_Status', 'Age_Bin']
data["Fare_per_Person"] = data["Fare"] / (1 + data["Neighbors"])

bins = pd.qcut(data["Fare_per_Person"], 6, labels=False, retbins=True)

data["Fare_per_Person_Bin"] = bins[0]
sns.kdeplot(data.loc[data["Survived"] == 0, "Fare"])

sns.kdeplot(data.loc[data["Survived"] == 1, "Fare"])

plt.vlines(bins[1], 0, 0.05, linestyles="dotted")
FEATURES_4 = ['Sex', 'Pclass', 'Embarked', 'Group_Status', 'Age_Bin', 'Fare_per_Person_Bin']
data["Family_Size"] = data["SibSp"] + data["Parch"]

data["Connections"] = data[["Family_Size", "Neighbors"]].max(axis=1)

bins = np.array([data["Connections"].min(), 1, 4, data["Connections"].max()])

data["Connections_Bin"] = np.digitize(data["Connections"], bins)
sns.kdeplot(data.loc[data["Survived"] == 0, "Connections"])

sns.kdeplot(data.loc[data["Survived"] == 1, "Connections"])

plt.vlines(bins, 0, 1, linestyles="dotted")
FEATURES_5 = ['Sex', 'Pclass', 'Embarked', 'Group_Status', 'Age_Bin', 'Fare_per_Person_Bin', "Connections_Bin"]
def model(X, y, features, random_state, parameters=None):

    model = RandomForestClassifier(random_state=random_state, 

                                   n_estimators=500,

                                   min_samples_split=0.05)

    if parameters is not None:

        model.set_params(**parameters)

    

    model.fit(X[features], y)

    score_cv = cross_val_score(model, X[features], y, cv=cv).mean()

    print(f"Model CV score is {score_cv:.5f}")

    return model
X_train = data.loc[data_train.index].drop("Survived", axis=1)

y_train = data.loc[data_train.index, "Survived"]

X_test = data.loc[data_test.index].drop("Survived", axis=1)

cv=KFold(10, shuffle=True, random_state=random_state)
MODEL_BASELINE = model(X_train, y_train, FEATURES_BASELINE, random_state)

# 0.76097
MODEL_1 = model(X_train, y_train, FEATURES_1, random_state)

# 0.81145
MODEL_2 = model(X_train, y_train, FEATURES_2, random_state)

# 0.80476
MODEL_3 = model(X_train, y_train, FEATURES_3, random_state)

# 0.83060
MODEL_4 = model(X_train, y_train, FEATURES_4, random_state)

# 0.83171
MODEL_5 = model(X_train, y_train, FEATURES_5, random_state)

# 0.82498
FEATURES = ['Sex', 'Pclass', 'Embarked', 'Group_Status', 'Age_Bin', 'Fare_per_Person_Bin', "Connections_Bin"]
params = {

    'n_estimators': [500, 1000],

    'min_samples_split': [2, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10]

}



results = pd.DataFrame()

gs = GridSearchCV(estimator=RandomForestClassifier(random_state), param_grid=params, cv=cv, verbose=1, n_jobs=-1)

gs.fit(X_train[FEATURES], y_train)

results = pd.DataFrame(gs.cv_results_)



table = pd.pivot_table(results, values="mean_test_score", index="param_min_samples_split", columns="param_n_estimators")

sns.heatmap(table, annot=True, cmap="YlGnBu")

# min_samples_split: [0.01]
params = {

    'n_estimators': [500, 1000],

    'max_depth': [3, 4, 5, 6, 7, None],

}



results = pd.DataFrame()

gs = GridSearchCV(estimator=RandomForestClassifier(random_state), param_grid=params, cv=cv, verbose=1, n_jobs=-1)

gs.fit(X_train[FEATURES], y_train)

results = pd.DataFrame(gs.cv_results_)



table = pd.pivot_table(results, values="mean_test_score", index="param_max_depth", columns="param_n_estimators")

sns.heatmap(table, annot=True, cmap="YlGnBu")

# max_depth: 5
model = RandomForestClassifier(random_state=random_state)

pipe = Pipeline([("rfe", RFE(model, verbose=0)), ("rf", model)])



params = {

    'rfe__n_features_to_select': [1,2,3,4,5,6,7],

    'rf__n_estimators': [500],

    'rf__min_samples_split': [2, 0.01],

    'rf__max_depth':  [5, 6, None]

}

grid = GridSearchCV(pipe, param_grid=params, cv=cv, verbose=0, n_jobs=-1)

grid.fit(X_train[FEATURES], y_train)

print(gs.best_params_)

grid.cv_results_
MODEL_6 = grid.best_estimator_

print(f"Model CV score is {grid.best_score_:.5f}")

# 0.84742
MODEL_7 = catboost.CatBoostClassifier(one_hot_max_size=4, iterations=1000, random_seed=random_state, verbose=False)

MODEL_7.fit(X_train[FEATURES_4], y_train)

score_cv = cross_val_score(MODEL_7, X_train[FEATURES_4], y_train, cv=cv).mean()

print(f"Model CV score is {score_cv:.5f}")

# 0.83507
model = KNeighborsClassifier()

pipe = Pipeline([("scaler", StandardScaler()), ("knn", model)])

params = {

    'knn__n_neighbors': [3, 9, 15, 20, 21, 22, 25, 30],

    'knn__weights': ['uniform', 'distance']

}

knn_grid = GridSearchCV(pipe, param_grid=params, cv=cv, verbose=0, n_jobs=-1)

knn_grid.fit(X_train[FEATURES], y_train)

knn_grid.cv_results_
results = pd.DataFrame(knn_grid.cv_results_)

table = pd.pivot_table(results, values="mean_test_score", index="param_knn__n_neighbors", columns="param_knn__weights")

sns.heatmap(table, annot=True, cmap="YlGnBu")
MODEL_8 = knn_grid.best_estimator_

print(f"Model CV score is {knn_grid.best_score_:.5f}")

# 0.84069
MODEL_4.fit(X_train[FEATURES_4], y_train)

prediction_4 = MODEL_4.predict_proba(X_test[FEATURES_4])[:,1]

submit = pd.DataFrame({"PassengerId": X_test.index,"Survived": np.round(prediction_4, 0).astype(int)})

submit.to_csv("MODEL_4.csv",index=False)
MODEL_6.fit(X_train[FEATURES], y_train)

prediction_6 = MODEL_6.predict_proba(X_test[FEATURES])[:,1]

submit = pd.DataFrame({"PassengerId": X_test.index,"Survived": np.round(prediction_6, 0).astype(int)})

submit.to_csv("MODEL_6.csv",index=False)
MODEL_7.fit(X_train[FEATURES_4], y_train)

prediction_7 = MODEL_7.predict_proba(X_test[FEATURES_4])[:,1]

submit = pd.DataFrame({"PassengerId": data_test.index,"Survived": np.round(prediction_7, 0).astype(int)})

submit.to_csv("MODEL_7.csv",index=False)
MODEL_8.fit(X_train[FEATURES], y_train)

prediction_8 = MODEL_8.predict_proba(X_test[FEATURES])[:,1]

submit = pd.DataFrame({"PassengerId": data_test.index,"Survived": np.round(prediction_8, 0).astype(int)})

submit.to_csv("MODEL_8.csv",index=False)