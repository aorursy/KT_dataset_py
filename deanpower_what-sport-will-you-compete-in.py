import numpy as np

import pandas as pd

import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    classification_report,

    f1_score,

    make_scorer,

    precision_recall_fscore_support,

    precision_score,

    recall_score,

)

from sklearn.model_selection import (

    GridSearchCV,

    cross_validate,

    train_test_split,

)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
df.isna().mean().round(4) * 100
df_cats = (

    df[["Sex", "Season"]]

    .melt()

    .groupby(["variable", "value"])

    .size()

    .to_frame(name="n")

)

df_cats["proportion"] = df_cats["n"].div(df_cats.n.sum(level=0), level=0)

df_cats
sex_per_year = (

    df.groupby(["Year", "Sex"])["ID"].count() /

    df.groupby(["Year"])["ID"].count()

)

sex_per_year = sex_per_year.reset_index()

sex_per_year = sex_per_year[sex_per_year["Sex"] == "M"].rename(

    columns={"ID": "Male_Porportion"}

)

sns.lineplot(x="Year", y="Male_Porportion", data=sex_per_year)
sns.pairplot(df.drop(columns=["ID"]))
df.drop(columns=["ID"]).corr()
df["Sport"].value_counts().head(10)
# import csv, filtered this time.

df = pd.read_csv(

    "../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv",

    usecols=["Season", "Year", "Sex", "Age",

             "Height", "Weight", "Sport", "Event"],

).dropna()



# remove Winter Olympics for simplicity.

df = df[df["Season"] == "Summer"]



# remove sports without stongly physique-dependent elements

df = df[

    ~df["Sport"].isin(

        ["Shooting", "Art Competitions", "Motorboating", "Sailing", "Equestrianism"]

    )

]



# Split athletics into sprints/jumps, throws and endurance to increase accuracy and reduce imbalance

df.loc[

    (df["Sport"] == "Athletics")

    & (

        df["Event"].str.contains(

            "jump|vault|60 |100 |200 |400 |athlon|all-round", case=False

        )

    ),

    "Sport",

] = "Athletics Sprints"

df.loc[

    (df["Sport"] == "Athletics") & (

        df["Event"].str.contains("put|throw", case=False)),

    "Sport",

] = "Athletics Throws"

df.loc[

    (df["Sport"] == "Athletics")

    & ~(

        df["Event"].str.contains(

            "jump|vault|60 |100 |200 |400 |athlon|all-round", case=False

        )

    )

    & ~(df["Event"].str.contains("put|throw", case=False)),

    "Sport",

] = "Athletics Endurance"

# just remove multi-events for simplicity

df = df[

    ~(

        (df["Sport"] == "Athletics")

        & (df["Event"].str.contains("athlon|all-round", case=False))

    )

]

df["Sport"] = df["Sport"].str.replace("Men's |Women's ", "")



# Pick top 10 sports by athlete count

df = df[df["Sport"].isin(df["Sport"].value_counts().head(10).index)]

df = df.drop(columns=["Season", "Event"])
df["Sport"].value_counts()
sns.lineplot(

    x="Year",

    y="count",

    hue="Sport",

    data=df.groupby("Year")["Sport"]

    .value_counts()

    .to_frame()

    .rename(columns={"Sport": "count"})

    .reset_index(),

)
sns.lineplot(x="Year", y="Height", hue="Sport", data=df)
sns.lineplot(x="Year", y="Weight", hue="Sport", data=df)
df = df[df["Year"] >= 1960]
df["Sport"].value_counts()
df = pd.get_dummies(df, columns=["Sex"]).drop(

    columns=["Sex_F"]

)  # convert Sex to just Male or not-Male
le = LabelEncoder()

df["Sport"] = le.fit_transform(df["Sport"])
df_train, df_test = train_test_split(df, train_size=0.7, random_state=48)



df_train_X = df_train.drop("Sport", axis=1)

min_max_scaler = MinMaxScaler().fit(df_train_X)  # used to normalise later

df_train_Y = df_train["Sport"]



df_test_X = df_test.drop("Sport", axis=1)

df_test_Y = df_test["Sport"]
len(df_train_Y)
train_counts = df_train_Y.value_counts()

train_counts.index = le.inverse_transform(train_counts.index)

train_counts
len(df_test_Y)
test_counts = df_test_Y.value_counts()

test_counts.index = le.inverse_transform(test_counts.index)

test_counts
scoring = {

    "accuracy": "accuracy",

    "weighted_precision": make_scorer(precision_score, average="weighted"),

    "weighted_recall": make_scorer(recall_score, average="weighted"),

    "weighted_F1": make_scorer(f1_score, average="weighted"),

}



classifiers = [

    (

        "DT",

        DecisionTreeClassifier(),

        {"max_depth": [3, 5, 10, None]},

    ),  # trying different max tree depths.

    (

        "RF",

        RandomForestClassifier(),

        {"max_depth": [3, 5, 10, None]},

    ),  # trying different max tree depths.

    (

        "LOGREG",

        LogisticRegression(),

        {"C": np.logspace(-5, 5, 5 + 5 + 1, base=10)},

    ),  # C ranges from 10^-5 - 10^5 in powers of 10.

    (

        "KNN",

        # number of nearest neighbours ranges from 1 to 1000 in powers of 10.

        KNeighborsClassifier(),

        {

            "n_neighbors": np.append(

                np.logspace(0, 3, 3 + 0 + 1, base=10).astype("int"),

                np.sqrt(len(df_train_X)).astype("int"),

            )

        },

    ),

    (

        "SVM",

        LinearSVC(),

        {"C": np.logspace(-5, 5, 5 + 5 + 1, base=10)},

    )  # C ranges from 10^-5 - 10^5 in powers of 10.

]



results = pd.DataFrame([])

models = []
for name, classifier, params in classifiers:



    if name in ("SVM", "LOGREG", "KNN"):

        train_X = min_max_scaler.transform(df_train_X)

        train_Y = df_train_Y

    else:

        train_X = df_train_X

        train_Y = df_train_Y



    clf = GridSearchCV(

        estimator=classifier,

        param_grid=params,

        scoring=scoring,

        cv=None,

        n_jobs=-1,

        refit="weighted_F1",

        verbose=3,

    )



    print("model = " + str(name))

    fit = clf.fit(train_X, train_Y)

    models.append((name, fit.best_estimator_))

    search = pd.DataFrame.from_dict(fit.cv_results_)[

        [

            "params",

            "mean_test_accuracy",

            "mean_test_weighted_precision",

            "mean_test_weighted_recall",

            "mean_test_weighted_F1",

        ]

    ]

    search["model"] = name

    search.columns = search.columns.str.replace("mean_test_", "")



    # baseline classifier

    dum_class = DummyClassifier("uniform", random_state=48)

    dum = cross_validate(dum_class, train_X, train_Y, cv=5, scoring=scoring)

    dum = pd.DataFrame.from_dict(dum).drop(columns=["fit_time", "score_time"])

    dum["model"] = name

    dum = dum.assign(**dum.mean()).iloc[[0]]

    dum.columns = dum.columns.str.replace("test_", "base_")



    search = pd.merge(search, dum, how="left", on=["model"])



    results = results.append(search, ignore_index=True)
best_models = results.loc[results.groupby("model")["weighted_F1"].idxmax()]

best_models
for model_name, model in models:



    test_X = df_test_X

    test_Y = df_test_Y



    Y_pred = model.predict(test_X)

    Y_pred = le.inverse_transform(Y_pred)

    Y_actual = test_Y

    Y_actual = le.inverse_transform(Y_actual)

    print("Classification Report:    " + model_name)

    print(classification_report(Y_actual, Y_pred))

    print(

        "Overall:    "

        + str(precision_recall_fscore_support(Y_actual, Y_pred, average="weighted"))

    )
df_train, df_test = train_test_split(df, train_size=0.7, random_state=48)



df_train_X = df_train.drop("Sport", axis=1)

min_max_scaler = MinMaxScaler().fit(df_train_X)  # used to normalise later

df_train_Y = df_train["Sport"]



# number of athletes per sport is imbalanced, use SMOTE to balance classes

sm = SMOTE(random_state=42)

df_train_X_SMOTE, df_train_Y_SMOTE = sm.fit_resample(df_train_X, df_train_Y)



df_test_X = df_test.drop("Sport", axis=1)

df_test_Y = df_test["Sport"]
len(df_train_Y_SMOTE)
train_counts = df_train_Y_SMOTE.value_counts()

train_counts.index = le.inverse_transform(train_counts.index)

train_counts
len(df_test_Y)
test_counts = df_test_Y.value_counts()

test_counts.index = le.inverse_transform(test_counts.index)

test_counts
scoring = {

    "accuracy": "accuracy",

    "weighted_precision": make_scorer(precision_score, average="weighted"),

    "weighted_recall": make_scorer(recall_score, average="weighted"),

    "weighted_F1": make_scorer(f1_score, average="weighted"),

}



classifiers = [

    ("RF", RandomForestClassifier(), {"max_depth": [3, 5, 10, None]})

]  # trying different max tree depths.



SMOTE_results = pd.DataFrame([])

SMOTE_models = []
for name, classifier, params in classifiers:



    train_X = df_train_X_SMOTE

    train_Y = df_train_Y_SMOTE



    clf = GridSearchCV(

        estimator=classifier,

        param_grid=params,

        scoring=scoring,

        cv=None,

        n_jobs=-1,

        refit="weighted_F1",

        verbose=3,

    )



    print("model = " + str(name))

    fit = clf.fit(train_X, train_Y)

    SMOTE_models.append((name, fit.best_estimator_))

    search = pd.DataFrame.from_dict(fit.cv_results_)[

        [

            "params",

            "mean_test_accuracy",

            "mean_test_weighted_precision",

            "mean_test_weighted_recall",

            "mean_test_weighted_F1",

        ]

    ]

    search["model"] = name

    search.columns = search.columns.str.replace("mean_test_", "")



    # baseline classifier

    dum_class = DummyClassifier("uniform", random_state=48)

    dum = cross_validate(dum_class, train_X, train_Y, cv=5, scoring=scoring)

    dum = pd.DataFrame.from_dict(dum).drop(columns=["fit_time", "score_time"])

    dum["model"] = name

    dum = dum.assign(**dum.mean()).iloc[[0]]

    dum.columns = dum.columns.str.replace("test_", "base_")



    search = pd.merge(search, dum, how="left", on=["model"])



    SMOTE_results = SMOTE_results.append(search, ignore_index=True)
SMOTE_results.loc[SMOTE_results.groupby("model")["weighted_F1"].idxmax()]
for model_name, model in SMOTE_models:



    test_X = df_test_X

    test_Y = df_test_Y



    Y_pred = model.predict(test_X)

    Y_pred = le.inverse_transform(Y_pred)

    Y_actual = df_test_Y

    Y_actual = le.inverse_transform(Y_actual)

    print("Classification Report:    " + model_name)

    print(classification_report(Y_actual, Y_pred))

    print(

        "Overall:    "

        + str(precision_recall_fscore_support(Y_actual, Y_pred, average="weighted"))

    )
RF = models[1][1]

RF.fit(df_train_X, df_train_Y)



features = {}



for feature, importance in zip(df_train_X.columns, RF.feature_importances_):

    features[feature] = importance



importances = (

    pd.DataFrame.from_dict(features, orient="index")

    .reset_index()

    .rename(columns={"index": "Attribute", 0: "Gini Importance"})

    .sort_values(by="Gini Importance", ascending=False)

)



sns.barplot(x="Attribute", y="Gini Importance", data=importances, color="blue")
data = {"Age": 24, "Height": 182, "Weight": 79, "Year": 2021, "Sex_M": 1}



pred_event = RF.predict(pd.DataFrame([data]))

all_events = le.inverse_transform(np.arange(0, df["Sport"].nunique(), 1))

all_probs = RF.predict_proba(pd.DataFrame([data]))[0]

pred_proba = all_probs[pred_event]

print(

    "Your predicted sport for the "

    + str(data["Year"])

    + " Olympic Games is "

    + str(le.inverse_transform(pred_event)[0])

    + " with a probability of "

    + str(round(100 * pred_proba[0], 1))

    + "%"

)

print("All probabilities:")

print([i for i in zip(all_events, all_probs)])