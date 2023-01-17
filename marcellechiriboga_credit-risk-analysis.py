import numpy as np

import pandas as pd



from pandas_profiling import ProfileReport

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
train_df = pd.read_csv("../input/fia-ml-t5/train_data.csv")

test_df = pd.read_csv("../input/fia-ml-t5/teste_data.csv")
train_df.head()
train_df.info()
# Default vs. non-default 

counts = train_df.default.value_counts()

default = counts[True]

non_default = counts[False]

perc_default = (default / (default + non_default)) * 100

perc_non_default = (non_default/(default + non_default)) * 100

print("There were {} default loans ({:.3f}%) and {} non-default loans ({:.3f}%).".format(default, perc_default, non_default, perc_non_default))
print("Number of missing `default` values:",train_df["default"].isna().sum())
train_df = train_df.dropna(subset = ["default"])
profile = ProfileReport(train_df, title = "Pandas Profiling Report - ignoring missing values for `default`", html = {"style": {"full_width": True}})

profile.to_notebook_iframe()
# Save an HTML report

profile.to_file(output_file = "eda_report.html")
target = "default"



# Convert `default` values from boolean to integer

y_train = train_df.loc[:, target].astype("int")



# Remove `default` values from boolean to integer

X_train = train_df.loc[:, train_df.columns != target]
dropped_columns =  [

    "ids",

    "channel", # 1 unique value

    "credit_limit", # 31.3% missing values, low correlation with target variable

    "job_name", # 6.3% missing values, high cardinality

    "n_issues", # 26% missing values, high correlation with n_accounts                                         

    "ok_since", # 58.5% missing values                                         

    "reason", # high cardinality

    "zip", # high cardinality

]



X_train = X_train.drop(columns = dropped_columns)

X_test = test_df.drop(columns = dropped_columns)
def encode_categories(train_df, test_df, variables):

    """

    Encode categorical features as a one-hot numeric array.



    Keyword arguments:

    train_df -- the training dataset. Used to fit encoders.

    test_df -- the test dataset.

    variables -- the features to be encoded.

    """

    train_oh = pd.get_dummies(train_df, columns = variables, prefix = variables)

    test_oh = pd.get_dummies(test_df, columns = variables, prefix = variables)

    # left join ensures only columns existing on the training set are in both final data frames

    train_oh, test_oh = train_oh.align(test_oh, join = "left", axis = 1)

        

    train_df = train_df.merge(train_oh).drop(columns = variables)

    test_df = test_df.merge(test_oh).drop(columns = variables)

        

    return train_df, test_df
X_train, X_test = encode_categories(X_train, X_test, [

    "facebook_profile",

    "gender",

    "real_state",

    "score_1",

    "score_2",

    "sign",

    "state"

])  
def train_validation_split(X, y, validation_size = 0.2, random_state = 17):

    """

    Splits a dataframe of input features and its corresponding target labels into a training and test set



    Keyword arguments:

    X -- Dataframe of feature observations.

    y -- Series of target labels.

    validation_size -- the percentage of observations to be split.

    random_state -- random seed.

    """

    X_train, X_validation, y_train, y_validation = train_test_split(X, 

                                                                    y, 

                                                                    test_size = validation_size, 

                                                                    shuffle = True,

                                                                    random_state = random_state)

    return X_train, X_validation, y_train, y_validation
X_train, X_validation, y_train, y_validation = train_validation_split(X_train, y_train)
def fit_imputers(X, variables_strategies):

    """

    Fits and returns imputers to use for filling missing values.



    Keyword arguments:

    X -- Dataframe of feature observations.

    variables_strategies -- A dictionary of variable names to imputing strategies. e.g { "income": "median" }

                            valid strategies are ["mean", "median", "most_frequent", "constant"]

    """

    imputers = {}

    for variable, strategy in variables_strategies.items():

        imputer = SimpleImputer(strategy = strategy, copy = False)

        if variable in X.columns:

            imputer.fit(X[[variable]])

            imputers[variable] = imputer

    

    return imputers
imputers = fit_imputers(X_train, 

                        variables_strategies = {

                            "amount_borrowed": "median",

                            "borrowed_in_months": "most_frequent", # 2 possible values

                            "income": "median",

                            "n_accounts": "median",

                            "n_bankruptcies": "most_frequent", # 92% zeroes

                            "n_defaulted_loans": "most_frequent", # 99.6% zeroes                                          

                            "risk_rate": "median",

                            "score_3": "median",

                            "score_4": "median",

                            "score_5": "median",

                            "score_6": "median"

                        })
def impute_missing_values(X, imputers):

    """

    Imputs missing values on variables of X with the provided imputers.



    Keyword arguments:

    X -- Dataframe of feature observations.

    imputers -- Dictionary of variable names and pre-fit imputers.

    """

    for variable, imputer in imputers.items():

        if variable in X.columns:

            X[variable] = imputer.transform(X[[variable]])

    return X
X_train = impute_missing_values(X_train, imputers)

X_validation = impute_missing_values(X_validation, imputers)

X_test = impute_missing_values(X_test, imputers)
baseline_model = DummyClassifier(strategy = "most_frequent")

baseline_model.fit(X_train, y_train)
accuracy_score(y_validation, baseline_model.predict(X_validation))
def report_score(model, X_train, X_validation, y_train, y_validation):

    """

    Prints a model's train and validation scores, computed using the Area Under the ROC curve (AUROC).



    Keyword arguments:

    model -- the model to score. Must support predict_proba

    X_train -- training data (features).

    X_validation -- validation data (features).

    y_train -- training targets

    y_validation -- validation targets

    """

    y_hat_train_proba = model.predict_proba(X_train)[:,1]

    y_hat_validation_proba = model.predict_proba(X_validation)[:,1]

    

    print("Train ROC AUC score:", roc_auc_score(y_train, y_hat_train_proba, average = "weighted"))

    print("Validation ROC AUC score:", roc_auc_score(y_validation, y_hat_validation_proba, average = "weighted"))
report_score(baseline_model, X_train, X_validation, y_train, y_validation)
k = X_train.shape[1]

best_features = SelectKBest(score_func = chi2, k = k)

fit = best_features.fit(X_train, y_train)

scores = pd.DataFrame(fit.scores_)

features = pd.DataFrame(X_train.columns)



featureScores = pd.concat([features, scores], axis = 1)

featureScores.columns = ["Feature", "Score"]

# Print features ordered by score

print(featureScores.nlargest(k, "Score"))
drop_features = [

    "score_4", 

    "score_5", 

    "score_6"] + list(X_train.columns[X_train.columns.str.startswith("state_")])



X_train = X_train.drop(columns = drop_features)

X_validation = X_validation.drop(columns = drop_features)

X_test = X_test.drop(columns = drop_features)
# Scale the data for use with logistic regression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(X_validation)
lr_model = LogisticRegression(random_state = 17, 

                              solver = "lbfgs", 

                              C = 1, 

                              class_weight = "balanced")

lr_model.fit(X_train_scaled, y_train)
report_score(lr_model, X_train_scaled, X_validation_scaled, y_train, y_validation)
dt_model = DecisionTreeClassifier(max_depth = 8,

                                  class_weight = "balanced",

                                  random_state = 17)

dt_model.fit(X_train, y_train)
report_score(dt_model, X_train, X_validation, y_train, y_validation)
rf_model = RandomForestClassifier(n_estimators = 200,

                                  max_depth = 15,

                                  criterion = "gini",

                                  class_weight = "balanced",

                                  min_samples_leaf = 20,

                                  n_jobs = -1,

                                  random_state = 17)

rf_model.fit(X_train, y_train)
report_score(rf_model, X_train, X_validation, y_train, y_validation)
xgb_model = XGBClassifier(n_estimators = 300,

                          scale_pos_weight = y_train[y_train == 0].count() / y_train[y_train == 1].count(),

                          objective = "binary:logistic",

                          random_state = 17,

                          n_jobs = -1)



xgb_model.fit(X_train, y_train)
report_score(xgb_model, X_train, X_validation, y_train, y_validation)
# xgb_params = {

#     "max_depth": [3, 4, 5, 6],

#     "gamma": [0, 0.1, 0.3, 0.5],

#     "min_samples_leaf": [2, 5, 7, 10]

# }



# grid_search = GridSearchCV(estimator = xgb_model, 

#                            param_grid = xgb_params, 

#                            cv = 3,

#                            scoring = "roc_auc_ovo_weighted",

#                            n_jobs = -1)



# grid_search.fit(X_train, y_train)



# print(grid_search.best_params_)
optimized_xgb_model = XGBClassifier(n_estimators = 300,

                                    max_depth = 3,

                                    scale_pos_weight = y_train[y_train == 0].count() / y_train[y_train == 1].count(),

                                    objective = "binary:logistic",

                                    gamma = 0.1,

                                    min_samples_leaf = 2,

                                    random_state = 17,

                                    n_jobs = -1)



voting_model = VotingClassifier(estimators = [("xgb", optimized_xgb_model), ("lr", make_pipeline(StandardScaler(), lr_model))],

                                voting = "soft",

                                n_jobs = -1)



voting_model.fit(X_train, y_train)
report_score(voting_model, X_train, X_validation, y_train, y_validation)
predictions = voting_model.predict_proba(X_test)
y_hat_test = predictions[:,1]
test_predictions = pd.DataFrame(data = {

    "ids": test_df["ids"], 

    "prob": y_hat_test

})
test_predictions.head()
# Save model predictions to .csv file

test_predictions.to_csv("predictions.csv", index = False)