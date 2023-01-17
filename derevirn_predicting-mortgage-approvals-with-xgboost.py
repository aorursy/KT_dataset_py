#This project was part of the Microsoft Data Science Capstone, that was completed recently.

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import category_encoders as ce

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance

%matplotlib inline 

pd.options.display.max_columns = None



X_train = pd.read_csv("../input/train_values.csv")

y_train = pd.read_csv("../input/train_labels.csv")

train = pd.concat([X_train, y_train], axis=1)
train.count()
train.drop("row_id", axis=1, inplace=True)



replace_dict = {

    'msa_md': -1,

    'state_code': -1,

    'county_code': -1,

    'occupancy': 3,

    'preapproval': 3,

    'applicant_ethnicity': [3, 4, 5],

    "applicant_race": [6, 7, 8],

    "applicant_sex": [3, 4, 5]

}



cat_cols_few = ["loan_type", "property_type", "loan_purpose", "occupancy", "preapproval",

                "applicant_ethnicity", "applicant_race", "applicant_sex", "co_applicant"] 



cat_cols_many = ["msa_md", "state_code", "county_code", "lender"]



numerical_cols = ["loan_amount", "applicant_income", "population", "minority_population_pct",

                 "ffiecmedian_family_income", "tract_to_msa_md_income_pct",

                 "number_of_owner-occupied_units", "number_of_1_to_4_family_units"]



train.replace(replace_dict, np.nan, inplace = True)

train.count()
train["accepted"].value_counts().plot(kind='bar')

plt.title('Accepted loan applications')

plt.show()

train["accepted"].value_counts(normalize = 'index')

train[numerical_cols].hist(figsize=(12,10), bins=20)

plt.suptitle("Histograms of numerical values")

plt.show()



print("Skewness of numerical columns:")

train[numerical_cols].skew()
import math

to_log = ["loan_amount", "applicant_income", "number_of_owner-occupied_units", "number_of_1_to_4_family_units"]

train[to_log] = train[to_log].applymap(math.log)



train[numerical_cols].hist(figsize=(12,10), bins=20)

plt.suptitle("Histograms of numerical values")

plt.show()



print("Skewness of numerical columns after applying log function:")

train[numerical_cols].skew()
import warnings

warnings.filterwarnings("ignore")



fig, axes = plt.subplots(ncols = 2, nrows = 4, figsize = (12,14))

fig.subplots_adjust(hspace = 0.4, wspace = 0.2)

fig.suptitle("KDE plots of numerical features")



for ax, col in zip(axes.flatten(), numerical_cols) :

    sns.kdeplot(train[train["accepted"] == 0][col], shade="True", label="Not accepted", ax = ax)

    sns.kdeplot(train[train["accepted"] == 1][col], shade="True", label="Accepted", ax = ax)

    ax.set_xlabel(col)

fig, axes = plt.subplots(ncols = 3, nrows = 3, figsize = (14,14))

fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

fig.suptitle("Categorical features with low cardinality")



for ax, col in zip(axes.flatten(), cat_cols_few) :

    pd.crosstab(train[col], train["accepted"]).plot(kind="bar", ax = ax)

    ax.set_xlabel(col)
fig, axes = plt.subplots(ncols = 2, nrows = 2, figsize = (14,10))

fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

fig.suptitle("Categorical features with high cardinality")



for ax, col in zip(axes.flatten(), cat_cols_many) :

    sns.kdeplot(train[train["accepted"] == 0][col], shade="True", label="Not accepted", ax = ax)

    sns.kdeplot(train[train["accepted"] == 1][col], shade="True", label="Accepted", ax = ax)

    ax.set_xlabel(col)

train["minority_population"] = (train["minority_population_pct"] / 100) * (train["population"])

train["tract_family_income"] = (train["tract_to_msa_md_income_pct"] /100) * (train["ffiecmedian_family_income"])



train["minority_population"] = train["minority_population"].apply(math.log)
new_cols = ["minority_population", "tract_family_income"]



fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize = (14,5))

fig.subplots_adjust(hspace = 0.5, wspace = 0.3)

fig.suptitle("New features")



for ax, col in zip(axes.flatten(), new_cols) :

    sns.kdeplot(train[train["accepted"] == 0][col], shade="True", label="Not accepted", ax = ax)

    sns.kdeplot(train[train["accepted"] == 1][col], shade="True", label="Accepted", ax = ax)

    ax.set_xlabel(col)
plt.figure(figsize=(16,12))

sns.heatmap(train.corr().round(decimals=2), annot=True)

plt.title("Correlation heatmap")

plt.show()
to_log = ["loan_amount", "applicant_income", "number_of_owner-occupied_units",

          "number_of_1_to_4_family_units", "minority_population"]



to_drop = ["row_id", "number_of_1_to_4_family_units",

           "occupancy", "preapproval", "county_code"]



num_cols = ["loan_amount", "applicant_income", "population", "minority_population_pct",

            "ffiecmedian_family_income", "tract_to_msa_md_income_pct",

            "number_of_owner-occupied_units"]



cat_cols_few = ["loan_type", "property_type", "loan_purpose",

            "applicant_ethnicity", "applicant_race",

            "applicant_sex", "co_applicant"]



def prepare_data(df):

    

    df["co_applicant"] = df["co_applicant"].astype("int8")

    df.replace(replace_dict, np.nan, inplace = True)

    

    for col in num_cols:

        df[col].fillna(df[col].median(), inplace=True)

        

    for col in cat_cols_few:

        df[col].fillna(df[col].mode()[0], inplace=True)

        

    df["minority_population"] = (df["minority_population_pct"] / 100) * (df["population"])

    df["tract_family_income"] = (df["tract_to_msa_md_income_pct"] / 100) * (df["ffiecmedian_family_income"])



    df[to_log] = df[to_log].applymap(math.log)

    

    to_drop.extend(["minority_population_pct", "population",

                    "ffiecmedian_family_income", "tract_to_msa_md_income_pct"])

    df.drop(to_drop, axis=1, inplace=True)

    

    df = pd.get_dummies(df, columns = cat_cols_few)

    

    return df
X_train = prepare_data(X_train)



ce_target = ce.TargetEncoder(cols = ["lender", "msa_md", "state_code"], smoothing = 5, return_df = True)

X_train = ce_target.fit_transform(X_train, y_train["accepted"])
X_train, X_test, y_train, y_test = train_test_split(X_train.values, y_train["accepted"].values, test_size=0.3, random_state=0)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.6, gamma=0, learning_rate=0.02, max_delta_step=0,

       max_depth=8, min_child_weight=8, missing=None, n_estimators=600,

       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0.2, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=0.7)



model.fit(X_train, y_train)

prediction = model.predict(X_test)
print('The accuracy is:', metrics.accuracy_score(y_test, prediction))

print("\n")

print("Classification Report:")

print(metrics.classification_report(y_test, prediction))



sns.heatmap(confusion_matrix(y_test, prediction), annot=True)

plt.title("Confusion Matrix")

plt.show()