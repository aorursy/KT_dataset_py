# Python Imports

!pip install catboost

!pip install xgboost



import pandas as pd

import numpy as np # Linear Algebra

import random, time, datetime



# Data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Data Preprocessing

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import PCA



# Machine Learning

from sklearn import model_selection, tree, preprocessing, metrics

from sklearn import linear_model

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from catboost import Pool, CatBoostRegressor

from sklearn.svm import LinearSVC

from sklearn import svm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier

import xgboost as xgb



# Validation & Scoring

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn import svm



from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, make_scorer, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



sns.set_style("whitegrid")

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
american = pd.read_csv("/kaggle/input/american-bank-data/Bank_of_America_data.csv")
american.shape
american.head()
american.tail()
american.describe()
american.isna().sum()
american.boxplot(figsize=(10,7));
#sns.pairplot(american, hue = "BAD");
correlation = abs(american.corr())

# Generate a mask for the upper triangle

mask = np.zeros_like(correlation, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
with sns.axes_style('white'):

    sns.jointplot("MORTDUE", "VALUE", american, kind='reg', color="purple")
#american.groupby(by="JOB").mean()["DEBTINC"].plot.bar();
sns.barplot(x="JOB", y="DEBTINC", data=american);
#american.groupby(by="JOB").sum()["DEBTINC"].plot.bar();
sns.barplot(x="JOB", y="DEBTINC", data=american, estimator=sum);
#american.groupby(by="JOB").mean()["LOAN"].plot.bar();
sns.barplot(x="JOB", y="LOAN", data=american);
#american.groupby(by="JOB").mean()["VALUE"].plot.bar();
#american.groupby(by="JOB").mean()["LOAN"].plot.bar();

sns.barplot(x="JOB", y="VALUE", data=american);
american.groupby(by="JOB").median()["LOAN"].plot.bar();

#sns.barplot(x="JOB", y="VALUE", data=american, estimator=sum);
effectifs = american["JOB"].value_counts()

modalites = effectifs.index # The index of the effectifs contains the modalities



tab = pd.DataFrame(modalites, columns = ["JOB"]) # Creating an array with the modalities

tab["Effectif"] = effectifs.values

#tab["f"] = tab["n"] / len(american) # len(data) returns the size of the sample

tab["Mean_loan"] = american.groupby(by="JOB").mean()["LOAN"].values

tab["Mean_value"] = american.groupby(by="JOB").mean()["VALUE"].values

tab["Difference"] = tab["Mean_value"] - tab["Mean_loan"]

tab
sns.lineplot(x="Mean_value", y="Mean_loan", data=tab);
sns.barplot(x="REASON", y="DEBTINC", data=american);
sns.barplot(x="REASON", y="VALUE", data=american);
sns.barplot(x="REASON", y="LOAN", data=american);
american[american["REASON"] == "DebtCon"]["JOB"].value_counts(normalize=True).plot(kind='pie');
american[american["REASON"] == "HomeImp"]["JOB"].value_counts(normalize=True).plot(kind='pie');
sns.boxplot(data=american, x="JOB", y="VALUE");
american["BAD"].unique()
american["BAD"].value_counts()
american["BAD"].hist();
american["BAD"].value_counts(normalize=True).plot(kind='pie');
american[["LOAN"]].boxplot();
american[["LOAN"]].hist();
american["LOAN"].describe()
american["MORTDUE"].isna().sum()
american[["MORTDUE"]].boxplot();
american["MORTDUE"].hist();
american["VALUE"].isna().sum()
american[["VALUE"]].boxplot();
american["VALUE"].hist();
american["REASON"].isna().sum()
american["REASON"].unique()
american.groupby("REASON").size().plot(kind='bar');
american["REASON"].value_counts()
sns.countplot(data=american, x="REASON");
american["REASON"].value_counts(normalize=True).plot(kind='pie');
american["JOB"].isna().sum()
american["JOB"].unique()
american.groupby("JOB").size().plot(kind='bar');
american["JOB"].value_counts()
sns.countplot(data=american, x="JOB");
american["JOB"].value_counts(normalize=True).plot(kind='pie');
american["YOJ"].isna().sum()
american[["YOJ"]].boxplot();
american[["YOJ"]].hist();
american["DEROG"].isna().sum()
american[["DEROG"]].boxplot();
american[["DEROG"]].hist(); # This expplains the outlier values !
american["DELINQ"].isna().sum()
american[["DELINQ"]].boxplot()
american[["DELINQ"]].hist()
american["DELINQ"].value_counts()
american["CLAGE"].isna().sum()
american[["CLAGE"]].boxplot();
american[["CLAGE"]].hist();
american[["NINQ"]].boxplot();
american[["NINQ"]].hist();
american["NINQ"].value_counts()
american[["NINQ"]].boxplot();
american[["NINQ"]].hist();
american["NINQ"].value_counts()
american[["DEBTINC"]].boxplot();
american[["DEBTINC"]].hist();
with_median = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='median')

)

median_vars = ["LOAN", "MORTDUE", "VALUE", "DEBTINC", "CLAGE"]

american[median_vars] = with_median.fit_transform(american[median_vars])
# Converting Discrete columns to type object

discrete_columns = ["REASON", "JOB", "DEROG", "DELINQ", "NINQ", "CLNO", "YOJ"]

american[discrete_columns] = american[discrete_columns].astype(object)
american.dtypes
discrete_columns
most_frequent = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='most_frequent')

)

most_frquent_value_vars = discrete_columns

american[discrete_columns] = most_frequent.fit_transform(american[discrete_columns])
american.head()
american.isna().sum()
std_scaling = make_pipeline(

    StandardScaler()

)

std_vars = ["LOAN", "MORTDUE", "VALUE", "CLAGE", "DEBTINC"]

american[std_vars] = std_scaling.fit_transform(american[std_vars])



minmax_scaling = make_pipeline(

    MinMaxScaler()

)

minmax_vars = ["YOJ"]

american[minmax_vars] = minmax_scaling.fit_transform(american[minmax_vars])
american.head()
american2 = pd.read_csv("/kaggle/input/american-bank-data/Bank_of_America_data.csv")

with_median = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='median')

)

median_vars = ["LOAN", "MORTDUE", "VALUE", "DEBTINC", "CLAGE"]

american2[median_vars] = with_median.fit_transform(american2[median_vars])

# Converting Discrete columns to type object

discrete_columns = ["REASON", "JOB", "DEROG", "DELINQ", "NINQ", "CLNO", "YOJ"]

american2[discrete_columns] = american2[discrete_columns].astype(object)

most_frequent = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='most_frequent')

)

most_frquent_value_vars = discrete_columns

american2[discrete_columns] = most_frequent.fit_transform(american2[discrete_columns])

std_scaling = make_pipeline(

    RobustScaler()

)

std_vars = ["LOAN", "MORTDUE", "VALUE", "CLAGE", "DEBTINC"]

american2[std_vars] = std_scaling.fit_transform(american2[std_vars])



minmax_scaling = make_pipeline(

    MinMaxScaler()

)

minmax_vars = ["YOJ"]

american2[minmax_vars] = minmax_scaling.fit_transform(american2[minmax_vars])
american3 = pd.read_csv("/kaggle/input/american-bank-data/Bank_of_America_data.csv")

with_median = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='mean')

)

median_vars = ["LOAN", "MORTDUE", "VALUE", "DEBTINC", "CLAGE"]

american3[median_vars] = with_median.fit_transform(american3[median_vars])

# Converting Discrete columns to type object

discrete_columns = ["REASON", "JOB", "DEROG", "DELINQ", "NINQ", "CLNO", "YOJ"]

american3[discrete_columns] = american3[discrete_columns].astype(object)

most_frequent = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='most_frequent')

)

most_frquent_value_vars = discrete_columns

american3[discrete_columns] = most_frequent.fit_transform(american3[discrete_columns])

std_scaling = make_pipeline(

    StandardScaler()

)

std_vars = ["LOAN", "MORTDUE", "VALUE", "CLAGE", "DEBTINC"]

american3[std_vars] = std_scaling.fit_transform(american3[std_vars])



minmax_scaling = make_pipeline(

    MinMaxScaler()

)

minmax_vars = ["YOJ"]

american3[minmax_vars] = minmax_scaling.fit_transform(american3[minmax_vars])
american4 = pd.read_csv("/kaggle/input/american-bank-data/Bank_of_America_data.csv")

with_median = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='mean')

)

median_vars = ["LOAN", "MORTDUE", "VALUE", "DEBTINC", "CLAGE"]

american4[median_vars] = with_median.fit_transform(american4[median_vars])

# Converting Discrete columns to type object

discrete_columns = ["REASON", "JOB", "DEROG", "DELINQ", "NINQ", "CLNO", "YOJ"]

american4[discrete_columns] = american4[discrete_columns].astype(object)

most_frequent = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='most_frequent')

)

most_frquent_value_vars = discrete_columns

american4[discrete_columns] = most_frequent.fit_transform(american4[discrete_columns])

std_scaling = make_pipeline(

    RobustScaler()

)

std_vars = ["LOAN", "MORTDUE", "VALUE", "CLAGE", "DEBTINC"]

american4[std_vars] = std_scaling.fit_transform(american4[std_vars])



minmax_scaling = make_pipeline(

    MinMaxScaler()

)

minmax_vars = ["YOJ"]

american4[minmax_vars] = minmax_scaling.fit_transform(american4[minmax_vars])
correlation = abs(american.corr())

mask = np.zeros_like(correlation, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
correlation2 = abs(american2.corr())

mask = np.zeros_like(correlation2, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(correlation2, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
correlation3 = abs(american3.corr())

mask = np.zeros_like(correlation3, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(correlation3, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
correlation4 = abs(american4.corr())

mask = np.zeros_like(correlation4, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(correlation4, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
american = pd.read_csv("/kaggle/input/american-bank-data/Bank_of_America_data.csv")

with_median = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='median')

)

median_vars = ["LOAN", "MORTDUE", "VALUE", "CLAGE"]

american[median_vars] = with_median.fit_transform(american[median_vars])

discrete_columns = ["REASON", "JOB", "DEROG", "DELINQ", "NINQ", "CLNO", "YOJ"]

american[discrete_columns] = american[discrete_columns].astype(object)

most_frequent = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='most_frequent')

)

most_frquent_value_vars = discrete_columns

american[discrete_columns] = most_frequent.fit_transform(american[discrete_columns])

std_scaling = make_pipeline(

    StandardScaler()

)

std_vars = ["LOAN", "MORTDUE", "VALUE", "CLAGE"]

american[std_vars] = std_scaling.fit_transform(american[std_vars])



minmax_scaling = make_pipeline(

    MinMaxScaler()

)

minmax_vars = ["YOJ"]

american[minmax_vars] = minmax_scaling.fit_transform(american[minmax_vars])



# DEBTINC

df = american.copy()

#df.drop("BAD", axis=1, inplace=True)

df.dropna(subset=["DEBTINC"], inplace=True)

y_reg = df["DEBTINC"]

df.drop("DEBTINC", axis=1, inplace=True)

X_reg = df.copy()

# Converting Discrete columns to type str

discrete_columns = ["REASON", "JOB", "DEROG", "DELINQ", "NINQ", "CLNO"]

X_reg[discrete_columns] = X_reg[discrete_columns].astype(str)

X_reg = pd.get_dummies(X_reg, drop_first=True)

#PCA

pca = PCA(n_components=102)

ss = StandardScaler()

ss.fit(X_reg)

X_scaled_reg = ss.transform(X_reg)

X_transformed_pca_reg = pca.fit_transform(X_scaled_reg)



def evaluate(train,target_train):

    results={}

    def test_model(clf):

        cv = KFold(n_splits=5,shuffle=True,random_state=45)

        mse = make_scorer(mean_squared_error)

        mse_val_score = cross_val_score(clf, train, target_train, cv=cv,scoring=mse)

        mse_val_score = np.sqrt(mse_val_score)

        scores=[mse_val_score.mean()]

        return scores

    clf = linear_model.LinearRegression()

    results["Linear"]=test_model(clf)

    clf = linear_model.Ridge()

    results["Ridge"]=test_model(clf)

    clf = linear_model.BayesianRidge()

    results["Bayesian Ridge"]=test_model(clf)

    clf = RandomForestRegressor()

    results["RandomForest"]=test_model(clf)

    clf = AdaBoostRegressor()

    results["AdaBoost"]=test_model(clf)

    clf = xgb.XGBRegressor()

    results["XGB"]=test_model(clf)

    clf = LGBMRegressor()

    results["LBGMRegressor"]=test_model(clf)

    results = pd.DataFrame.from_dict(results,orient='index')

    results.columns=["RMSE"] 

    results.sort_values('RMSE', inplace=True)

    #results=results.sort(columns=["R Square Score"],ascending=False)

    results.plot(kind="bar",title="Model Scores")

    axes = plt.gca()

    axes.set_ylim([0.5,20])

    return results



evaluate(X_reg,y_reg)
american5 = pd.read_csv("/kaggle/input/american-bank-data/Bank_of_America_data.csv")

with_median = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='median')

)

median_vars = ["LOAN", "MORTDUE", "VALUE", "CLAGE"]

american5[median_vars] = with_median.fit_transform(american5[median_vars])

discrete_columns = ["DEROG", "DELINQ", "NINQ", "CLNO", "YOJ"]

american5[discrete_columns] = american5[discrete_columns].astype(object)

most_frequent = make_pipeline(

    SimpleImputer(missing_values=np.nan, strategy='most_frequent')

)

most_frquent_value_vars = discrete_columns

american5[discrete_columns] = most_frequent.fit_transform(american5[discrete_columns])

std_scaling = make_pipeline(

    StandardScaler()

)

std_vars = ["LOAN", "MORTDUE", "VALUE", "CLAGE"]

american5[std_vars] = std_scaling.fit_transform(american5[std_vars])



minmax_scaling = make_pipeline(

    MinMaxScaler()

)

minmax_vars = ["YOJ"]

american5[minmax_vars] = minmax_scaling.fit_transform(american5[minmax_vars])

# Encoding for regression model only

  # Job

american5["JOB"].fillna("Unspecified_Job", inplace=True)

  # Reason

american5["REASON"].fillna("Unspecified_Reason", inplace=True)

american5 = pd.get_dummies(american5).copy()
american5.head()
df = american5.copy()

df.drop("BAD", axis=1, inplace=True)

df.dropna(subset=["DEBTINC"], inplace=True)

y_debt = df["DEBTINC"]

df.drop("DEBTINC", axis=1, inplace=True)

X_debt = df.copy()
pd.set_option('display.max_columns', 50)

X_debt.head()
reg = RandomForestRegressor()

reg.fit(X_debt, y_debt);
for _,row in american5.iterrows():

    if np.isnan(row["DEBTINC"]):

        data  = pd.Series(np.array([row["LOAN"],row["MORTDUE"],row["VALUE"],row["YOJ"],row["DEROG"],row["DELINQ"],

                                   row["CLAGE"],row["NINQ"],row["CLNO"],row["REASON_DebtCon"],row["REASON_HomeImp"],row["REASON_Unspecified_Reason"],

                                   row["JOB_Mgr"],row["JOB_Office"],row["JOB_Other"],

                                   row["JOB_ProfExe"],row["JOB_Sales"],row["JOB_Self"], row["JOB_Unspecified_Job"]]))

        american5.loc[_,'DEBTINC'] = reg.predict([data])[0]
american5.isna().sum()
correlation5 = american5.corr()
matrices = [correlation, correlation2, correlation3, correlation4, correlation5]

for i in matrices:

    mat = np.matrix(i)

    print(str(sum(mat[:,1])))
data = american4.copy()
data_corr = abs(data.corr())

mask = np.zeros_like(data_corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(data_corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
data.drop(["YOJ", "VALUE", "LOAN", "MORTDUE", "CLNO"], axis=1, inplace=True)
data_corr = abs(data.corr())

mask = np.zeros_like(data_corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(data_corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True);
data.dtypes
discrete_columns = ["DEROG", "DELINQ", "NINQ", "REASON", "JOB"]

data[discrete_columns] = data[discrete_columns].astype(object)

data.dtypes
data = pd.get_dummies(data, drop_first=True)

data.head()
data.shape
X = data.drop("BAD", axis=1)

y = data["BAD"].copy()

X.head()
pca = PCA(n_components=42)

ss = StandardScaler()

ss.fit(X)

X_scaled = ss.transform(X)

X_transformed_pca = pca.fit_transform(X_scaled)

pca_variance = pca.explained_variance_

explained_variance = pca.explained_variance_ratio_

sum(explained_variance.tolist()[0:42])
plt.figure(figsize=(8, 6))

plt.bar(range(42), pca_variance, alpha=0.5, align='center', label='Individual variance')

plt.legend()

plt.ylabel('Variance ratio')

plt.xlabel('Principal components')

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(X_transformed_pca[:,0], X_transformed_pca[:,41], c=data['BAD'], cmap='viridis')

plt.show()
def evaluate(algo, X_train, Y_train, cv):

    # One Pass

    model = algo.fit(X_train, Y_train)

    acc = round(model.score(X_train, Y_train) * 100, 2)

    

    # Cross Validation 

    train_pred = model_selection.cross_val_predict(algo, 

                                                  X_train, 

                                                  Y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

    # Cross-validation accuracy metric

    acc_cv = round(metrics.accuracy_score(Y_train, train_pred) * 100, 2)

    recall_cv = round(metrics.recall_score(Y_train, train_pred) * 100, 2)

    precision_cv = round(metrics.precision_score(Y_train, train_pred) * 100, 2)

    f1_cv = round(metrics.f1_score(Y_train, train_pred) * 100, 2)

    

    return train_pred, acc, acc_cv, recall_cv, precision_cv, f1_cv



def fit_ml_algo(algo, X_train, y_train, cv):

    # One Pass

    model = algo.fit(X_train, y_train)

    acc = round(model.score(X_train, y_train) * 100, 2)

    

    # Cross Validation 

    train_pred = model_selection.cross_val_predict(algo, 

                                                  X_train, 

                                                  Y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

    # Cross-validation accuracy metric

    acc_cv = round(metrics.accuracy_score(Y_train, train_pred) * 100, 2)

    recall_cv = round(metrics.recall_score(Y_train, train_pred) * 100, 2)

    precision_cv = round(metrics.precision_score(Y_train, train_pred) * 100, 2)

    f1_cv = round(metrics.f1_score(Y_train, train_pred) * 100, 2)

    

    print("Model used :", algo.best_estimator_)

    return train_pred, acc, acc_cv, recall_cv, precision_cv, f1_cv



def fit_ml_algo_ne(algo, X_train, Y_train, cv):

    # One Pass

    model = algo.fit(X_train, Y_train)

    acc = round(model.score(X_train, Y_train) * 100, 2)

    

    # Cross Validation 

    train_pred = model_selection.cross_val_predict(algo, 

                                                  X_train, 

                                                  Y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

    # Cross-validation accuracy metric

    acc_cv = round(metrics.accuracy_score(Y_train, train_pred) * 100, 2)

    recall_cv = round(metrics.recall_score(Y_train, train_pred) * 100, 2)

    precision_cv = round(metrics.precision_score(Y_train, train_pred) * 100, 2)

    f1_cv = round(metrics.f1_score(Y_train, train_pred) * 100, 2)

    

    return train_pred, acc, acc_cv, recall_cv, precision_cv, f1_cv
X_train = X_transformed_pca

y_train = y
# Logistic Regression

start_time = time.time()

train_pred_log, acc_log, acc_cv_log, recall_cv_log, precision_cv_log, f1_cv_log = fit_ml_algo_ne(LogisticRegression(), 

                                                               X_train, 

                                                               y_train, 

                                                                    10)

log_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_log)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)

print("Recall CV 10-Fold: %s" % recall_cv_log)

print("Precision CV 10-Fold: %s" % precision_cv_log)

print("F1 CV 10-Fold: %s" % f1_cv_log)

print("Running Time: %s" % datetime.timedelta(seconds=log_time))
start_time = time.time()

knn_params = {'n_neighbors':list(range(1,15)), 'weights': ['distance', 'uniform']}

knn = KNeighborsClassifier()



grid_search = GridSearchCV(knn, knn_params, cv=5)

grid_search.fit(X_train, y_train)



train_pred_knn, acc_knn, acc_cv_knn, recall_cv_knn, precision_cv_knn, f1_cv_knn = fit_ml_algo_ne(grid_search.best_estimator_, X_train,y_train,10)

knn_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_knn)

print("Accuracy CV 10-Fold: %s" % acc_cv_knn)

print("Recall CV 10-Fold: %s" % recall_cv_knn)

print("Precision CV 10-Fold: %s" % precision_cv_knn)

print("F1 CV 10-Fold: %s" % f1_cv_knn)

print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
start_time = time.time()

train_pred_dt, acc_dt, acc_cv_dt, recall_cv_dt, precision_cv_dt, f1_cv_dt = fit_ml_algo_ne(DecisionTreeClassifier(), 

                                                               X_train, 

                                                               y_train, 

                                                                    10)

dt_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_dt)

print("Accuracy CV 10-Fold: %s" % acc_cv_dt)

print("Recall CV 10-Fold: %s" % recall_cv_dt)

print("Precision CV 10-Fold: %s" % precision_cv_dt)

print("F1 CV 10-Fold: %s" % f1_cv_dt)

print("Running Time: %s" % datetime.timedelta(seconds=dt_time))
rf_params = {'n_estimators': list(range(1,15))

            }



grid_search = GridSearchCV(RandomForestClassifier(), rf_params, cv=10)

grid_search.fit(X_train, y_train)



print("Best params : ",grid_search.best_params_)

start_time = time.time()



train_pred_rf, acc_rf, acc_cv_rf, recall_cv_rf, precision_cv_rf, f1_cv_rf = fit_ml_algo_ne(grid_search.best_estimator_,X_train,y_train,10)



rf_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_rf)

print("Accuracy CV 10-Fold: %s" % acc_cv_rf)

print("Recall CV 10-Fold: %s" % recall_cv_rf)

print("Precision CV 10-Fold: %s" % precision_cv_rf)

print("F1 CV 10-Fold: %s" % f1_cv_rf)

print("Running Time: %s" % datetime.timedelta(seconds=rf_time))
sgdc_params = {"loss": ["hinge", "log"], "penalty": ["l1", "l2"], "max_iter": [1,2,3,4,5]}



start_time = time.time()



grid_search = GridSearchCV(SGDClassifier(), sgdc_params, cv=5)

grid_search.fit(X_train, y_train)



print("Best params : ",grid_search.best_params_)



train_pred_sgdc, acc_sgdc, acc_cv_sgdc, recall_cv_sgdc, precision_cv_sgdc, f1_cv_sgdc  = fit_ml_algo_ne(grid_search.best_estimator_,X_train,y_train,10)



sgdc_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_sgdc)

print("Accuracy CV 10-Fold: %s" % acc_cv_sgdc)

print("Recall CV 10-Fold: %s" % recall_cv_sgdc)

print("Precision CV 10-Fold: %s" % precision_cv_sgdc)

print("F1 CV 10-Fold: %s" % f1_cv_sgdc)

print("Running Time: %s" % datetime.timedelta(seconds=sgdc_time))
train_pred_nb, acc_gnb, acc_cv_gnb, recall_cv_gnb, precision_cv_gnb, f1_cv_gnb  = fit_ml_algo_ne(GaussianNB(),X_train,y_train,10)

gnb_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_gnb)

print("Accuracy CV 10-Fold: %s" % acc_cv_gnb)

print("Recall CV 10-Fold: %s" % recall_cv_gnb)

print("Precision CV 10-Fold: %s" % precision_cv_gnb)

print("F1 CV 10-Fold: %s" % f1_cv_gnb)

print("Running Time: %s" % datetime.timedelta(seconds=gnb_time))
"""gbc_params = {"loss": ["deviance", "exponential"],

              "learning_rate": [1,0.6 ,0.5,0.4,0.3, 0.25, 0.1, 0.05, 0.01],

              "n_estimators": [10,50,100]

            }



grid_search = GridSearchCV(GradientBoostingClassifier(), gbc_params, cv=5)

grid_search.fit(X_train, y_train)

print("Best params : ",grid_search.best_params_)

train_pred_gbc, acc_gbc, acc_cv_gbc, recall_cv_gbc, precision_cv_gbc, f1_cv_gbc = fit_ml_algo_ne(grid_search.best_estimator_,X_train,y_train,10)

gbc_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_gbc)

print("Accuracy CV 10-Fold: %s" % acc_cv_gbc)

print("Recall CV 10-Fold: %s" % recall_cv_gbc)

print("Precision CV 10-Fold: %s" % precision_cv_gbc)

print("F1 CV 10-Fold: %s" % f1_cv_gbc)

print("Running Time: %s" % datetime.timedelta(seconds=gbc_time))"""
# Linear SVC

lsvc_params = {"penalty": ["l2"],

                "loss": ["hinge", "squared_hinge"],

               "dual": [True],

               "C": [0.001,0.01,0.1,1,10]

            }



grid_search = GridSearchCV(LinearSVC(), lsvc_params, cv=5)

grid_search.fit(X_train, y_train)

print("Best params : ",grid_search.best_params_)



train_pred_lsvc, acc_lsvc, acc_cv_lsvc, recall_cv_lsvc, precision_cv_lsvc, f1_cv_lsvc = fit_ml_algo_ne(grid_search.best_estimator_,X_train,y_train,10)

lsvc_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_lsvc)

print("Accuracy CV 10-Fold: %s" % acc_cv_lsvc)

print("Recall CV 10-Fold: %s" % recall_cv_lsvc)

print("Precision CV 10-Fold: %s" % precision_cv_lsvc)

print("F1 CV 10-Fold: %s" % f1_cv_lsvc)

print("Running Time: %s" % datetime.timedelta(seconds=lsvc_time))
# XGBoost Classifier

start_time = time.time()

xgb_params = {"early_stopping_rounds": [1,2,5],

                "n_estimators": [5,10,15],

               "learning_rate": [0.001,0.03,0.05],

               "n_jobs": [0,1,2,5]

            }



grid_search = GridSearchCV(XGBClassifier(), xgb_params, cv=5)

grid_search.fit(X_train, y_train)

print("Best params : ",grid_search.best_params_)



train_pred_xgb, acc_xgb, acc_cv_xgb, recall_cv_xgb, precision_cv_xgb, f1_cv_xgb = fit_ml_algo_ne(grid_search.best_estimator_,X_train, y_train,10)

xgb_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_xgb)

print("Accuracy CV 10-Fold: %s" % acc_cv_xgb)

print("Recall CV 10-Fold: %s" % recall_cv_xgb)

print("Precision CV 10-Fold: %s" % precision_cv_xgb)

print("F1 CV 10-Fold: %s" % f1_cv_xgb)

print("Running Time: %s" % datetime.timedelta(seconds=xgb_time))
from catboost import CatBoostClassifier, Pool, cv



train_pool = Pool(X_train, 

                  y_train)

catboost_model = CatBoostClassifier(iterations=2500,

                                    custom_loss=['F1', 'Precision', 'Recall', 'Accuracy'],

                                    loss_function='Logloss',

                                    #task_type="GPU",

                                    learning_rate=0.1,

                                    devices='0:1')

                                

catboost_model.fit(train_pool,plot=True)

catboost_model.best_score_



# Print out the CatBoost model metrics

print("---CatBoost Metrics---")

print(str(format(catboost_model.best_score_)))
acc_catboost = catboost_model.best_score_["learn"]["Accuracy"] * 100

recall_catboost = catboost_model.best_score_["learn"]["Recall"] * 100

precision_catboost = catboost_model.best_score_["learn"]["Precision"] * 100

f1_catboost = catboost_model.best_score_["learn"]["F1"] * 100
my_score_cv = cross_val_score(catboost_model, X_train, y_train, 

                         cv = 5, 

                         scoring = 'recall')
my_score_cv
# Cross Validation 

train_pred = model_selection.cross_val_predict(catboost_model, 

                                                  X_train, 

                                                  y_train, 

                                                  cv=5, 

                                                  n_jobs = -1)
    # Cross-validation accuracy metric

acc_cv_catboost = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

recall_cv_catboost = round(metrics.recall_score(y_train, train_pred) * 100, 2)

precision_cv_catboost = round(metrics.precision_score(y_train, train_pred) * 100, 2)

f1_cv_catboost = round(metrics.f1_score(y_train, train_pred) * 100, 2)
print("Accuracy CV 10-Fold: %s" % acc_cv_catboost)

print("Recall CV 10-Fold: %s" % recall_cv_catboost)

print("Precision CV 10-Fold: %s" % precision_cv_catboost)

print("F1 CV 10-Fold: %s" % f1_cv_catboost)
# Light GBM  Classifier

start_time = time.time()

lgbm_params = {

    'n_estimators': [100, 200, 300],

    """'colsample_bytree': [0.7, 0.8],

    'max_depth': [15,20,25],

    'num_leaves': [50, 100, 200],

    'reg_alpha': [1.1, 1.2, 1.3],

    'reg_lambda': [1.1, 1.2, 1.3],

    'min_split_gain': [0.3, 0.4],"""

    'subsample': [0.7, 0.8, 0.9],

    'subsample_freq': [20]

}



grid_search = GridSearchCV(LGBMClassifier(), lgbm_params, cv=5)

grid_search.fit(X_train, y_train)

print("Best params : ",grid_search.best_params_)



train_pred_lgbm, acc_lgbm, acc_cv_lgbm, recall_cv_lgbm, precision_cv_lgbm, f1_cv_lgbm  = fit_ml_algo_ne(grid_search.best_estimator_,X_train, y_train,10)

lgbm_time = (time.time() - start_time)

print("Accuracy: (1 Fold) %s" % acc_lgbm)

print("Accuracy CV 10-Fold: %s" % acc_cv_lgbm)

print("Recall CV 10-Fold: %s" % recall_cv_lgbm)

print("Precision CV 10-Fold: %s" % precision_cv_lgbm)

print("F1 CV 10-Fold: %s" % f1_cv_lgbm)

print("Running Time: %s" % datetime.timedelta(seconds=lgbm_time))
cv_models = pd.DataFrame({

    'Model': ['KNN', 

              'Logistic Regression',

              'Gaussian Naive Bayes',

              'Stochastic Gradient Decent',

              'Linear SVC', 

              'Decision Tree',

              #'Gradient Boosting Trees',

              'XGBoost',

              'Random Forest',

              "LGBM",

              "Catboost",

              "Catboost_1F"

              ],

    'Accuracy': [

        acc_cv_knn,

        acc_cv_log,

        acc_cv_gnb,

        acc_cv_sgdc,

        acc_cv_lsvc,

        acc_cv_dt,

        #acc_cv_gbc,

        acc_cv_xgb,

        acc_cv_rf,

        acc_cv_lgbm,

        acc_cv_catboost,

        acc_catboost

    ], 'Precision': [

        precision_cv_knn,

        precision_cv_log,

        precision_cv_gnb,

        precision_cv_sgdc,

        precision_cv_lsvc,

        precision_cv_dt,

        #precision_cv_gbc,

        precision_cv_xgb,

        precision_cv_rf,

        precision_cv_lgbm,

        precision_cv_catboost,

        precision_catboost

    ], 'Recall': [

        recall_cv_knn,

        recall_cv_log,

        recall_cv_gnb,

        recall_cv_sgdc,

        recall_cv_lsvc,

        recall_cv_dt,

        #recall_cv_gbc,

        recall_cv_xgb,

        recall_cv_rf,

        recall_cv_lgbm,

        recall_cv_catboost,

        recall_catboost

    ], 'F1-Score' : [

        f1_cv_knn,

        f1_cv_log,

        f1_cv_gnb,

        f1_cv_sgdc,

        f1_cv_lsvc,

        f1_cv_dt,

        #f1_cv_gbc,

        f1_cv_xgb,

        f1_cv_rf,

        f1_cv_lgbm,

        f1_cv_catboost,

        f1_catboost

    ]})
print('---Cross-validation Scores---')

cv_models.sort_values(by='Recall', ascending=False)