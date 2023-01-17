import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

df.head()
df.describe()
assert not df.isna().any().any(), "Missing values found"
df["DEATH_EVENT"].value_counts()
fig = px.scatter_matrix(df, color="DEATH_EVENT")

fig.update_traces(diagonal_visible=False)

fig.update(layout_showlegend=False, layout_coloraxis_showscale=False)

fig.show()
cols_to_plot = ["DEATH_EVENT", "serum_sodium", "serum_creatinine", "platelets", "creatinine_phosphokinase", "age", "ejection_fraction", "time"]

df_to_plot = df[cols_to_plot].melt(id_vars=["DEATH_EVENT"])

fig = px.box(df_to_plot, color="DEATH_EVENT", y="value", facet_col="variable")    

fig.update_yaxes(matches=None)

fig.show()
df["age_th"] = df["age"].apply(lambda x: 0 if x < 65 else 1)

df["serum_sodium_th"] = df["serum_sodium"].apply(lambda x: 0 if x < 135 else 1)

df["serum_creatin_th"] = df["serum_creatinine"].apply(lambda x: 0 if x < 1.2 else 1)

df["time_th"] = df["time"].apply(lambda x: 0 if x < 95 else 1)

df["ejection_fraction_th"] = df["ejection_fraction"].apply(lambda x: 0 if x < 38 else 1)
df.describe()
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.metrics import f1_score

from sklearn.decomposition import PCA



from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

import xgboost
%%time

logistic_param_grid = {"logisticregression__solver": ["liblinear"], "logisticregression__C": np.arange(0.2, 1.6, 0.1)}

logreg_pipeline = make_pipeline(StandardScaler(), LogisticRegression())

clf = GridSearchCV(logreg_pipeline, n_jobs=-1, cv=StratifiedShuffleSplit(5, random_state=10), param_grid=logistic_param_grid)

clf.fit(df.drop("DEATH_EVENT", axis=1), df["DEATH_EVENT"])

print("F1 Score: ", clf.best_score_)

print("Best Params", clf.best_params_)

best_clf = clf.best_estimator_  # This is the best estimator
weights = pd.Series(dict(zip(df.drop("DEATH_EVENT", axis=1).columns, best_clf[1].coef_[0])))

weights.sort_values().plot.bar(figsize=(20, 5))

plt.xlabel("Feature")

plt.ylabel("Weight")

plt.title("Feature Importance")

plt.show()
%%time



# I used a bigger grid initially, but I reduced it now for the sake of speeding the process. Check previous commits

xgboost_param_grid = {

    'booster': ['gbtree'], 

    'colsample_bytree': [0.9], 

    'eta': [0.5], 

    'eval_metric': ['auc'], 

    'gamma': [0.0], 

    'lambda': [0.8], 

    'min_child_weight': [9], 

    'n_estimators': [600], 

    'subsample': [0.9]}



clf = GridSearchCV(xgboost.XGBClassifier(), n_jobs=-1, cv=StratifiedShuffleSplit(5, random_state=10), param_grid=xgboost_param_grid)

clf.fit(df.drop("DEATH_EVENT", axis=1), df["DEATH_EVENT"])

print("F1 Score: ", clf.best_score_)

print("Best Params", clf.best_params_)

best_clf = clf.best_estimator_  # This is the best estimator
xgboost.plot_importance(best_clf)

plt.show()