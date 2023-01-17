import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%pylab inline
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import GridSearchCV
raw = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

raw.tail(10)
main = "#68C643"

secondary = "#A144C5"
sns.heatmap(raw[["anaemia", "diabetes", "smoking", "high_blood_pressure", "sex"]].corr(), annot=True)
sns.countplot(raw["sex"], color=main)
conditions_df = raw[["anaemia", "diabetes", "smoking", "high_blood_pressure", "sex", "age"]]

for i in ["anaemia", "diabetes", "smoking", "high_blood_pressure"]:

    plot = sns.FacetGrid(conditions_df, col="sex")

    plot.map(sns.countplot, i, color=main)
sns.heatmap(raw[["anaemia", "diabetes", "smoking", "high_blood_pressure", "age"]].corr(), annot=True)
raw[["age", "sex"]].groupby("sex").agg(["min", "max", "mean", "median"])
ageplot = sns.FacetGrid(raw[["age", "sex"]], col="sex")

ageplot.map(sns.kdeplot, "age", shade=True, color=main)
for i in ["anaemia", "diabetes", "smoking", "high_blood_pressure"]:

    plot = sns.FacetGrid(conditions_df, col="sex")

    plot.map(sns.swarmplot, i, "age", color=secondary)

    plot.map(sns.boxplot, i, "age", color=main)
raw[(raw["sex"] == 0) & (raw["smoking"] == 1)]
raw[raw["sex"] == 0].iloc[:4]
levels_df = raw[["creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "DEATH_EVENT"]]

sns.heatmap(levels_df.corr(), annot=True)
sns.pairplot(data=levels_df, hue="DEATH_EVENT", corner=True, kind="reg", diag_kind="hist", palette=sns.color_palette([main, secondary]))
for i in ["creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]:

    death_plot = sns.FacetGrid(levels_df, col="DEATH_EVENT")

    death_plot.map(sns.swarmplot, i, color=main)
raw["DEATH_EVENT"].value_counts()
ds_df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

y = ds_df.pop("DEATH_EVENT")

normalize_cols = ["ejection_fraction","serum_creatinine", "serum_sodium", "time", "creatinine_phosphokinase", "platelets"]

ds_df_n = ds_df

ds_df_n[normalize_cols] = ((ds_df[normalize_cols] - ds_df[normalize_cols].min()) / (ds_df[normalize_cols].max() - ds_df[normalize_cols].min())) * 20

X_train, X_test, y_train, y_test = train_test_split(ds_df_n, y, test_size=0.33)

class_weights = compute_class_weight("balanced", np.unique(y_train), y_train)

print(class_weights)

ds_df.head()
ds_df_n.head()
# I have used a GridSearchCV, and personal judgement to come up with the following

# hyperparameters for the RandomForest model

rf = RandomForestClassifier(bootstrap=False, criterion="entropy", max_depth=3, n_estimators=120)

rf.fit(X_train, y_train)

print(rf.score(X_test, y_test))

plot_confusion_matrix(rf, X_test, y_test)
plot_confusion_matrix(rf, ds_df_n, y)

print(rf.score(ds_df_n, y))
from joblib import dump, load

dump(rf, "heart_failure_predictor.joblib")
def generate_predictions(filepath, normalize_cols, modelpath):

    from joblib import load

    import pandas as pd

    def preprocess_data(dataframe):

        dataframe[normalize_cols] = ((dataframe[normalize_cols] - dataframe[normalize_cols].min()) / (dataframe[normalize_cols].max() - dataframe[normalize_cols].min())) * 20

        return dataframe

    model = load(modelpath)

    # VERY IMPORTANT!!!

    #

    # REMOVE .drop("DEATH_EVENT", axis=1) in when using on a real, unpredicted dataset

    # I'm just putting it there to conveniently demonstrate the usage on the same dataset

    data = preprocess_data(pd.read_csv(filepath)).drop("DEATH_EVENT", axis=1) # REMOVE this on real dataset

    data.insert(0, "predicted", model.predict(data))

    data.to_csv("./results.csv", index=False)

    return data
result = generate_predictions("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv",

                    ["ejection_fraction","serum_creatinine", "serum_sodium", "time", "creatinine_phosphokinase", "platelets"],

                    "./heart_failure_predictor.joblib")

result.sample(10)