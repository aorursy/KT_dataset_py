import os

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.graph_objects as go

from mlxtend.plotting import plot_confusion_matrix



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler

import tensorflow as tf



import optuna

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostClassifier



def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

seed_everything(0)



sns.set_style("whitegrid")

palette_ro = ["#ee2f35", "#fa7211", "#fbd600", "#75c731", "#1fb86e", "#0488cf", "#7b44ab"]



ROOT = "../input/heart-failure-clinical-data"
df = pd.read_csv(ROOT + "/heart_failure_clinical_records_dataset.csv")



print("Data shape: ", df.shape)

df.head()
df.isnull().sum()
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 12))



sns.countplot(x="anaemia", ax=ax1, data=df,

              palette=palette_ro[3::-3], alpha=0.9)

sns.countplot(x="diabetes", ax=ax2, data=df,

              palette=palette_ro[3::-3], alpha=0.9)

sns.countplot(x="high_blood_pressure", ax=ax3, data=df,

              palette=palette_ro[3::-3], alpha=0.9)

sns.countplot(x="sex", ax=ax4, data=df,

              palette=palette_ro[2::3], alpha=0.9)

sns.countplot(x="smoking", ax=ax5, data=df,

              palette=palette_ro[3::-3], alpha=0.9)

sns.countplot(x="DEATH_EVENT", ax=ax6, data=df,

              palette=palette_ro[1::5], alpha=0.9)

fig.suptitle("Distribution of the binary features and DEATH_EVENT", fontsize=18);
bin_features = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]



df_d = pd.DataFrame(columns=[0, 1, "value"])

for col in bin_features:

    for u in df[col].unique():

        df_d.loc[col+"_"+str(u)] = 0

        for i in df["DEATH_EVENT"].unique():

            if u == 0:

                df_d["value"][col+"_"+str(u)] = "0 (False)"

            else:

                df_d["value"][col+"_"+str(u)] = "1 (True)"

            df_d[i][col+"_"+str(u)] = df[df[col]==u]["DEATH_EVENT"].value_counts(normalize=True)[i] * 100



df_d = df_d.reindex(index=["anaemia_0", "anaemia_1", "diabetes_0", "diabetes_1", "high_blood_pressure_0", "high_blood_pressure_1",

                           "sex_0", "sex_1", "smoking_0", "smoking_1"])

df_d.at["sex_0", "value"] = "0 (Female)"

df_d.at["sex_1", "value"] = "1 (Male)"



fig = go.Figure(data=[

    go.Bar(y=[["anaemia", "anaemia","diabetes","diabetes","high_blood_pressure","high_blood_pressure","sex","sex","smoking","smoking"], list(df_d["value"])],

           x=df_d[0], name="DEATH_EVENT = 0<br>(survived)", orientation='h', marker=dict(color=palette_ro[1])),

    go.Bar(y=[["anaemia", "anaemia","diabetes","diabetes","high_blood_pressure","high_blood_pressure","sex","sex","smoking","smoking"], list(df_d["value"])],

           x=df_d[1], name="DEATH_EVENT = 1<br>(dead)", orientation='h', marker=dict(color=palette_ro[6]))

])

fig.update_layout(barmode="stack",

                  title="Percentage of DEATH_EVENT per binary features")

fig.update_yaxes(autorange="reversed")

fig.show(config={"displayModeBar": False})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



range_bin_width = np.arange(df["age"].min(), df["age"].max()+5, 5)



sns.distplot(df["age"], ax=ax1, bins=range_bin_width, color=palette_ro[5])

sns.distplot(df[df["DEATH_EVENT"]==0].age, label="DEATH_EVENT=0", ax=ax2, bins=range_bin_width, color=palette_ro[1])

sns.distplot(df[df["DEATH_EVENT"]==1].age, label="DEATH_EVENT=1", ax=ax2, bins=range_bin_width, color=palette_ro[6])

ax1.set_title("age distribution", fontsize=16);

ax2.set_title("Relationship between age and DEATH_EVENT", fontsize=16)



ax1.axvline(x=df["age"].median(), color=palette_ro[5], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==0].age.median(), color=palette_ro[1], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==1].age.median(), color=palette_ro[6], linestyle="--", alpha=0.5)



ax1.annotate("Min: {:.0f}".format(df["age"].min()), xy=(df["age"].min(), 0.010), 

             xytext=(df["age"].min()-7, 0.015),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:.0f}".format(df["age"].max()), xy=(df["age"].max(), 0.005), 

             xytext=(df["age"].max(), 0.008),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.2"))

ax1.annotate("Med: {:.0f}".format(df["age"].median()), xy=(df["age"].median(), 0.032), 

             xytext=(df["age"].median()-8, 0.035),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))



ax2.annotate("Survived\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==0].age.median()), xy=(df[df["DEATH_EVENT"]==0].age.median(), 0.033), 

             xytext=(df[df["DEATH_EVENT"]==0].age.median()-18, 0.035),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))

ax2.annotate("Dead\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==1].age.median()), xy=(df[df["DEATH_EVENT"]==1].age.median(), 0.026), 

             xytext=(df[df["DEATH_EVENT"]==1].age.median()+7, 0.029),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))

ax2.legend();
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



sns.distplot(df["creatinine_phosphokinase"], ax=ax1, color=palette_ro[5])

sns.distplot(df[df["DEATH_EVENT"]==0].creatinine_phosphokinase, label="DEATH_EVENT=0", ax=ax2, hist=None, color=palette_ro[1])

sns.distplot(df[df["DEATH_EVENT"]==1].creatinine_phosphokinase, label="DEATH_EVENT=1", ax=ax2, hist=None, color=palette_ro[6])

ax1.set_title("creatinine_phosphokinase distribution", fontsize=16);

ax2.set_title("Relationship between creatinine_phosphokinase and DEATH_EVENT", fontsize=16)



ax1.axvline(x=df["creatinine_phosphokinase"].median(), color=palette_ro[5], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==0].creatinine_phosphokinase.median(), color=palette_ro[1], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==1].creatinine_phosphokinase.median(), color=palette_ro[6], linestyle="--", alpha=0.5)



ax1.annotate("Min: {:,}".format(df["creatinine_phosphokinase"].min()), xy=(df["creatinine_phosphokinase"].min(), 0.00085), 

             xytext=(df["creatinine_phosphokinase"].min()-700, 0.0010),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:,}".format(df["creatinine_phosphokinase"].max()), xy=(df["creatinine_phosphokinase"].max(), 0.00005), 

             xytext=(df["creatinine_phosphokinase"].max()-500, 0.0002),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Med: {:.0f}".format(df["creatinine_phosphokinase"].median()), xy=(df["creatinine_phosphokinase"].median(), 0.0014), 

             xytext=(df["creatinine_phosphokinase"].median()+500, 0.0015),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))



ax2.annotate("Survived\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==0].creatinine_phosphokinase.median()), xy=(df[df["DEATH_EVENT"]==0].creatinine_phosphokinase.median(), 0.00145), 

             xytext=(df[df["DEATH_EVENT"]==0].creatinine_phosphokinase.median()+600, 0.00145),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))

ax2.annotate("Dead\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==1].creatinine_phosphokinase.median()), xy=(df[df["DEATH_EVENT"]==1].creatinine_phosphokinase.median(), 0.00135), 

             xytext=(df[df["DEATH_EVENT"]==1].creatinine_phosphokinase.median()+700, 0.00125),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))

ax2.legend();
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



range_bin_width = np.arange(df["ejection_fraction"].min(), df["ejection_fraction"].max()+1, 1)



sns.distplot(df["ejection_fraction"], ax=ax1, bins=range_bin_width, color=palette_ro[5])

sns.distplot(df[df["DEATH_EVENT"]==0].ejection_fraction, label="DEATH_EVENT=0", ax=ax2, bins=range_bin_width, color=palette_ro[1])

sns.distplot(df[df["DEATH_EVENT"]==1].ejection_fraction, label="DEATH_EVENT=1", ax=ax2, bins=range_bin_width, color=palette_ro[6])

ax1.set_title("ejection_fraction distribution", fontsize=16);

ax2.set_title("Relationship between ejection_fraction and DEATH_EVENT", fontsize=16)



ax1.axvline(x=df["ejection_fraction"].median(), color=palette_ro[5], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==0].ejection_fraction.median(), color=palette_ro[1], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==1].ejection_fraction.median(), color=palette_ro[6], linestyle="--", alpha=0.5)



ax1.annotate("Min: {:,}".format(df["ejection_fraction"].min()), xy=(df["ejection_fraction"].min(), 0.005), 

             xytext=(df["ejection_fraction"].min()-5, 0.022),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:,}".format(df["ejection_fraction"].max()), xy=(df["ejection_fraction"].max(), 0.001), 

             xytext=(df["ejection_fraction"].max(), 0.022),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.2"))

ax1.annotate("Med: {:.0f}".format(df["ejection_fraction"].median()), xy=(df["ejection_fraction"].median(), 0.041), 

             xytext=(df["ejection_fraction"].median()+5, 0.074),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))



ax2.annotate("Survived\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==0].ejection_fraction.median()), xy=(df[df["DEATH_EVENT"]==0].ejection_fraction.median(), 0.051), 

             xytext=(df[df["DEATH_EVENT"]==0].ejection_fraction.median()+5, 0.091),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))

ax2.annotate("Dead\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==1].ejection_fraction.median()), xy=(df[df["DEATH_EVENT"]==1].ejection_fraction.median(), 0.03), 

             xytext=(df[df["DEATH_EVENT"]==1].ejection_fraction.median()-18, 0.04),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))

ax2.legend();
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



sns.distplot(df["platelets"], ax=ax1, color=palette_ro[5])

sns.distplot(df[df["DEATH_EVENT"]==0].platelets, label="DEATH_EVENT=0", ax=ax2, hist=None, color=palette_ro[1])

sns.distplot(df[df["DEATH_EVENT"]==1].platelets, label="DEATH_EVENT=1", ax=ax2, hist=None, color=palette_ro[6])

ax1.set_title("platelets distribution", fontsize=16);

ax2.set_title("Relationship between platelets and DEATH_EVENT", fontsize=16)



ax1.axvline(x=df["platelets"].median(), color=palette_ro[5], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==0].platelets.median(), color=palette_ro[1], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==1].platelets.median(), color=palette_ro[6], linestyle="--", alpha=0.5)



ax1.annotate("Min: {:,.0f}".format(df["platelets"].min()), xy=(df["platelets"].min(), 2e-7), 

             xytext=(df["platelets"].min()-50000, 7e-7),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:,.0f}".format(df["platelets"].max()), xy=(df["platelets"].max(), 1e-7), 

             xytext=(df["platelets"].max()-30000, 7e-7),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.2"))

ax1.annotate("Med: {:,.0f}".format(df["platelets"].median()), xy=(df["platelets"].median(), 5.9e-6), 

             xytext=(df["platelets"].median()+25000, 5.5e-6),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))



ax2.annotate("Survived\nMed: {:,.0f}".format(df[df["DEATH_EVENT"]==0].platelets.median()), xy=(df[df["DEATH_EVENT"]==0].platelets.median(), 6.2e-6), 

             xytext=(df[df["DEATH_EVENT"]==0].platelets.median()+50000, 5.5e-6),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))

ax2.annotate("Dead\nMed: {:,.0f}".format(df[df["DEATH_EVENT"]==1].platelets.median()), xy=(df[df["DEATH_EVENT"]==1].platelets.median(), 4.5e-6), 

             xytext=(df[df["DEATH_EVENT"]==1].platelets.median()-200000, 5.2e-6),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))

ax2.legend();
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



range_bin_width = np.arange(df["serum_creatinine"].min(), df["serum_creatinine"].max()+0.25, 0.25)



sns.distplot(df["serum_creatinine"], ax=ax1, bins=range_bin_width, color=palette_ro[5])

sns.distplot(df[df["DEATH_EVENT"]==0].serum_creatinine, label="DEATH_EVENT=0", ax=ax2, bins=range_bin_width, color=palette_ro[1])

sns.distplot(df[df["DEATH_EVENT"]==1].serum_creatinine, label="DEATH_EVENT=1", ax=ax2, bins=range_bin_width, color=palette_ro[6])

ax1.set_title("serum_creatinine distribution", fontsize=16);

ax2.set_title("Relationship serum_creatinine age and DEATH_EVENT", fontsize=16)



ax1.axvline(x=df["serum_creatinine"].median(), color=palette_ro[5], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==0].serum_creatinine.median(), color=palette_ro[1], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==1].serum_creatinine.median(), color=palette_ro[6], linestyle="--", alpha=0.5)



ax1.annotate("Min: {:.1f}".format(df["serum_creatinine"].min()), xy=(df["serum_creatinine"].min(), 0.31), 

             xytext=(df["serum_creatinine"].min()-0.7, 0.5),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:.1f}".format(df["serum_creatinine"].max()), xy=(df["serum_creatinine"].max(), 0.05), 

             xytext=(df["serum_creatinine"].max()-0.2, 0.25),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.2"))

ax1.annotate("Med: {:.1f}".format(df["serum_creatinine"].median()), xy=(df["serum_creatinine"].median(), 1.22), 

             xytext=(df["serum_creatinine"].median()+0.5, 1.3),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))



ax2.annotate("Survived\nMed: {:.1f}".format(df[df["DEATH_EVENT"]==0].serum_creatinine.median()), xy=(df[df["DEATH_EVENT"]==0].serum_creatinine.median(), 1.47), 

             xytext=(df[df["DEATH_EVENT"]==0].serum_creatinine.median()-1.3, 1.5),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))

ax2.annotate("Dead\nMed: {:.1f}".format(df[df["DEATH_EVENT"]==1].serum_creatinine.median()), xy=(df[df["DEATH_EVENT"]==1].serum_creatinine.median(), 0.62), 

             xytext=(df[df["DEATH_EVENT"]==1].serum_creatinine.median()+0.4, 0.7),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))

ax2.legend();
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



range_bin_width = np.arange(df["serum_sodium"].min(), df["serum_sodium"].max()+1, 1)



sns.distplot(df["serum_sodium"], ax=ax1, bins=range_bin_width, color=palette_ro[5])

sns.distplot(df[df["DEATH_EVENT"]==0].serum_sodium, label="DEATH_EVENT=0", ax=ax2, bins=range_bin_width, color=palette_ro[1])

sns.distplot(df[df["DEATH_EVENT"]==1].serum_sodium, label="DEATH_EVENT=1", ax=ax2, bins=range_bin_width, color=palette_ro[6])

ax1.set_title("serum_sodium distribution", fontsize=16);

ax2.set_title("Relationship between serum_sodium and DEATH_EVENT", fontsize=16)



ax1.axvline(x=df["serum_sodium"].median(), color=palette_ro[5], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==0].serum_sodium.median(), color=palette_ro[1], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==1].serum_sodium.median(), color=palette_ro[6], linestyle="--", alpha=0.5)



ax1.annotate("Min: {:.0f}".format(df["serum_sodium"].min()), xy=(df["serum_sodium"].min(), 0.005), 

             xytext=(df["serum_sodium"].min()-3, 0.015),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:.0f}".format(df["serum_sodium"].max()), xy=(df["serum_sodium"].max(), 0.005), 

             xytext=(df["serum_sodium"].max(), 0.015),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.2"))

ax1.annotate("Med: {:.0f}".format(df["serum_sodium"].median()), xy=(df["serum_sodium"].median(), 0.103), 

             xytext=(df["serum_sodium"].median()-6, 0.115),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))



ax2.annotate("Survived\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==0].serum_sodium.median()), xy=(df[df["DEATH_EVENT"]==0].serum_sodium.median(), 0.117), 

             xytext=(df[df["DEATH_EVENT"]==0].serum_sodium.median()+5, 0.135),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))

ax2.annotate("Dead\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==1].serum_sodium.median()), xy=(df[df["DEATH_EVENT"]==1].serum_sodium.median(), 0.09), 

             xytext=(df[df["DEATH_EVENT"]==1].serum_sodium.median()-5.5, 0.11),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))

ax2.legend();
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))



range_bin_width = np.arange(df["time"].min(), df["time"].max()+10, 10)



sns.distplot(df["time"], ax=ax1, bins=range_bin_width, color=palette_ro[5])

sns.distplot(df[df["DEATH_EVENT"]==0].time, label="DEATH_EVENT=0", ax=ax2, bins=range_bin_width, color=palette_ro[1])

sns.distplot(df[df["DEATH_EVENT"]==1].time, label="DEATH_EVENT=1", ax=ax2, bins=range_bin_width, color=palette_ro[6])

ax1.set_title("time distribution", fontsize=16);

ax2.set_title("Relationship between time and DEATH_EVENT", fontsize=16)



ax1.axvline(x=df["time"].median(), color=palette_ro[5], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==0].time.median(), color=palette_ro[1], linestyle="--", alpha=0.5)

ax2.axvline(x=df[df["DEATH_EVENT"]==1].time.median(), color=palette_ro[6], linestyle="--", alpha=0.5)



ax1.annotate("Min: {:.0f}".format(df["time"].min()), xy=(df["time"].min(), 0.0021), 

             xytext=(df["time"].min()-30, 0.0032),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Max: {:.0f}".format(df["time"].max()), xy=(df["time"].max(), 0.001), 

             xytext=(df["time"].max(), 0.0017),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.2"))

ax1.annotate("Med: {:.0f}".format(df["time"].median()), xy=(df["time"].median(), 0.0041), 

             xytext=(df["time"].median()+8, 0.0052),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))



ax2.annotate("Survived\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==0].time.median()), xy=(df[df["DEATH_EVENT"]==0].time.median(), 0.0035), 

             xytext=(df[df["DEATH_EVENT"]==0].time.median()-40, 0.007),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=0.25"))

ax2.annotate("Dead\nMed: {:.0f}".format(df[df["DEATH_EVENT"]==1].time.median()), xy=(df[df["DEATH_EVENT"]==1].time.median(), 0.0082), 

             xytext=(df[df["DEATH_EVENT"]==1].time.median()+7, 0.0105),

             bbox=dict(boxstyle="round", fc="none", ec="gray"),

             arrowprops=dict(arrowstyle="->", facecolor='slategray', edgecolor='slategray',

                             connectionstyle="arc3, rad=-0.25"))

ax2.legend();
fig, ax = plt.subplots(1, 1, figsize=(12, 8))



sns.scatterplot(x=df["serum_creatinine"], y=df["ejection_fraction"], ax=ax,

                palette=[palette_ro[1], palette_ro[6]], hue=df["DEATH_EVENT"])

ax.plot([0.9, 5.3], [13, 80.0], color="gray", ls="--")



fig.suptitle("Relationship between serum_creatinine and ejection_fraction against DEATH_EVENT", fontsize=18);
fig, ax = plt.subplots(1, 1, figsize=(12, 8))



sns.heatmap(df.corr(), ax=ax, vmax=1, vmin=-1, center=0,

            annot=True, fmt=".2f",

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            mask=np.triu(np.ones_like(df.corr(), dtype=np.bool)))



fig.suptitle("Diagonal correlation matrix", fontsize=18);
num_features = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]

num_features_s = []



for i in range(len(num_features)):

    num_features_s.append(num_features[i] + "_s")



sc = StandardScaler()

df[num_features_s] = sc.fit_transform(df[num_features])

df.head()
X = df.copy()

y = X["DEATH_EVENT"]

X = X.drop(["DEATH_EVENT"], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)

X_train.head()
features = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets",

            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]

NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = lgb.LGBMClassifier(objective="binary",

                             metric="binary_logloss")

    clf.fit(X_tr, y_tr, eval_set = [(X_va, y_va)],

            early_stopping_rounds=10,

            verbose=-1)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features], num_iteration=clf.best_iteration_)

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"\nOut-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of default LightGBM", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default LightGBM", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1-score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["ejection_fraction", "serum_creatinine", "time"]



def objective(trial):

    skf = StratifiedKFold(n_splits=NFOLD)

    oof = np.zeros((len(X_train), ))



    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

        print(f"FOLD {fold_id+1}")

        X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]



        clf = lgb.LGBMClassifier(objective="binary",

                                 metric="binary_logloss",

                                 colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.4, 1.0),

                                 learning_rate = trial.suggest_uniform("learning_rate", 1e-8, 1.0),

                                 max_depth = trial.suggest_int("max_depth", 2, 32),

                                 min_child_samples = trial.suggest_int("min_child_samples", 3, 500),

                                 min_child_weight = trial.suggest_loguniform("min_child_weight", 1e-4, 1e+1),

                                 n_estimators = trial.suggest_int("n_estimators", 20, 200),

                                 num_leaves = trial.suggest_int("num_leaves", 2, 512),

                                 reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),

                                 reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),

                                 subsample = trial.suggest_uniform("subsample", 0.4, 1.0),

                                 subsample_freq = trial.suggest_int("subsample_freq", 0, 7),

                                )

        clf.fit(X_tr, y_tr, eval_set = [(X_va, y_va)],

                early_stopping_rounds=20,

                verbose=-1)

        oof[va_idx] = clf.predict(X_va)

        

    score = accuracy_score(y_train, oof)

    return score



study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

study.optimize(objective, n_trials=100)
study.best_params
optuna.importance.get_param_importances(study)
fig = optuna.visualization.plot_param_importances(study)

fig.show(config={"displayModeBar": False})
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = lgb.LGBMClassifier(objective="binary",

                             metric="binary_logloss",

                             **study.best_params)

    clf.fit(X_tr, y_tr, eval_set = [(X_va, y_va)],

            early_stopping_rounds=10,

            verbose=-1)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features], num_iteration=clf.best_iteration_)

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_gbm = np.mean(y_preds, axis=1)



print(f"\nOut-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of optimized LightGBM", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized LightGBM", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets",

            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]

NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = xgb.XGBClassifier()

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of default XGBoost", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default XGBoost", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1-score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["ejection_fraction", "serum_creatinine", "time"]



def objective(trial):

    skf = StratifiedKFold(n_splits=NFOLD)

    models = []

    imp = np.zeros((NFOLD, len(features)))

    oof = np.zeros((len(X_train), ))

    y_preds = np.zeros((len(X_test), NFOLD))



    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

        # print(f"FOLD {fold_id+1}")

        X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]



        clf = xgb.XGBClassifier(n_estimators = trial.suggest_int("n_estimators", 20, 200),

                                max_depth = trial.suggest_int("max_depth", 2, 32),

                                learning_rate = trial.suggest_uniform("learning_rate", 1e-8, 1.0),

                                min_child_weight = trial.suggest_loguniform("min_child_weight", 1e-4, 1e+1),

                                subsample = trial.suggest_uniform("subsample", 0.4, 1.0),

                                colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.4, 1.0),

                                reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),

                                reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),

                                scale_pos_weight = trial.suggest_int("scale_pos_weight", 1, 100)

                                )

        clf.fit(X_tr, y_tr)

        oof[va_idx] = clf.predict(X_va)

        

    score = accuracy_score(y_train, oof)

    return score



study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

study.optimize(objective, n_trials=100)
study.best_params
optuna.importance.get_param_importances(study)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = xgb.XGBClassifier(**study.best_params)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_xgb = np.mean(y_preds, axis=1)



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of optimized XGBoost", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized XGBoost", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets",

            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]

NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = CatBoostClassifier(loss_function="Logloss")

    clf.fit(X_tr, y_tr, eval_set = [(X_va, y_va)],

            early_stopping_rounds=10,

            verbose=False)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.get_feature_importance()



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of default CatBoost", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default CatBoost", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1-score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["ejection_fraction", "serum_creatinine", "time"]



def objective(trial):

    skf = StratifiedKFold(n_splits=NFOLD)

    models = []

    imp = np.zeros((NFOLD, len(features)))

    oof = np.zeros((len(X_train), ))

    y_preds = np.zeros((len(X_test), NFOLD))



    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

        # print(f"FOLD {fold_id+1}")

        X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]



        clf = CatBoostClassifier(loss_function="Logloss",

                                 iterations = trial.suggest_int("iterations", 1000, 6000),

                                 learning_rate = trial.suggest_uniform("learning_rate", 1e-4, 1e-1),

                                 l2_leaf_reg = trial.suggest_loguniform("l2_leaf_reg", 1e-8, 10.0),

                                 # bagging_temperature = trial.suggest_loguniform("bagging_temperature", 1e-8, 100.0),

                                 subsample = trial.suggest_uniform("subsample", 0.4, 1.0),

                                 # random_strength = trial.suggest_loguniform("random_strength", 1e-8, 100.0),

                                 depth = trial.suggest_int("depth", 2, 16),

                                 min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 200),

                                 # od_type = trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),

                                 # od_wait = trial.suggest_int("od_wait", 10, 50)

                                )

        clf.fit(X_tr, y_tr, eval_set = [(X_va, y_va)],

                early_stopping_rounds=10,

                verbose=False)

        oof[va_idx] = clf.predict(X_va)

        

    score = accuracy_score(y_train, oof)

    return score



study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

study.optimize(objective, n_trials=80)
study.best_params
optuna.importance.get_param_importances(study)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = CatBoostClassifier(loss_function="Logloss",

                             **study.best_params)

    clf.fit(X_tr, y_tr, eval_set = [(X_va, y_va)],

            early_stopping_rounds=10,

            verbose=False)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.get_feature_importance()



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_cat = np.mean(y_preds, axis=1)



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of optimized CatBoost", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized CatBoost", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets",

            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]

NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = RandomForestClassifier(random_state=0)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of default Random forest", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default Random forest", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["ejection_fraction", "serum_creatinine", "time"]



def objective(trial):

    skf = StratifiedKFold(n_splits=NFOLD)

    models = []

    imp = np.zeros((NFOLD, len(features)))

    oof = np.zeros((len(X_train), ))

    y_preds = np.zeros((len(X_test), NFOLD))



    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

        # print(f"FOLD {fold_id+1}")

        X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]



        clf = RandomForestClassifier(random_state=0,

                                     n_estimators = trial.suggest_int("n_estimators", 20, 200),

                                     max_depth = trial.suggest_int("max_depth", 2, 32),

                                     min_samples_split = trial.suggest_int("min_samples_split", 2, 16),

                                     min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 16))

        clf.fit(X_tr, y_tr)

        oof[va_idx] = clf.predict(X_va)

        

    score = accuracy_score(y_train, oof)

    return score



study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

study.optimize(objective, n_trials=50)
study.best_params
optuna.importance.get_param_importances(study)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = RandomForestClassifier(random_state=0,

                                 **study.best_params)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_rf = np.mean(y_preds, axis=1)



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of optimized Random forest", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized Random forest", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets",

            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]

NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = ExtraTreesClassifier(random_state=0)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of default Extremely randomized trees", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default Extremely randomized trees", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["age", "ejection_fraction", "serum_creatinine", "time"]



def objective(trial):

    skf = StratifiedKFold(n_splits=NFOLD)

    models = []

    imp = np.zeros((NFOLD, len(features)))

    oof = np.zeros((len(X_train), ))

    y_preds = np.zeros((len(X_test), NFOLD))



    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

        # print(f"FOLD {fold_id+1}")

        X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]



        clf = RandomForestClassifier(random_state=0,

                                     n_estimators = trial.suggest_int("n_estimators", 20, 200),

                                     max_depth = trial.suggest_int("max_depth", 2, 32),

                                     min_samples_split = trial.suggest_int("min_samples_split", 2, 16),

                                     min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 16))

        clf.fit(X_tr, y_tr)

        oof[va_idx] = clf.predict(X_va)

        

    score = accuracy_score(y_train, oof)

    return score



study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

study.optimize(objective, n_trials=50)
study.best_params
optuna.importance.get_param_importances(study)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(features)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = ExtraTreesClassifier(random_state=0,

                               **study.best_params)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_ert = np.mean(y_preds, axis=1)



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), features), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of optimized Extremely randomized trees", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized Extremely randomized trees", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["age_s", "anaemia", "creatinine_phosphokinase_s", "diabetes", "ejection_fraction_s", "high_blood_pressure", "platelets_s",

            "serum_creatinine_s", "serum_sodium_s", "sex", "smoking", "time_s"]

NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = LogisticRegression(random_state=0)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default linear model", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["age_s", "creatinine_phosphokinase_s", "ejection_fraction_s", "serum_creatinine_s", "serum_sodium_s", "time_s"]

NFOLD = 10



def objective(trial):

    skf = StratifiedKFold(n_splits=NFOLD)

    models = []

    imp = np.zeros((NFOLD, len(features)))

    oof = np.zeros((len(X_train), ))

    y_preds = np.zeros((len(X_test), NFOLD))



    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

        # print(f"FOLD {fold_id+1}")

        X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]



        clf = LogisticRegression(random_state=0,

                                 C = trial.suggest_uniform("C", 0.1, 10.0),

                                 intercept_scaling = trial.suggest_uniform("intercept_scaling", 0.1, 2.0),

                                 max_iter = trial.suggest_int("max_iter", 100, 1000)

                                 )

        clf.fit(X_tr, y_tr)

        oof[va_idx] = clf.predict(X_va)

        

    score = accuracy_score(y_train, oof)

    return score



study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

study.optimize(objective, n_trials=20)
study.best_params
optuna.importance.get_param_importances(study)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = LogisticRegression(random_state=0,

                             **study.best_params)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test[features])

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_lm = np.mean(y_preds, axis=1)



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized linear model", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["age_s", "anaemia", "creatinine_phosphokinase_s", "diabetes", "ejection_fraction_s", "high_blood_pressure", "platelets_s",

            "serum_creatinine_s", "serum_sodium_s", "sex", "smoking", "time_s"]

NFOLD = 10

seed_everything(0)



BATCH_SIZE = 32



skf = StratifiedKFold(n_splits=NFOLD)

oof = np.zeros((len(X_train), 1))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(128, activation="relu", input_shape=(len(features), )),

        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Dense(1)

    ])

    

    model.compile(loss="binary_crossentropy",

                  optimizer=tf.keras.optimizers.Adam(lr=0.001,

                                                     decay=0.0),

                  metrics=["accuracy"])

    

    model.fit(X_tr, y_tr,

              validation_data=(X_va, y_va),

              epochs=100, batch_size=BATCH_SIZE,

              verbose=0)

    

    oof[va_idx] = model.predict(X_va, batch_size=BATCH_SIZE, verbose=0)

    y_preds += model.predict(X_test[features], batch_size=BATCH_SIZE, verbose=0) / NFOLD



oof = (np.mean(oof, axis=1) > 0.5).astype(int)

y_pred = (np.mean(y_preds, axis=1) > 0.5).astype(int)



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of deep learning", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
features = ["ejection_fraction_s", "serum_creatinine_s", "serum_sodium_s", "time_s"]

NFOLD = 10

seed_everything(0)



BATCH_SIZE = 32



skf = StratifiedKFold(n_splits=NFOLD)

oof = np.zeros((len(X_train), 1))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train[features].iloc[tr_idx], X_train[features].iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(64, activation="elu", input_shape=(len(features), )),

        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(64, activation="elu"),

        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(1)

    ])

    

    model.compile(loss="binary_crossentropy",

                  optimizer=tf.keras.optimizers.Adam(lr=0.001,

                                                     decay=0.0),

                  metrics=["accuracy"])

    

    model.fit(X_tr, y_tr,

              validation_data=(X_va, y_va),

              epochs=102, batch_size=BATCH_SIZE,

              verbose=0)

    

    oof[va_idx] = model.predict(X_va, batch_size=BATCH_SIZE, verbose=0)

    y_preds += model.predict(X_test[features], batch_size=BATCH_SIZE, verbose=0) / NFOLD



oof = (np.mean(oof, axis=1) > 0.5).astype(int)

y_pred = (np.mean(y_preds, axis=1) > 0.5).astype(int)

y_pred_dl = y_pred



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized deep learning", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);
y_pred_em = y_pred_gbm + y_pred_xgb*2 + y_pred_cat + y_pred_rf + y_pred_ert*2 + y_pred_lm + y_pred_dl

y_pred_em = (y_pred_em > 3.0).astype(int)



print(f"Test accuracy:        {accuracy_score(y_test, y_pred_em)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred_em)}")
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred_em), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of the ensembled model", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1-score={:0.4f}".format(accuracy_score(y_test, y_pred_em), f1_score(y_test, y_pred_em)), fontsize=14)

plt.xticks(np.arange(2), [False, True], fontsize=16)

plt.yticks(np.arange(2), [False, True], fontsize=16);