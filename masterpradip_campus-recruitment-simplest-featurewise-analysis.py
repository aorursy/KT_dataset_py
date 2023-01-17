# Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
# Load data

path = "../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv"

df = pd.read_csv(path)

df.head()
# Drop sl_no

df.drop("sl_no", axis=1, inplace=True)

df.info()
# Fill NaN in salary as 0

df["salary"].fillna(value=0, inplace=True)

df.info()
# Gender vs Status

sns.countplot(x="gender", hue="status", data=df)
# Placed/Notplaced

df_placed = df.loc[df["status"] == "Placed"]

df_notplaced = df.loc[df["status"] == "Not Placed"]
df_placed.head()
df_notplaced.head()
# Male vs Female among placed candidates

px.pie(values=df_placed["gender"].value_counts().tolist(), 

        names=list(dict(df_placed["gender"].value_counts())), 

        title="Male v Female", 

        color_discrete_sequence=["red", "blue"])
# Distribution of salary among placed candidates over gender

sns.kdeplot(df_placed.salary[df.gender=="M"])

sns.kdeplot(df_placed.salary[df.gender=="F"])

plt.legend(["Male", "Female"])

plt.xlabel("salary")

plt.title("Distribution of salary among placed candidates over gender")
# Distribution of salary in placed and non-placed candidates over ssc_p

sns.kdeplot(df_placed["ssc_p"])

sns.kdeplot(df_notplaced["ssc_p"])

plt.legend(["Placed", "Non-placed"])

plt.xlabel("ssc_p")

plt.title("Distribution of salary among placed and non-placed candidates over SSC%")
# Distribution of ssc_b among placed candidates

px.pie(values=df_placed["ssc_b"].value_counts().tolist(), 

        names=list(dict(df_placed["ssc_b"].value_counts())), 

        title="Distribution of SSC board among placed candidates")
# Distribution of salary in placed and non-placed candidates over hsc_p

sns.kdeplot(df_placed["hsc_p"])

sns.kdeplot(df_notplaced["hsc_p"])

plt.legend(["Placed", "Non-placed"])

plt.xlabel("hsc_p")

plt.title("Distribution of salary among placed and non-placed candidates over HSC%")
# Distribution of hsc_b among placed candidates

px.pie(values=df_placed["hsc_b"].value_counts().tolist(), 

        names=list(dict(df_placed["hsc_b"].value_counts())), 

        title="Distribution of HSC board among placed candidates")
# Distribution of hsc_s among placed candidates

px.pie(values=df_placed["hsc_s"].value_counts().tolist(), 

        names=list(dict(df_placed["hsc_s"].value_counts())), 

        title="Distribution of HSC streams among placed candidates")
# Distribution of salary in placed and non-placed candidates over degree_p

sns.kdeplot(df_placed["degree_p"])

sns.kdeplot(df_notplaced["degree_p"])

plt.legend(["Placed", "Non-placed"])

plt.xlabel("degree_p")

plt.title("Distribution of salary among placed and non-placed candidates over Degree%")
# Distribution of degree_t among placed candidates

px.pie(values=df_placed["degree_t"].value_counts().tolist(), 

        names=list(dict(df_placed["degree_t"].value_counts())), 

        title="Distribution of Degree streams among placed candidates")
# Distribution of workex among placed candidates

px.pie(values=df_placed["workex"].value_counts().tolist(), 

        names=list(dict(df_placed["workex"].value_counts())), 

        title="Distribution of work experience among placed candidates")
# Distribution of salary in placed and non-placed candidates over etest_p

sns.kdeplot(df_placed["etest_p"])

sns.kdeplot(df_notplaced["etest_p"])

plt.legend(["Placed", "Non-placed"])

plt.xlabel("etest_p")

plt.title("Distribution of salary among placed and non-placed candidates over Etest%")
# Distribution of specialisation among placed candidates

px.pie(values=df_placed["specialisation"].value_counts().tolist(), 

        names=list(dict(df_placed["specialisation"].value_counts())), 

        title="Distribution of specialisation among placed candidates")
# Distribution of salary in placed and non-placed candidates over mba_p

sns.kdeplot(df_placed["mba_p"])

sns.kdeplot(df_notplaced["mba_p"])

plt.legend(["Placed", "Non-placed"])

plt.xlabel("mba_p")

plt.title("Distribution of salary among placed and non-placed candidates over MBA%")
# Encoding categorical features

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



cat_features = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_p", "degree_t", "workex", "specialisation", "status"]

df[cat_features] = df[cat_features].apply(le.fit_transform)

df.head()
# Correlation among all the features

plt.figure(figsize=(18, 15))

sns.heatmap(data=df.corr(), annot=True)
# Effect of different fearure on each other

plt.figure(figsize=(18, 15))

sns.pairplot(data=df)
# Most important features

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_classif



X = df.drop(["status", "salary"], axis=1)

y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)



selector = SelectKBest(f_classif)

selector.fit(X_train, y_train)



print(*zip(X.columns, selector.scores_))