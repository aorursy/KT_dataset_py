import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.getcwd()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
sns.set_context(context="notebook", font_scale=1.5)

sns.set_style(style="whitegrid")
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

df.head()
df.info()
df.nunique()
print("gender :")

print(df["gender"].unique())

print("")

print("race/ethnicity :")

print(df["race/ethnicity"].unique())

print("")

print("parental level of education :")

print(df["parental level of education"].unique())

print("")

print("lunch :")

print(df["lunch"].unique())

print("")

print("test preparation course :")

print(df["test preparation course"].unique())
plt.figure(figsize=(5,5))

sns.countplot(data=df, x="gender")
plt.figure(figsize=(10,5))

sns.countplot(data=df.sort_values(by="race/ethnicity"), x="race/ethnicity")
plt.figure(figsize=(18,5))

sns.countplot(data=df, x="parental level of education")
plt.figure(figsize=(5,5))

sns.countplot(data=df, x="lunch")
plt.figure(figsize=(5,5))

sns.countplot(data=df, x="test preparation course")
plt.figure(figsize=(25,5))



plt.subplot(1,3,1)

sns.distplot(df["math score"], kde=False)

plt.title("math score")

plt.ylabel("Number")



plt.subplot(1,3,2)

sns.distplot(df["reading score"], kde=False)

plt.title("reading score")

plt.ylabel("Number")



plt.subplot(1,3,3)

sns.distplot(df["writing score"], kde=False)

plt.title("writing score")

plt.ylabel("Number")
x="gender"



plt.figure(figsize=(25,5))



plt.subplot(1,3,1)

sns.boxplot(data=df, x=x, y='math score')



plt.subplot(1,3,2)

sns.boxplot(data=df, x=x, y='reading score')



plt.subplot(1,3,3)

sns.boxplot(data=df, x=x, y='writing score')
x="race/ethnicity"

df_s = df.sort_values(by=x)



plt.figure(figsize=(25,5))



plt.subplot(1,3,1)

sns.boxplot(data=df_s, x=x, y='math score')



plt.subplot(1,3,2)

sns.boxplot(data=df_s, x=x, y='reading score')



plt.subplot(1,3,3)

sns.boxplot(data=df_s, x=x, y='writing score')
x="parental level of education"



plt.figure(figsize=(25,5))



plt.subplot(1,3,1)

sns.boxplot(data=df, x=x, y='math score')

plt.xticks(rotation=90)



plt.subplot(1,3,2)

sns.boxplot(data=df, x=x, y='reading score')

plt.xticks(rotation=90)



plt.subplot(1,3,3)

sns.boxplot(data=df, x=x, y='writing score')

plt.xticks(rotation=90)
x="lunch"



plt.figure(figsize=(25,5))



plt.subplot(1,3,1)

sns.boxplot(data=df, x=x, y='math score')



plt.subplot(1,3,2)

sns.boxplot(data=df, x=x, y='reading score')



plt.subplot(1,3,3)

sns.boxplot(data=df, x=x, y='writing score')
x="test preparation course"



plt.figure(figsize=(25,5))



plt.subplot(1,3,1)

sns.boxplot(data=df, x=x, y='math score')



plt.subplot(1,3,2)

sns.boxplot(data=df, x=x, y='reading score')



plt.subplot(1,3,3)

sns.boxplot(data=df, x=x, y='writing score')
df.groupby("gender").mean()
df_gp = df.groupby("gender").mean()

df_long = pd.DataFrame(df_gp.stack())

df_long = df_long.reset_index()

df_long.columns=("gender", "Exams", "Avg. Score")

df_long.head()



sns.pointplot(data=df_long, x="Exams", y="Avg. Score", hue="gender", hue_order=["male", "female"])

plt.yticks(np.arange(50, 100, 10))

plt.xticks(rotation=90)

plt.legend(title="gender", bbox_to_anchor=(1,1))
df.groupby("race/ethnicity").mean()
df_gp = df.groupby("race/ethnicity").mean()

df_long = pd.DataFrame(df_gp.stack())

df_long = df_long.reset_index()

df_long.columns=("race/ethnicity", "Exams", "Avg. Score")

df_long.head()



sns.pointplot(data=df_long, x="Exams", y="Avg. Score", hue="race/ethnicity")

plt.yticks(np.arange(50, 100, 10))

plt.xticks(rotation=90)

plt.legend(title="race/ethnicity", bbox_to_anchor=(1,1))
df.groupby("parental level of education").mean()
df_gp = df.groupby("parental level of education").mean()

df_long = pd.DataFrame(df_gp.stack())

df_long = df_long.reset_index()

df_long.columns=("parental level of education", "Exams", "Avg. Score")

df_long.head()



sns.pointplot(data=df_long, x="Exams", y="Avg. Score", hue="parental level of education")

plt.yticks(np.arange(50, 100, 10))

plt.xticks(rotation=90)

plt.legend(title="parental level of education", bbox_to_anchor=(1,1))
df.groupby("lunch").mean()
df_gp = df.groupby("lunch").mean()

df_long = pd.DataFrame(df_gp.stack())

df_long = df_long.reset_index()

df_long.columns=("lunch", "Exams", "Avg. Score")

df_long.head()



sns.pointplot(data=df_long, x="Exams", y="Avg. Score", hue="lunch")

plt.yticks(np.arange(50, 100, 10))

plt.xticks(rotation=90)

plt.legend(title="lunch", bbox_to_anchor=(1,1))
df.groupby("test preparation course").mean()
df_gp = df.groupby("test preparation course").mean()

df_long = pd.DataFrame(df_gp.stack())

df_long = df_long.reset_index()

df_long.columns=("test preparation course", "Exams", "Avg. Score")

df_long.head()



sns.pointplot(data=df_long, x="Exams", y="Avg. Score", hue="test preparation course")

plt.yticks(np.arange(50, 100, 10))

plt.xticks(rotation=90)

plt.legend(title="test preparation course", bbox_to_anchor=(1,1))