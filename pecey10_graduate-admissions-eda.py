import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

sns.set()

%matplotlib inline
df = pd.read_csv("../input/Admission_Predict.csv")

df.head()
df.info()
df.drop("Serial No.", axis=1, inplace=True)

df.columns = df.columns.map(lambda x: x.strip().replace(" ","_"))
for column in df.columns:

    if(df[column].nunique() < 10):

        fig = sns.countplot(df[column])

    else:

        fig = sns.distplot(df[column], hist_kws={"alpha": 1}, kde=False)

    plt.title("Distribution of " + column.replace("_", " "))

    plt.show()
fig = sns.scatterplot(x="CGPA", y="Chance_of_Admit", data=df)

plt.title("CGPA vs Chance of Admit")

plt.show()



fig = sns.scatterplot(x="TOEFL_Score", y="Chance_of_Admit", data=df)

plt.title("TOEFL Score vs Chance of Admit")

plt.show()



fig = sns.scatterplot(x="GRE_Score", y="Chance_of_Admit", data=df)

plt.title("GRE Score vs Chance of Admit")

plt.show()
df[["CGPA", "GRE_Score", "TOEFL_Score", "Chance_of_Admit"]].corr()
fig = sns.boxplot(x="SOP", y="Chance_of_Admit", data=df)

plt.title("SOP vs Chance of Admit")

plt.show()



fig = sns.boxplot(x="LOR", y="Chance_of_Admit", data=df)

plt.title("LOR vs Chance of Admit")

plt.show()



fig = sns.boxplot(x="Research", y="Chance_of_Admit", data=df)

plt.title("Research vs Chance of Admit")

plt.show()
fig = sns.boxplot(x="LOR", y="CGPA", data=df)

plt.title("CGPA vs LOR")

plt.show()



fig = sns.boxplot(x="LOR", y="CGPA", data=df[df.Research == 0])

plt.title("CGPA vs LOR - Without Research")

plt.show()



fig = sns.boxplot(x="LOR", y="CGPA", data=df[df.Research == 1])

plt.title("CGPA vs LOR - With Research")

plt.show()
fig = sns.boxplot(x="Research", y="CGPA", data=df)

plt.title("Research Experience vs CGPA")

plt.show()
fig = sns.boxplot(x="Research", y="LOR", data=df)

plt.title("Research Experience vs LOR")

plt.show()



fig = sns.boxplot(x="Research", y="SOP", data=df)

plt.title("Research Experience vs SOP")

plt.show()
fig = sns.boxplot(x="SOP", y="TOEFL_Score", data=df)

plt.title("SOP vs TOEFL Score")

plt.show()
fig = sns.scatterplot(x="CGPA", y="GRE_Score", data=df)

plt.title("CGPA vs GRE Score")

plt.show()