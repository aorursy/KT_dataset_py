import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("ggplot")

%matplotlib inline
df = pd.read_csv("../input/titanic/train.csv"); df.head()
e_survival = df.Survived.mean(); e_survival
e_survival_given_male = df[df.Sex == "male"].Survived.mean(); e_survival_given_male
dist_survival_given_sex = df.groupby("Sex").Survived.mean(); dist_survival_given_sex
dist_survival_given_sex_and_class = (df

                                     .groupby(["Sex", "Pclass"])

                                     .Survived.mean()

                                    ); dist_survival_given_sex_and_class
dist_survival_given_sex_and_class.plot.bar();
g = sns.jointplot(df.Survived, df.Age, kind="scatter", height=7, space=0, alpha=0.5)
fig, ax = plt.subplots(figsize=(8,6))



(df[["Survived", "Age"]]

 .groupby('Survived')

 .Age

 .plot(kind='hist', ax=ax, alpha=0.5, bins=20, legend=True,

       title="Distribution of Age for survival vs. not survival")

);