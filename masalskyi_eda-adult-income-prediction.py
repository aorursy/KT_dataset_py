# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(16, 12)}); # you can change this if needed
df = pd.read_csv("../input/adult.csv")
df.head()
df.info()
# Визуализируйте распределение значений возраста для лиц, зарабатывающих < 50K

df[df["income"] =="<=50K"]["age"].hist()

plt.xlabel("age")

plt.ylabel("count")

plt.show()
# Найдите среднее значение возраста для лиц, зарабатывающих < 50K

df[df["income"] =="<=50K"]["age"].mean()
# Найдите стандартное отклонение возраста для лиц, зарабатывающих < 50K

df[df["income"] =="<=50K"]["age"].std()
# Визуализируйте распределение значений возраста для лиц, зарабатывающих > 50K

df[df["income"] ==">50K"]["age"].hist()

plt.xlabel("age")

plt.ylabel("count")

plt.show()
# Найдите среднее значение возраста для лиц, зарабатывающих > 50K

df[df["income"] ==">50K"]["age"].mean()
# Найдите стандартное отклонение возраста для лиц, зарабатывающих  50K

df[df["income"] ==">50K"]["age"].std()
df[df["income"] ==">50K"]["education"].value_counts()
df["relationship"].value_counts()
df[(df["relationship"] == "Husband") ]["income"].value_counts(normalize = True)
df[(df["relationship"] == "Unmarried") & (df["gender"] == "Male")]["income"].value_counts(normalize = True)
df["hours-per-week"].max()
df[df["hours-per-week"] == 99].shape[0]
df[df["hours-per-week"] == 99]["income"].value_counts(normalize = True)
df["income_cat"] = df["income"].map({"<=50K":0,">50K":1})
from scipy.stats import pointbiserialr

pointbiserialr(df['hours-per-week'], df['income_cat'])
df.groupby("education")["hours-per-week"].mean().plot(kind = "bar")

plt.ylabel('hours-per-week') 

plt.show();
sns.countplot(y = "native-country", hue = "income", data = df)
df.groupby("race")["income"].value_counts(normalize = True)
from scipy.stats import fisher_exact

fisher_exact(pd.crosstab(df['gender'], df['income']))
df.groupby("gender")["income"].value_counts(normalize = True)
df[df["capital-gain"]> 0 ]["capital-gain"].hist()
df[df["capital-loss"] > 0 ]["capital-loss"].hist()
df[(df["capital-loss"] > 0) & (df["gender"] == "Male")]["income"].value_counts(normalize = True)
df[(df["capital-loss"] > 0) & (df["gender"] == "Female")]["income"].value_counts(normalize = True)
df[df["capital-gain"]> 0 ].groupby("gender")["income"].value_counts(normalize = True)
df["fnlwgt"].sum()//1000000000
df["educational-num"].hist()
from scipy.stats import spearmanr

spearmanr(df["educational-num"], df["hours-per-week"])
numeric = ["age","educational-num", "hours-per-week", "capital-gain", "capital-loss","fnlwgt"]
sns.pairplot(df[numeric])
sns.boxplot(df['age'])
sns.heatmap(df[numeric].corr(method='spearman'))
sns.countplot(y = "age", hue = "gender", data = df[df["income"]== ">50K"])
sns.countplot(y = "educational-num", hue = "income", data = df)
sns.countplot(y = "workclass", hue = "income", data = df)