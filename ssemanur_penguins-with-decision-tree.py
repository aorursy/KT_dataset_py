import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv", sep=',')

df.head()
df.info()
df.isnull().sum()
df.dropna(inplace=True)
sns.distplot(a=df['culmen_length_mm'])
sns.distplot(a=df['culmen_depth_mm'])
sns.distplot(a=df['flipper_length_mm'])
y_is_class = pd.crosstab(index=df["island"], 

                             columns=df["species"],

                             margins=True,normalize='index')



y_is_class
y_sex_class = pd.crosstab(index=df["sex"], 

                             columns=df["species"],

                             margins=True,normalize='index')



y_sex_class
df[df["sex"]=="."]
df.drop([336],inplace=True)
plt.pie(df["species"].value_counts(),labels = df["species"].unique())

plt.show()
corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(1, 139, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df=pd.get_dummies(df, columns = ["species"], prefix = ["species"])

df=pd.get_dummies(df, columns = ["island"], prefix = ["island"])

df=pd.get_dummies(df, columns = ["sex"], prefix = ["sex"])

del df["island_Dream"]

del df["sex_MALE"]
df.head()
y = df[df.columns[4:7]]

X = df.drop(df[df.columns[4:7]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30, 

                                                    random_state=985)
cart = DecisionTreeClassifier()

cart_tuned = DecisionTreeClassifier(max_depth = 6, min_samples_split = 54).fit(X_train, y_train)

y_pred = cart_tuned.predict(X_test)

print(classification_report(y_test, y_pred))
accuracy_score(y_test,y_pred)