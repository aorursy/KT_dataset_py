# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/deep-learning-az-ann/Churn_Modelling.csv")
df.head()
#let us list the columns

df.columns
#let us know the data type of each column

df.dtypes
# let us get the basic statistical description

df.describe()
df.info()
#remove unnecessary columns like row number, surname and customerid

df.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)
# let see refined data

df.head()
import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(df["Age"])

plt.axvline(np.mean(df["Age"]))
# let us see the box plot



sns.boxplot(df["Age"])
# Let us see how many exit

sns.countplot(df["Exited"])
#replace 1 and 0 for the visuals

df["Exited"] = df.Exited.replace({1:"Exited", 0: "Not Exited"})
df.head()
#box plot 

sns.boxplot(x=df["Age"], y=df["Exited"]).set(title="Box plot of person by Age based on exited or not")
#by gender, we will see that we have pretty much more data for male

sns.countplot(df["Gender"]).set(title="Plot by gender")
sns.boxplot(x=df["Age"], y=df["Gender"], hue=df["Exited"]).set(title="Box plot of person by Age based on exited or not")
#Lets see the countplot by Geography, it seems we have more data from France

sns.countplot(df["Geography"]).set(title="Count Plot by geography")

#group by exited

sns.countplot(df["Geography"],hue=df["Exited"]).set(title="Count Plot by geography")

sns.boxplot(x = df["Age"],y = df["Exited"], hue=df["Geography"])
sns.boxplot(x = df["Age"],hue = df["Exited"], y=df["Geography"])
#lets see some more visualization

df.columns
df["HasCrCard"] = df.HasCrCard.replace({1:"Yes", 0:"No"})
sns.countplot(df["HasCrCard"]).set(title="Count Plot of Person who has Credit card")
#let us see how many with credit card has exited

sns.countplot(x=df["HasCrCard"], hue=df['Exited']).set(title="Count Plot of Person who has Credit card")
#lets dras 2d contigency table to see how many people have exited with credit card or another way

pd.crosstab(df["Geography"], df["Exited"],normalize="columns")
pd.crosstab([df.HasCrCard,df.Geography],df.Exited)
#lets normalize with total all

pd.crosstab([df.HasCrCard,df.Geography],df.Exited, normalize='all')
#lets normalize with total Columns

pd.crosstab([df.HasCrCard,df.Geography],df.Exited, normalize='columns')
#lets normalize with total all

pd.crosstab([df.HasCrCard,df.Geography],df.Exited, normalize='columns')
df.columns
pd.crosstab(df.Tenure,df.Exited, normalize="columns")
_, ax =  plt.subplots(1, 2, figsize=(15, 7))

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[0])

sns.scatterplot(x = "Age", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[1])
_ = sns.FacetGrid(df, col="Geography",  row="Exited").map(plt.scatter, "CreditScore", "Balance", alpha=0.3).add_legend() #a;pha is intensity
plt.figure(figsize=(8, 8))

sns.swarmplot(x = "HasCrCard", y = "Age", data = df, hue="Exited")

"""Draw a categorical scatterplot with non-overlapping points.



This function is similar to stripplot(), but the points are adjusted (only along the categorical axis) so that

they don’t overlap. This gives a better representation of the distribution of values,

but it does not scale well to large numbers of observations. 

This style of plot is sometimes called a “beeswarm”."""
"""Initialize the matplotlib figure and FacetGrid object.



This class maps a dataset onto multiple axes arrayed in a grid of rows and columns that correspond to levels of variables in the dataset. The plots it produces are often called “lattice”, “trellis”, or “small-multiple” graphics."""

facet = sns.FacetGrid(df,hue="Exited",aspect=3)

facet.map(sns.kdeplot,"Balance",shade=True)

facet.set(xlim=(0,df["Balance"].max()))

facet.add_legend()



plt.show()
plt.figure(figsize=(12,6))

bplot = df.boxplot(patch_artist=True)

plt.xticks(rotation=90)       

plt.show()
plt.subplots(figsize=(12,8))

sns.heatmap(df.corr(), annot=True, cmap="YlGnBu_r")

plt.show()
df.head(3)
X = df.drop(["Exited","Geography","Gender","HasCrCard","IsActiveMember"], axis=1)

y = df["Exited"]


from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score, f1_score,average_precision_score, confusion_matrix,

                             average_precision_score, precision_score, recall_score, roc_auc_score, )

from mlxtend.plotting import plot_confusion_matrix



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler





from xgboost import XGBClassifier, plot_importance

from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = GaussianNB()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)
clf = LogisticRegression()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)
clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)
clf = RandomForestClassifier(n_estimators = 200, random_state=200)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy_score(pred, y_test)