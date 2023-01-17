import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
tt = pd.read_csv("../input/train.csv")
tt.head()
tt.shape
tt.dtypes
tt.dtypes
tt.info()
tt["Sex"] = tt["Sex"].astype("category")
tt.info()
tt.head(10)
tt["Embarked"] = tt["Embarked"].astype("category")
tt.info()
tt["Parch"].unique()
tt["Parch"] = tt["Parch"].astype("category")
tt.info()
tt["Survived"].value_counts()
tt["Survived"].plot()
plt.style.available
plt.style.use("tableau-colorblind10")
tt["Survived"].value_counts().plot(kind = "bar", y = "Survived")
tt.head()
tt["Pclass"].unique()
tt["Pclass"] = tt["Pclass"].astype("category")
tt.info()
tt["SibSp"].unique()
tt["SibSp"] = tt["SibSp"].astype("category")
tt.info()
tt["Pclass"].value_counts().plot(kind = "bar")
tt.head()
tt["Survived"] = tt["Survived"].astype("bool")
tt.info()
mask  = tt["Survived"] == True
mask2 = tt["Sex"] == "female"
tt[mask&mask2]
tt[mask&mask2].shape
tt.head()
mask1 = tt["Survived"] == True

mask3 = tt["Pclass"] == 1
tt[mask1&mask3]
tt[mask1&mask3].shape
mask4 = tt["Pclass"] == 2
tt[mask1&mask4].shape
136+87
(223/342)*100
tt.head()
mask5 = tt["Age"] <=22
tt[mask1&mask5]
mask5 = (tt["Age"] > 22) & (tt["Age"] <=35)
tt[mask1&mask5]
tt["Cabin"].str.contains("NaN").value_counts()
mask6 = tt["Cabin"].str.startswith("A")
tt[mask1&mask6]
tt["Cabin_ID"] = tt["Cabin"].str.extract('([A-Z]+)')
tt["Cabin_No"] = tt["Cabin"].str.replace('([A-Z]+)', '')
tt.head()
tt["Cabin_ID"].value_counts().plot(kind = "bar")
tt[mask1].Cabin_ID.value_counts().plot(kind = "bar")