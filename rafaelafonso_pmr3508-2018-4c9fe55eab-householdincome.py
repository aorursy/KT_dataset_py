import pandas as pd
import sklearn
household = pd.read_csv("../input/train.csv",
            sep=r'\s*,\s*',
            engine='python',
            na_values="NaN")
household.shape
household.head()
import matplotlib.pyplot as plt
household["Target"].value_counts()
household["Target"].value_counts().plot(kind="pie")
household["rooms"].value_counts().plot(kind="bar")
household["v18q"].value_counts()
household["tamviv"].value_counts().plot(kind="bar")
household["escolari"].value_counts().plot(kind="pie")
household["age"].mean()
XYhouse = household[["v2a1","rooms","v14a","v18q","tamviv","escolari","age","female","estadocivil3","Target"]]
XYhouse.shape
nHouse = XYhouse.dropna()
nHouse.shape
household.shape
Xhouse = nHouse[["v2a1","rooms","v14a","v18q","tamviv","escolari","age","female","estadocivil3"]]
Yhouse = nHouse.Target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=45)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xhouse, Yhouse, cv=10)
scores
sum(scores) / len(scores)