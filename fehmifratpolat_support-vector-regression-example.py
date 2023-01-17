import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/diamonds/diamonds.csv")
df.head()
df.drop(columns="Unnamed: 0", inplace= True)
df.head()
df.describe()
plt.figure(figsize=(10,8))
plt.scatter(x="depth", y="price", data=df)

plt.ylabel("Depth")
plt.xlabel("Price")
sns.countplot(x="color", data=df)
sns.countplot(x="cut", data=df)
df["cut"].unique()
df["color"].unique()
df["clarity"].unique()
cut_dict = {"Fair" : 1, "Good" : 2, "Very Good" : 3, "Premium" : 4, "Ideal" : 5}
color_dict = {'E' : 1, 'I' : 2, 'J' : 3, 'H' : 4, 'F' : 5, 'G' : 6, 'D' : 7}
clarity_dict = {'SI2' : 1, 'SI1' : 2, 'VS1' : 3, 'VS2' : 4, 'VVS2' : 5, 'VVS1' : 6, 'I1' :7, 'IF' : 8}

df["cut"] = df["cut"].map(cut_dict)
df["color"] = df["color"].map(color_dict)
df["clarity"] = df["clarity"].map(clarity_dict)
df.head()
df.info()
df.isnull().sum()
df = sklearn.utils.shuffle(df)
df.head()
X = df.drop("price", axis = 1).values
y = df["price"].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
clf = svm.SVR(kernel='linear')
clf.fit(X_train, y_train)
clf.score(X_test, y_test)