import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
startups = pd.read_csv("50_Startups.csv")
df = startups.copy()
df.head(5)
df.info()
df.shape
df.isnull().sum()
df.corr()
sns.heatmap(df.corr())
sns.scatterplot(x = df["R&D Spend"], y = df["Profit"], color="blue");
df.hist()
df.describe()
df.describe().T
df["State"] = pd.Categorical(df["State"])
stateDummies = pd.get_dummies(df["State"], prefix = "State")
df = pd.concat([df,stateDummies],axis = 1)
df.head()
df.drop(["State","State_California"],axis = 1, inplace = True)
df.head(5)
Y = df["Profit"]
X = df["R&D Spend"]
Y
X
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3,
                                                    random_state = 42, shuffle=1)
X_train
X_test
Y_train
Y_test
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)#BURADA TAKILDIM BAYAĞI ARAŞTIMA YAPTIM AMA BU HATAYI ÇÖZEMEDİM.








