import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
startups = pd.read_csv('../input/sp-startup/50_Startups.csv') 
df = startups
df.head()
df.info()
df.shape
df.isnull().sum()
df.corr()
sns.heatmap(df.corr())
sns.scatterplot(df["R&D Spend"], df["Profit"],data=df);
df.hist()
df.describe().T
df["State"].unique()
df_state = pd.get_dummies(df["State"])
df_state.head()
df_state.columns = ['New York', 'California', 'Florida']


df = df.drop(["New York"], axis = 1)
df.head()

X = df.drop("Profit", axis = 1)
y = df["Profit"]
X
Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 8, shuffle=1)
X_train
X_test
Y_test
Y_train
model=LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
df_test = pd.DataFrame({"Gerçek Değerler" : Y_test, "Tahmini Değerler" : y_pred})






