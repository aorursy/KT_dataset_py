import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/Admission_Predict.csv')
df.columns = df.columns.to_series().apply(lambda x: x.strip())
df.head()
df.shape
df = df.dropna()
df.shape
df.describe()
plt.figure(figsize=(10, 5))
p = sns.heatmap(df.corr(), annot=True)
p = sns.pairplot(df)
p = sns.countplot(x="Research", data=df)
p = sns.countplot(x="University Rating", data=df)
p = sns.lineplot(x="GRE Score", y="Chance of Admit", data=df)
_ = plt.title("GRE Score vs Chance of Admit")
p = sns.lineplot(x="TOEFL Score", y="Chance of Admit", data=df)
_ = plt.title("TOEFL Score vs Chance of Admit")
p = sns.lineplot(x="University Rating", y="Chance of Admit", data=df)
_ = plt.title("University Rating vs Chance of Admit")
p = sns.lineplot(x="CGPA", y="Chance of Admit", data=df)
_ = plt.title("CGPA vs Chance of Admit")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
df = pd.read_csv("../input/Admission_Predict.csv")
df.columns = df.columns.to_series().apply(lambda x: x.strip())
serialNo = df["Serial No."].values
df.drop(["Serial No."], axis=1, inplace=True)

df = df.rename(columns = {'Chance of Admit': 'Chance of Admit'})

y = df["Chance of Admit"].values
x_data = df.drop(["Chance of Admit"], axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=123)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
mean_squared_error(y_test, predictions)