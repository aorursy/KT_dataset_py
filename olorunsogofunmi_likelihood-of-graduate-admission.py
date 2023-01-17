#Loading the necessary libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

#Loading the dataset

df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df.drop("Serial No.",axis=1,inplace=True)

#Checking if there are any null values in the dataset

df.info()
sns.jointplot("GRE Score","CGPA",data=df,color="purple",kind="scatter")
sns.set_style("darkgrid")

sns.distplot(df["GRE Score"],kde=False)
sns.pairplot(df,palette="coolwarm")
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(df.drop("Chance of Admit ",axis=1))

scaled_features = scaler.transform(df.drop("Chance of Admit ",axis=1))

df_new = pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_new.head()

X = df_new

y = df["Chance of Admit "]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)



model = LinearRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,predictions)))