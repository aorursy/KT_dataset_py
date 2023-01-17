import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

df.drop("Serial No.", inplace=True, axis=1)

df.describe()
import seaborn as sns

sns.heatmap(df.corr(), annot=True,cmap="YlGnBu")
plt.figure(figsize=(20,10))

sns.set(style="whitegrid")

print(df.columns)

ax = sns.lineplot(y=df["Chance of Admit "], x=df["GRE Score"])
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(20,10))



from sklearn import preprocessing

x = df.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_sc = pd.DataFrame(x_scaled, columns=df.columns)

df_sc.boxplot()
from sklearn.linear_model import LinearRegression as lr

from sklearn.metrics import mean_squared_error



x = df.drop("Chance of Admit ", axis=1)

y = df["Chance of Admit "]



#Split train test 80-20

x_train = x[:400]

x_test = x[400:]

y_train = y[:400]

y_test = y[400:]



print("Predicting the chance of admission with a linear regression model...")

#Linear regression model

reg = lr().fit(x_train, y_train)

rp = reg.predict(x_test)



error = mean_squared_error(y_test, rp)



print("Mean squared_error:",error)
i=0

#Pretty close predictions

print("Real vs predicted (in %, first 15 examples):\n")

for t in list(y_test[:15]):

    print(np.round(t*100,2),"vs",np.round(rp[i]*100,3))

    i+=1