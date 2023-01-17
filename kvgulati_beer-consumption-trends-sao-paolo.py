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
from pandas import DataFrame

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn import preprocessing

import seaborn as sns
dataset = pd.read_csv(os.path.join(dirname, filename),decimal=',')

dataset.columns=["Date","Median_Temp","Min_Temp","Max_Temp","Rainfall","Weekend","Consumption"]



#Comvert the Consumption column values to float

dataset["Consumption"] = dataset["Consumption"].astype(float)



dataset.head()
#drop Blank rows read from the input CSV

dataset = dataset.dropna()
plt.figure(figsize=(7,7))

sns.heatmap(dataset.corr())

plt.title("Correlation Heatmap")

plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,4))

fig.suptitle("Beer consumption trends along Max Temperature, Rainfall and Weekends")

ax1.set_xlabel("Max Temperature (celsius)")

ax1.set_ylabel("Beer consumption (litres)")

ax1.scatter(dataset["Max_Temp"], dataset["Consumption"], alpha=0.3)



ax2.set_xlabel("Rainfall (mm)")

ax2.set_ylabel("Beer consumption (litres)")

ax2.scatter(dataset["Rainfall"], dataset["Consumption"], alpha=0.3)



ax3 = sns.boxplot(x=dataset["Weekend"], y=dataset["Consumption"])



plt.show()
fig, (axs1, axs2, axs3) = plt.subplots(1,3, figsize=(16,4))

fig.suptitle("Prediction accuracy with different variables")



X = DataFrame(dataset, columns=["Max_Temp"])

y = DataFrame(dataset, columns=["Consumption"])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 20)



regr = LinearRegression()

regr.fit(X_train, y_train)



y_pred = regr.predict(X_test)

axs1.set_title("Max Temp vs Consumption")

acc = round(r2_score(y_test, y_pred, multioutput='variance_weighted')*100,2)

axs1.set_xlabel(acc)

axs1.scatter(X_train, y_train, alpha=0.4, color="blue")

axs1.plot(X_train, regr.predict(X_train),color="red")





X = DataFrame(dataset, columns=["Min_Temp"])

y = DataFrame(dataset, columns=["Consumption"])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 20)



regr = LinearRegression()

regr.fit(X_train, y_train)



y_pred = regr.predict(X_test)

axs2.set_title("Min Temp vs Consumption")

acc = round(r2_score(y_test, y_pred, multioutput='variance_weighted')*100,2)

axs2.set_xlabel(acc)

axs2.scatter(X_train, y_train, alpha=0.4, color="blue")

axs2.plot(X_train, regr.predict(X_train),color="red")



X = DataFrame(dataset, columns=["Median_Temp"])

y = DataFrame(dataset, columns=["Consumption"])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 20)



regr = LinearRegression()

regr.fit(X_train, y_train)



y_pred = regr.predict(X_test)

axs3.set_title("Median Temp vs Consumption")

acc = round(r2_score(y_test, y_pred, multioutput='variance_weighted')*100,2)

axs3.set_xlabel(acc)

axs3.scatter(X_train, y_train, alpha=0.4, color="blue")

axs3.plot(X_train, regr.predict(X_train),color="red")



plt.show()



fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,4))



X = DataFrame(dataset, columns=["Rainfall"])

y = DataFrame(dataset, columns=["Consumption"])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 20)



regr = LinearRegression()

regr.fit(X_train, y_train)



y_pred = regr.predict(X_test)

ax1.set_title("Rainfall vs Consumption")

acc = round(r2_score(y_test, y_pred, multioutput='variance_weighted')*100,2)

ax1.set_xlabel(acc)

ax1.scatter(X_train, y_train, alpha=0.4, color="blue")

ax1.plot(X_train, regr.predict(X_train),color="red")



X = DataFrame(dataset, columns=["Weekend"])

y = DataFrame(dataset, columns=["Consumption"])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 20)



regr = LinearRegression()

regr.fit(X_train, y_train)



y_pred = regr.predict(X_test)

ax2.set_title("Weekend vs Consumption")

acc = round(r2_score(y_test, y_pred, multioutput='variance_weighted')*100,2)

ax2.set_xlabel(acc)

ax2.scatter(X_train, y_train, alpha=0.4, color="blue")

ax2.plot(X_train, regr.predict(X_train),color="red")



plt.show()
X = DataFrame(dataset, columns=["Max_Temp","Rainfall","Weekend"])

y = DataFrame(dataset, columns=["Consumption"])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 20)



regr = LinearRegression()

regr.fit(X_train, y_train)



y_pred = regr.predict(X_test)



print("Accuracy %age =",round(r2_score(y_test, y_pred, multioutput='variance_weighted')*100,2))