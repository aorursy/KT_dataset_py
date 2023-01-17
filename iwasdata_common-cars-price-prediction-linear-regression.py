import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
df = pd.read_csv("../input/autos.csv", encoding = "iso8859-1")



df = df.drop(["dateCrawled", "abtest", "dateCreated", "nrOfPictures", "lastSeen", "postalCode", "seller", "offerType", "model"], axis = 1)





df["monthOfRegistration"] = np.where(df["monthOfRegistration"] == 0, 6, df["monthOfRegistration"])

df["registration"] = df["yearOfRegistration"] + (df["monthOfRegistration"] -1) /12

df = df.drop(["yearOfRegistration", "monthOfRegistration"], axis = 1)



df = df.drop(df[(df["powerPS"] == 0) | (df["price"] == 0)].index)



df["notRepairedDamage"] = np.where(df["notRepairedDamage"] == "ja", 1, df["notRepairedDamage"])

df["notRepairedDamage"] = np.where(df["notRepairedDamage"] == "nein", 0, df["notRepairedDamage"])

df = df[df["notRepairedDamage"].notnull()]

#convert values to integer so I can work with them / visualize them more easiliy

df["notRepairedDamage"] = pd.to_numeric(df["notRepairedDamage"])

 

df = df[(df["price"] < 100000) & (df["powerPS"] < 2000) & (df["registration"] <= 2019)]
g = sns.pairplot(df.sample(300))

plt.show()
print("cars with unrepaired damage: " + str(len(df[df["notRepairedDamage"] == 1])))

print("cars without unrepaired damage: " + str(len(df[df["notRepairedDamage"] == 0])))

print("total cars: " + str(245541 + 29962))

print("percentage with unrepaired damage: " + str(100 / 275503 * 29962) + "%")

print("percentage without unrepaired damage: " + str(100 / 275503 * 245541) + "%")

labels = 'no unrepaired damage', 'unrepaired damage'

sizes = [89.12461933263884, 10.875380667361155]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Distribution of unrepaired damaged cars in data", fontweight="bold")



plt.show()
print(df[df["notRepairedDamage"]==1]["price"].mean())

print(df[df["notRepairedDamage"]==0]["price"][:29962].mean())
df2 = pd.get_dummies(df, columns = ["vehicleType", "gearbox", "fuelType", "brand"])

df2.head()
df2 = df2[(df2["price"] > 500) & (df2["price"] < 20000)]
df2.head()
y = df2[["price"]].values

X = df2.drop(["price", "name"], axis = 1).values



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)
model = LinearRegression()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))