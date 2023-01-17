import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/merc.csv", low_memory=False)
print("The shape of data:", df.shape)

df.head().T
df.info()
df.describe(include="O")
df.describe()
len(df.model.unique())
fig = plt.figure(figsize=(20, 10))

sns.countplot(y='model', data=df);

plt.title("Car Model");

plt.xlabel("Count of the Cars");

plt.ylabel("Car Model");
df.model.unique().tolist()    
df.model.value_counts()
df.model.value_counts().nlargest(5)
plt.figure(figsize=(20,10))

df.model.value_counts().nlargest(5).plot(kind='bar')

plt.title("Top 5 Most Popular Car");
sns.set(style="whitegrid")

plt.figure(figsize=(15, 10))

mileage_wise_price = df.groupby('model')['mileage'].mean().reset_index()

m=sns.barplot(x='model', y='mileage', data=mileage_wise_price);

m.set_xticklabels(df['model'].values,rotation=30)

plt.title("Model-Wise-Mileage")

plt.show()
plt.figure(figsize=(20,10))

df.groupby('model')['mileage'].mean().sort_values().nlargest(10).plot(kind='bar')

plt.title( "Top 10 cars mileages wise")

plt.ylabel("Mileage")

plt.show()
plt.figure(figsize=(20,10))

df.groupby('model')['mileage'].mean().sort_values().nsmallest(5).plot(kind='bar')

plt.title( "Top 5 cars with lowest mileages")

plt.ylabel("Mileage")

plt.show()
plt.figure(figsize=(20, 5))

year_sale_model = df.groupby('year')['model'].count().reset_index()

c = sns.barplot(x='year', y='model', data=year_sale_model)

c.set_title("Car Sales by Year")

c.set_ylabel("Model Count")

plt.show()
plt.figure(figsize=(20, 10))

y=sns.swarmplot(x='year', hue='model',y='price', data=df, size=10);

y.set_title("Cars Sales in Year and Price Range");
plt.figure(figsize=(20, 10))

f = sns.lineplot(x='model', y='price', hue='transmission', data=df)

f.set_xticklabels(df['model'].values, rotation=30)

f.set_title("Populare Transmission by Price ")

plt.show()
plt.figure(figsize=(20, 10))

f = sns.lineplot(x='model', y='price', hue='fuelType', data=df)

f.set_xticklabels(df['model'].values, rotation=30)

plt.title("Fuel Type Wise Car Price");
plt.figure(figsize=(20, 8))

sns.violinplot(y='mileage', x='fuelType', data=df);

plt.title("Fule Type Wise Price");
#Let's convert categorical features to numbers



labelencoder = LabelEncoder()

df = df.apply(lambda col: labelencoder.fit_transform(col))
plt.figure(figsize=(20,10))

plt.scatter('engineSize', 'mpg', c='mileage', s='price', data=df)

plt.legend()

plt.xlabel('Engine Size')

plt.ylabel('MPG')

plt.show()
plt.figure(figsize=(20,20))

sns.heatmap(df.corr(), annot=True, cmap="RdYlGn");
#Let's make X & y



X = df.drop("price", axis=1)

y = df["price"]



# Split data into train & test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)



lin = LinearRegression().fit(X_train, y_train)

pred = lin.predict(X_test).round(2)

print(f'Linear Model Coefficient (w) :{(lin.coef_)}')

print(f'Linear Model Intercept (b) :{(lin.intercept_).round(2)}')

print("R^2 score on train:", lin.score(X_train, y_train).round(2))

print("R^2 score on test", lin.score(X_test, y_test).round(2))

print("Mean absolue error (MAE):", mean_absolute_error(y_test, pred))

print("Mean squared error (MSE):", mean_squared_error(y_test, pred))
plt.figure(figsize=(20,5))

lent = [i for i in range(len(X_test))]

plt.plot(lent, X_test, 'bs')

plt.plot(lent, y_test, 'ro')

plt.plot(lent, pred, 'g^')

plt.title("Actual Vs Predicted")

plt.xlabel("Actual Price")

plt.ylabel("Prediction Price")

plt.show()
plt.figure(figsize=(20,5))

lent = [i for i in range(len(X_test))]

plt.plot(lent, y_test, 'ro')

plt.plot(lent, pred,'g^')

plt.title("Actual Vs Predicted (Price)")

plt.xlabel("Actual Price")

plt.ylabel("Prediction Price")

plt.show()
plt.figure(figsize=(20,5))

plt.plot(y_test-pred, 'b*')

plt.title("error")

plt.show()
# Error distribution

plt.figure(figsize=(20,10))

sns.distplot(y_test - pred, bins=100)

plt.xlabel('Actual_price - Predicted_price')

plt.ylabel('Index')

plt.title('Error distribution')

plt.show()
df_model = pd.DataFrame(data={"Actual Value": y_test,

                             "Predicted Value": pred})

df_model["Difference"] = df_model["Predicted Value"].round(2)-df_model['Actual Value']

df_model[:10]
plt.figure(figsize=(15,10))

plt.scatter(data=df_model, x='Actual Value', y="Predicted Value", marker="*")

plt.title("");
plt.figure(figsize=(20,10))

sns.scatterplot(x='Actual Value', y='Predicted Value', data=df_model);
rfr = RandomForestRegressor(n_estimators=114, max_depth=15, random_state=100).fit(X_train, y_train)

print("R2", rfr.score(X_test, y_test))

prediction = rfr.predict(X_test)
plt.figure(figsize=(20,5))

lent = [i for i in range(len(X_test))]

plt.plot(lent, X_test, 'bs')

plt.plot(lent, prediction, 'ro')

plt.plot(lent, y_test, 'g^')

plt.title("Actual Vs Predicted")

plt.xlabel("Actual Price")

plt.ylabel("Prediction Price")

plt.show()
plt.figure(figsize=(20,5))

lent = [i for i in range(len(X_test))]

plt.plot(lent, y_test, 'ro')

plt.plot(lent, prediction,'g^')

plt.title("Actual Vs Predicted (Price)")

plt.xlabel("Actual Price")

plt.ylabel("Prediction Price")

plt.show()
plt.figure(figsize=(20,5))

plt.plot(y_test-prediction, 'r*')

plt.title("error")

plt.show()
# Error distribution

plt.figure(figsize=(20,10))

sns.distplot(y_test - prediction, bins=100)

plt.xlabel('Actual_price - Predicted_price')

plt.ylabel('Index')

plt.title('Error distribution')

plt.show()