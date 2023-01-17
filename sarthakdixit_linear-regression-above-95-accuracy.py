import pandas as pd

df = pd.read_csv("../input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
df.head()
df.describe(include="all")
df = df.drop("floor", axis=1)
# Cleaning Data

df['area'].fillna(df['area'].median(), inplace = True)

df['rooms'].fillna(df['rooms'].median(), inplace = True)

df['bathroom'].fillna(df['bathroom'].median(), inplace = True)

df['hoa (R$)'].fillna(df['hoa (R$)'].median(), inplace = True)

df['rent amount (R$)'].fillna(df['rent amount (R$)'].median(), inplace = True)

df['property tax (R$)'].fillna(df['property tax (R$)'].median(), inplace = True)

df['fire insurance (R$)'].fillna(df['fire insurance (R$)'].median(), inplace = True)
df.head()
X = df.iloc[:, 0:11].values

y = df.iloc[:, -1].values
# Encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:,0] = labelencoder_X_1.fit_transform(X[:, 0])

X[:,5] = labelencoder_X_1.fit_transform(X[:, 5])

X[:,6] = labelencoder_X_1.fit_transform(X[:, 6])

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("city", OneHotEncoder(), [0])], remainder = 'passthrough')

X = ct.fit_transform(X)

dataframe = pd.DataFrame.from_records(X)

dataframe = dataframe.drop(dataframe.columns[0], axis=1)

X = dataframe.iloc[:, 0:14]

dataframe
# Splitting dataset into training and testing set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 20)
# Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

lr.score(X_test, y_test)
# Predicting rent pricing of houses with different features

import numpy as np

new_pred = lr.predict(sc.transform(np.array([[0.0, 0.0, 0.0, 1.0, 70, 2, 1, 1, 0, 0, 2065, 3300, 211, 42]])))

print(new_pred)

new_pred = lr.predict(sc.transform(np.array([[0.0, 1.0, 0.0, 0.0, 60, 2, 1, 0, 0, 0, 2000, 3000, 100, 30]])))

print(new_pred)

new_pred = lr.predict(sc.transform(np.array([[0.0, 0.0, 1.0, 0.0, 50, 1, 0, 1, 0, 0, 1000, 2000, 150, 40]])))

print(new_pred)

new_pred = lr.predict(sc.transform(np.array([[1.0, 0.0, 0.0, 0.0, 65, 2, 1, 1, 1, 0, 2500, 1000, 200, 20]])))

print(new_pred)