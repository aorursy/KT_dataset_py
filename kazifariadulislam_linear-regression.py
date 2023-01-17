import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/delhi-house-price-prediction/MagicBricks.csv')

df
df.isnull().sum().sum()
df.info()
print("Number of entries in Area column with null values are:",df['Area'].isnull().sum().sum())

print("Number of entries in BHK column with null values are:",df['BHK'].isnull().sum().sum())

print("Number of entries in Bathroom column with null values are:",df['Bathroom'].isnull().sum().sum())

print("Number of entries in Furnishing column with null values are:",df['Furnishing'].isnull().sum().sum())

print("Number of entries in Locality column with null values are:",df['Locality'].isnull().sum().sum())

print("Number of entries in Parking column with null values are:",df['Parking'].isnull().sum().sum())

print("Number of entries in Price column with null values are:",df['Price'].isnull().sum().sum())

print("Number of entries in Status column with null values are:",df['Status'].isnull().sum().sum())

print("Number of entries in Transaction column with null values are:",df['Transaction'].isnull().sum().sum())

print("Number of entries in Type column with null values are:",df['Type'].isnull().sum().sum())

print("Number of entries in Per_Sqft column with null values are:",df['Per_Sqft'].isnull().sum().sum())
df = df.drop(['Per_Sqft'], axis=1)
df['Bathroom'].mode()
df['Bathroom'].fillna(value = 2, inplace = True)
df[df['Furnishing'].isnull()]
df['Furnishing'].fillna(value = "Unspecified", inplace = True)
df[df['Parking'].isnull()]
plt.scatter(df['Parking'],df.Area)
df['Parking'].fillna(value = 1, inplace = True)
df[df['Type'].isnull()]
df.Type.fillna(value = 'Unspecified', inplace = True)
df.loc[509:512]
train_data,test_data = train_test_split(df,train_size = 0.8,random_state=3)



lr = LinearRegression()

X_train = np.array(df[['Area',]], dtype=pd.Series).reshape(-1,1)

y_train = np.array(df['Price'], dtype=pd.Series)

lr.fit(X_train,y_train)



X_test = np.array(test_data['Area'], dtype=pd.Series).reshape(-1,1)

y_test = np.array(test_data['Price'], dtype=pd.Series)



pred = lr.predict(X_test)
plt.scatter(X_test,y_test,color='#224499',label="Data", alpha=.6)

plt.plot(X_test,lr.predict(X_test),color="#ff5577",label="Predicted Regression Line")

plt.xlabel("Living Area (sqft)")

plt.ylabel("Price (INR)")

plt.legend()