

import numpy as np #

import pandas as pd

import seaborn as sns #visualisation

import matplotlib.pyplot as plt 

%matplotlib inline

sns.set(color_codes=True)

from sklearn import datasets , linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

lr=LinearRegression(normalize=True)

from sklearn.metrics import accuracy_score
df=pd.read_csv('/kaggle/input/mydata/data.csv')
df.head(10) #starting 10 rows of the dataset
df.tail(10) #displaying last 10 rows of the data
df.info() #getting the summary of the dataframe
df.shape #for  summary of the data frame
df.describe(include='all')
df.isnull().sum() #checking if there are some null values


df = df.drop(['Engine Fuel Type', 'Number of Doors','Market Category'], axis=1) #these are the colomn to be drop which meant to no use to me .

df.head(5)
df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode","highway MPG": "MPG-H", "city mpg": "MPG-C", "MSRP": "Price" })

df.head(5) #Renaming the coloumn
df.head(10) #data after rename
df.isnull().sum() #checking null values after dropping the coloumn  .
df.shape #Total number of rows and columns
duplicate_rows_df = df[df.duplicated()]

print("number of duplicate rows: ", duplicate_rows_df.shape)


df = df.drop_duplicates()

df.head(5)
df.shape #Total number of rows and columnsafter dropping the values
df = df.dropna()

df.count()
sns.boxplot(x=df['Price'])
sns.boxplot(x=df['HP'])
sns.boxplot(x=df['Cylinders'])
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

df.shape
sns.boxplot(x=df['Price']) #boxplot of price
sns.boxplot(x=df['HP']) #boxplot of hp
sns.boxplot(x=df['Cylinders']) #boxplot of cylinders
counts = df['Make'].value_counts()*100/sum(df['Make'].value_counts())

popular_labels = counts.index[:15]

plt.figure(figsize=(10,5))

plt.barh(popular_labels, width=counts[:15])

plt.title('Top 15 Car brands')

plt.show()
prices = df[['Make','Price']].loc[(df['Make'] == 'Chevrolet')|

               (df['Make'] == 'Ford')|

               (df['Make'] == 'Volkswagen')|

               (df['Make'] == 'Toyota')|

               (df['Make'] == 'Dodge')|

               (df['Make'] == 'Nissan')|

               (df['Make'] == 'GMC')|

               (df['Make'] == 'Honda')|

               (df['Make'] == 'Mazda')].groupby('Make').mean()

print(prices)
df.corr() #FINDING THE CORRELATION MATRIX FROM HERE
corrMatrix = df.corr()

sns.heatmap(corrMatrix,annot=True)


sns.barplot(df['Year'],df['Price'])
sns.barplot(df['Cylinders'],df['Price'])
sns.barplot(df['MPG-H'],df['Price'])
sns.barplot(df['MPG-C'],df['Price'])
sns.barplot(df['Popularity'],df['Price'])
df['Price'].plot.hist()

plt.xlabel('Price', fontsize=12)
(df['Price'].loc[df['Price']<4.223125e+04 ]).plot.hist()
df['Year'].plot.hist()

plt.xlabel('Car Year', fontsize=12)
df['Popularity'].plot.hist()

plt.xlabel('Popularity of the Car', fontsize=12)

fig, ax=plt.subplots(figsize=(5,5))

ax.scatter(df['Popularity'],df['Price'])

plt.title('Scatter between price and popularity')

ax.set_xlabel=('Popularity')

ax.st_ylabel=('Price')

plt.show()
fig, ax=plt.subplots(figsize=(5,5))

ax.scatter(df['Cylinders'],df['Price'])

plt.title('Scatter between price and cylinders')

ax.set_xlabel=('Cylinders')

ax.st_ylabel=('Price')

plt.show()
fig, ax=plt.subplots(figsize=(5,5))

ax.scatter(df['Cylinders'],df['Price'])

plt.title('Scatter between price and cylinders')

ax.set_xlabel=('Cylinders')

ax.st_ylabel=('Price')

plt.show()
fig, ax=plt.subplots(figsize=(5,5))

ax.scatter(df['MPG-C'],df['Price'])

plt.title('Scatter between MPG-C and Price')

ax.set_xlabel=('MPG-C')

ax.st_ylabel=('Price')

plt.show()


akm = df.select_dtypes(exclude=[np.number]) 

akm


from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()

for i in akm:

  df[i] = label_enc.fit_transform(df[i])

print('Label Encoded Data')

df.head()
dcode=df



dcode["Make"]=dcode["Make"].astype("category")

dcode["Make"]=dcode["Make"].cat.codes



dcode["Model"] = dcode["Model"].astype('category')

dcode["Model"] = dcode["Model"].cat.codes



'''dcode["Year"]=dcode["Year"].astype("category")

dcode["Year"]=dcode["Year"].cat.codes

dcode["HP"]=dcode["HP"].astype("category")

dcode["HP"]=dcode["HP"].cat.codes

dcode["Cylinders"]=dcode["Cylinders"].astype("category")

dcode["Cylinders"]=dcode["Cylinders"].cat.codes

dcode["MPG-H"]=dcode["MPG-H"].astype("category")

dcode["MPG-H"]=dcode["MPG-H"].cat.codes

dcode["MPG-C"]=dcode["MPG-C"].astype("category")

dcode["MPG-C"]=dcode["MPG-C"].cat.codes

dcode["Popularity"]=dcode["Popularity"].astype("category")

dcode["Popularity"]=dcode["Popularity"].cat.codes'''



dcode["Transmission"]=dcode["Transmission"].astype("category")

dcode["Transmission"]=dcode["Transmission"].cat.codes



dcode["Drive Mode"]=dcode["Drive Mode"].astype("category")

dcode["Drive Mode"]=dcode["Drive Mode"].cat.codes



dcode["Vehicle Size"]=dcode["Vehicle Size"].astype("category")

dcode["Vehicle Size"]=dcode["Vehicle Size"].cat.codes



dcode["Vehicle Style"]=dcode["Vehicle Style"].astype("category")

dcode["Vehicle Style"]=dcode["Vehicle Style"].cat.codes
from sklearn import preprocessing

X = np.asarray(dcode[['Make', 'Model', 'Transmission', 'Drive Mode','Vehicle Size','Vehicle Style']])

y = np.asarray(dcode['Price'])

X = preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=44)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from sklearn import linear_model

lm = linear_model.LinearRegression()

model = lm.fit(X_train,y_train)

predictions = lm.predict(X_test)
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train,y_train)

y_pred = model.predict(X_test)

model.score(X_test,y_pred)

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

d_m = DecisionTreeClassifier(random_state = 0)

d_m.fit(X_train,y_train)

y_pred = d_m.predict(X_test)

print("Confusion Matrix:\n\n", confusion_matrix(y_test, y_pred)) 

print ("\nAccuracy : ", accuracy_score(y_test,y_pred)*100)


model.coef_


model.intercept_
model.predict(X_test)
y_pred = model.predict(X_test) 

plt.plot(y_test, y_pred, '.')



# plot a line, a perfit predict would all fall on this line

x = np.linspace(0, 330, 100)

y = x

plt.plot(x, y)

plt.show()
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import math



print('MSE: %.2f' % mean_squared_error(y_test, y_pred))

print('R Squared : %.2f' % r2_score(y_test, y_pred))

print('MAE :%.2f' % mean_absolute_error(y_test, y_pred))

print('RMSE : %.2f' % math.sqrt(mean_squared_error(y_test, y_pred)))