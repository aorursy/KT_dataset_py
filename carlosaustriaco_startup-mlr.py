#importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
#getting the data

df = pd.read_csv("../input/startup-logistic-regression/50_Startups.csv")
#show the first five elements of the dataframe

df.head()
#segregation of numerical columns and categorical columns



cat_col = df.select_dtypes(include=['object']).columns

num_col = df.select_dtypes(exclude=['object']).columns

df_cat = df[cat_col]

df_num = df[num_col]
#Informations about the dataframe

df.info()
#analyzing if there is null values

df_null = df.isna().mean()

df_null.sort_values(ascending = False)
df.dtypes
outliers = ['Profit']

plt.rcParams['figure.figsize'] = [8,8]

sns.boxplot(data=df[outliers], orient="v", palette="Set1" ,whis=1.5,saturation=1, width=0.7)

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Profit Range", fontweight = 'bold')

plt.xlabel("Continuous Variable", fontweight = 'bold')

df.shape
#checking for duplicates



df.loc[df.duplicated()]
#Visualizing the diferent startup locations

plt.rcParams['figure.figsize'] = [5,5]

ax=df['State'].value_counts().plot(kind='bar',stacked=True, colormap = 'Set1', color= "green")

ax.title.set_text('State Locations')

plt.xlabel("Names of the States",fontweight = 'bold')

plt.ylabel("Count of States",fontweight = 'bold')
plt.figure(figsize=(8,8))



plt.title('Startup Profit Distribution Plot')

sns.distplot(df['Profit'])
ax = sns.pairplot(df[num_col])
plt.figure(figsize = (8, 8))

sns.heatmap(df.corr(), cmap="RdYlGn")

plt.show()
plt.figure(figsize=(8, 8))

sns.boxplot(x = 'State', y = 'Profit', data = df)

plt.show()
#Separating the dependent and independent values

X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values
#transforming the categorical value

ct = ColumnTransformer(transformers = [("enconder", OneHotEncoder(), [3])],

                       remainder = "passthrough")

X = np.array(ct.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

np.set_printoptions(precision = 2)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
r2_score(y_pred, y_test)
print("R2 value is " + str(r2_score(y_pred, y_test)))

print("The Mean Squared Error is " + str(mean_squared_error(y_test, y_pred)))
# getting the coeficients of the model

regressor.coef_
#getting the constant of the model

regressor.intercept_
#Visualizing once again the independent variables

X[:5, :]
#Visualizing once again the dataset

df.head()