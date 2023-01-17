# linear algebra
import numpy as np 
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

house_sales = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
house_sales.head(4)
#Checking the null value count for each columns. There is no missing values
total = house_sales.isnull().sum().sort_values(ascending=False)
total
house_sales.columns
house_sales.shape
house_sales.describe()
house_sales.info()
house_sales['waterfront'].value_counts()
house_sales['view'].value_counts()
house_sales["yr_renovated"].value_counts()
df_house = house_sales.copy()
df_house = df_house.drop(["id","waterfront","view","sqft_living15", "sqft_lot15"],axis=1)
# Take the renovated years for the age of the building
df_house["year"] = df_house[["yr_built","yr_renovated"]].max(axis=1)
df_house["date"] = df_house["date"].str[:4].astype(int)
df_house.head(4)
# Instead of a year column, I create a new column called age of the built and drop the other unrequired columns
df_house["age"] = df_house["date"] - df_house["year"]
df_house = df_house.drop(["date","yr_built","yr_renovated"], axis=1)
df_house.head(4)
corr = df_house.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corr, vmax=.8, square = True); 
cols = corr.nlargest(15,'price')['price'].index
cm = np.corrcoef(df_house[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize = (12,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.set()
correlated_columns = ["price","sqft_living", "grade", "sqft_above","bathrooms"]
sns.pairplot(df_house[correlated_columns], size = 2.5)
plt.show()
df_house = df_house.drop(["year","zipcode","lat", "long"], axis=1)
df_house.head(4)
X= df_house.iloc[:,1:].values
y= df_house.iloc[:,0].values
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train,y_train)
y_pred= regressor.predict(X_test)
regressor.predict([[3, 2.25, 2570, 7242, 2, 3, 7,2170,400,23]])
df_house.head(4)
import statsmodels.regression.linear_model as lm
#The 0th column contains only 1 in each rows 
X= np.append(arr = np.ones((21613,1)).astype(int), values = X, axis=1) 
X_opt= X[:, [0,1,2,3,4,5,6,7,8,9,10]] 
regressor_OLS=lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
