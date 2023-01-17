import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
data_path = "../input/housing.csv"
df = pd.read_csv(data_path)
df.describe()
df.info()
# In the given datasets we have 9 continuous variables and one categorical variable. ML algorithms do not work well with categorical data. 
# So, we will convert the categorical data. 
df.columns
df.ocean_proximity.value_counts()
sns.countplot(df.ocean_proximity)
new_val = pd.get_dummies(df.ocean_proximity)
new_val.head(5)
df[new_val.columns] = new_val
df.describe()
df.columns
df = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income'
       , '<1H OCEAN', 'INLAND',
       'ISLAND', 'NEAR BAY', 'NEAR OCEAN','median_house_value']]
df.describe() 
# Now, let's understand the correlation between variable by plotting correlation plot
df.corr()
plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), annot=True)
df.corr().sort_values(ascending=False, by = 'median_house_value').median_house_value
df.hist(figsize=(15,12))
df.median_house_value.hist()
sns.distplot(df.median_house_value)
# We can see that the median house value is mostly falls between 10,0000 to 30,0000 with few exceptions. 
# We will need to replace all the null values.
df.isna().sum() 
# So, we have 207 null values. We can drop the rows with null values or we can replace the null values.
# 207 is too big a number to drop rows
df = df.fillna(df.mean())
df.isna().sum() 
from sklearn import preprocessing
convert = preprocessing.StandardScaler() 
df.columns 
feature = df.drop(['median_house_value'], axis=1)
label = df.median_house_value
featureT = convert.fit_transform(feature.values)
labelT = convert.fit_transform(df.median_house_value.values.reshape(-1,1)).flatten() 
featureT
labelT
from sklearn.model_selection import train_test_split
feature_train, feature_test,label_train, label_test = train_test_split(featureT,labelT, test_size=0.2, random_state=19)                                   
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
linear_reg = linear_model.LinearRegression()
linear_reg.fit(feature_train,label_train)
r2_score(linear_reg.predict(feature_train),label_train)
from sklearn.model_selection import cross_val_score
cross_val_score(linear_reg, feature_train,label_train, cv=10) 
reg_score = r2_score(linear_reg.predict(feature_test),label_test) 
reg_score
linear_reg.coef_
pd.DataFrame(linear_reg.coef_, index=feature.columns, columns=['Coefficient']).sort_values(ascending=False, by = 'Coefficient')
df.corr().median_house_value.sort_values(ascending=False) 
ransac_reg = linear_model.RANSACRegressor()
ransac_reg.fit(feature_train,label_train)
r2_score(ransac_reg.predict(feature_train),label_train)
ransac_score = r2_score(ransac_reg.predict(feature_test),label_test)
ransac_score
# Ransac regrssor is performing way poorly than Linear Regresson
ridge_reg = linear_model.Ridge(random_state=19) 
ridge_reg.fit(feature_train,label_train) 
r2_score(ridge_reg.predict(feature_train),label_train)
ridge_score = r2_score(ridge_reg.predict(feature_test),label_test) 
ridge_score
ridge_reg.coef_
pd.DataFrame(ridge_reg.coef_, index=feature.columns, columns=['Coefficient']).sort_values(ascending=False, by = 'Coefficient') 
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(feature_train,label_train)
r2_score(tree_reg.predict(feature_train),label_train)
# 99% seems like overfitting. Let's cross validate it.

cross_val_score(tree_reg, feature_train, label_train, cv=10)

tree_score = r2_score(tree_reg.predict(feature_test),label_test) 
tree_score
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(feature_train,label_train)
r2_score(forest_reg.predict(feature_train),label_train)
cross_val_score(forest_reg, feature_train, label_train, cv=10)
# let's see how well the random forest regressor fits well with the test data
forest_score = r2_score(forest_reg.predict(feature_test),label_test) 
forest_score
# 76% is not a bad score. We can also use GridSearchCV to find the best paramters for random forest regressor
data = [reg_score, ransac_score, ridge_score, tree_score, forest_score]
index = ['Linear Regression', 'Ransac Regression', 'Ridge Regression', 'Decision Tree Regressor', 'Random Forest Regressor']
pd.DataFrame(data, index=index, columns=['Scores']).sort_values(ascending = False, by=['Scores'])
# So, the random forest regressor is winner here out of all the ML Algorithm
