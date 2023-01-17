import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#visualization libraries

import seaborn as sns

import matplotlib.pyplot as plt



#model selection and evaluation

from sklearn.model_selection import train_test_split 

from sklearn.metrics import r2_score





# Model libraries

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from xgboost import XGBRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

data.head()
data.info()
f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(),annot=True, fmt=".2f", linewidths=.5, ax=ax)

plt.show()
plt.subplots(figsize=(7, 5))

sns.countplot(data["bedrooms"])

plt.show()
plt.subplots(figsize=(15, 5))

sns.countplot(data["bathrooms"])

plt.show()
sns.countplot(data["floors"])

plt.show()
sns.countplot(data["waterfront"])

plt.show()
sns.countplot(data["view"])

plt.show()
sns.countplot(data["condition"])

plt.show()
plt.subplots(figsize=(15, 5))

sns.countplot(data["grade"])

plt.show()
fig, ax= plt.subplots(figsize=(27,30), ncols=3, nrows=6)

sns.scatterplot(x="bedrooms", y="price",data=data, ax=ax[0][0])

sns.scatterplot(x="bathrooms", y="price",data=data, ax=ax[0][1])

sns.scatterplot(x="sqft_living", y="price",data=data, ax=ax[0][2])

sns.scatterplot(x="sqft_lot", y="price",data=data, ax=ax[1][0])

sns.scatterplot(x="floors", y="price",data=data, ax=ax[1][1])

sns.scatterplot(x="waterfront", y="price",data=data, ax=ax[1][2])

sns.scatterplot(x="view", y="price",data=data, ax=ax[2][0])

sns.scatterplot(x="condition", y="price",data=data, ax=ax[2][1])

sns.scatterplot(x="grade", y="price",data=data, ax=ax[2][2])

sns.scatterplot(x="sqft_above", y="price",data=data, ax=ax[3][0])

sns.scatterplot(x="sqft_basement", y="price",data=data, ax=ax[3][1])

sns.scatterplot(x="yr_built", y="price",data=data, ax=ax[3][2])

sns.scatterplot(x="yr_renovated", y="price",data=data, ax=ax[4][0])

sns.scatterplot(x="zipcode", y="price",data=data, ax=ax[4][1])

sns.scatterplot(x="lat", y="price",data=data, ax=ax[4][2])

sns.scatterplot(x="long", y="price",data=data, ax=ax[5][0])

sns.scatterplot(x="sqft_living15", y="price",data=data, ax=ax[5][1])

sns.scatterplot(x="sqft_lot15", y="price",data=data, ax=ax[5][2])

plt.show();
model = []

score = []

x_train, x_test, y_train, y_test = train_test_split(data.drop(["id","date","price","zipcode"],axis=1),data["price"],test_size=0.2,random_state=42)

print("X Train Shape", x_train.shape)

print("Y Train Shape", y_train.shape)

print("X Test Shape", x_test.shape)

print("Y Test Shape", y_test.shape)
linear_model = LinearRegression()

linear_model.fit(x_train,y_train)

linear_model_predict = linear_model.predict(x_test)

print("Score: ",r2_score(linear_model_predict,y_test))

model.append("Multi Linear Regression")

score.append(r2_score(linear_model_predict,y_test))
ridge_model = Ridge()

ridge_model.fit(x_train,y_train)

ridge_model_predict = ridge_model.predict(x_test)

print("Score: ",r2_score(ridge_model_predict,y_test))

model.append("Ridge Regression")

score.append(r2_score(ridge_model_predict,y_test))
lasso_model = Lasso()

lasso_model.fit(x_train,y_train)

lasso_model_predict = lasso_model.predict(x_test)

print("Score: ",r2_score(lasso_model_predict,y_test))

model.append("Lasso Regression")

score.append(r2_score(lasso_model_predict,y_test))
elasticnet_model = ElasticNet()

elasticnet_model.fit(x_train,y_train)

elasticnet_model_predict = elasticnet_model.predict(x_test)

print("Score: ",r2_score(elasticnet_model_predict,y_test))

model.append("Elastic Net Regression")

score.append(r2_score(elasticnet_model_predict,y_test))
tree_reg = DecisionTreeRegressor()

tree_reg.fit(x_train,y_train)

tree_reg_predict = tree_reg.predict(x_test)

print("Score: ",r2_score(tree_reg_predict,y_test))

model.append("Decision Tree Regression")

score.append(r2_score(tree_reg_predict,y_test))
reg = RandomForestRegressor(n_estimators=100, random_state = 42)

reg.fit(x_train,y_train)

reg_predict = reg.predict(x_test)

print("Score: ",r2_score(reg_predict,y_test))

model.append("Random Forest Regression")

score.append(r2_score(reg_predict,y_test))
reg_ada = AdaBoostRegressor(random_state=0, n_estimators=5)

reg_ada.fit(x_train,y_train)

reg_ada_predict = reg_ada.predict(x_test)

print("Score: ",r2_score(reg_ada_predict,y_test))

model.append("Ada Boost Regression")

score.append(r2_score(reg_ada_predict,y_test))
reg_gb = GradientBoostingRegressor()

reg_gb.fit(x_train,y_train)

reg_gb_predict = reg_gb.predict(x_test)

print("Score: ",r2_score(reg_gb_predict,y_test))

model.append("Gradient Boosting Regression")

score.append(r2_score(reg_gb_predict,y_test))
reg_xgb = XGBRegressor()

reg_xgb.fit(x_train,y_train)

reg_xgb_predict = reg_xgb.predict(x_test)

print("Score: ",r2_score(reg_xgb_predict,y_test))

model.append("XGBoost Regression")

score.append(r2_score(reg_xgb_predict,y_test))
plt.subplots(figsize=(15, 5))

sns.barplot(x=score,y=model,palette = sns.cubehelix_palette(len(score)))

plt.xlabel("Score")

plt.ylabel("Regression")

plt.title('Regression Score')

plt.show()
#set ids as Id and predict survival 

ids = data['id']

predict = reg_xgb.predict(data.drop(["id","date","price","zipcode"],axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'HouseID' : ids, 'Price': predict})

output.to_csv('submission.csv', index=False)