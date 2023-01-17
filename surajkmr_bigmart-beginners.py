import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sale_train = pd.read_csv("../input/Train.csv")

sale_test = pd.read_csv("../input/Test.csv")

sale_train.head()
sale_train.isnull().sum()
sns.distplot(sale_train["Item_Outlet_Sales"])

print('Skewness: %f' % sale_train['Item_Outlet_Sales'].skew(), ", highly skewed")
sale_train.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean().plot.bar()
sale_train.groupby('Item_Type')['Item_Outlet_Sales'].sum().plot.bar()
sale_train.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().plot.bar()

sale_train.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')
train_id = sale_train.Item_Identifier

test_id = sale_test.Item_Identifier

y_sales = sale_train.Item_Outlet_Sales
sale_train = sale_train.drop(['Item_Outlet_Sales',"Item_Identifier" ], axis = 1)

sale_test  = sale_test.drop(["Item_Identifier"] , axis =1)
combined_data = pd.concat([sale_train, sale_test] , ignore_index = True)

combined_data.sample(5)
sns.countplot(x="Outlet_Size", data= combined_data)
combined_data ["Outlet_Size"] = combined_data["Outlet_Size"].fillna((combined_data["Outlet_Size"].mode()[0] ))

combined_data["Item_Fat_Content"] = combined_data["Item_Fat_Content"].replace({"low fat" :"Low Fat","LF" :"Low Fat", "reg" : "Regular"})

sns.countplot(x="Item_Fat_Content", data= combined_data)
sns.boxplot(x = "Item_Weight", data = combined_data)
combined_data["Item_Weight"] = combined_data["Item_Weight"].fillna((combined_data["Item_Weight"].mean() ))

combined_data.isnull().sum()
combined_data.columns
combined_data = pd.get_dummies(combined_data, columns = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Establishment_Year","Outlet_Size", "Outlet_Location_Type", "Outlet_Type" ], drop_first = True)

combined_data.head()
X_train = combined_data[:len(sale_train)]

X_test = combined_data[len(sale_train):]
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score
trainX, testX, trainY, testY = train_test_split(X_train, y_sales,test_size = 0.2, random_state = 0) 

from  sklearn.preprocessing  import StandardScaler

slc= StandardScaler()

trainX = slc.fit_transform(trainX)

X_test = slc.transform(X_test)

testX = slc.transform(testX)
num_folds = 10

seed = 0

scoring = 'neg_mean_squared_error'

kfold = KFold(n_splits=num_folds, random_state=seed)
model = XGBRegressor(n_estimators=70 , learning_rate = .1)

score_= cross_val_score(model, trainX, trainY, cv=kfold, scoring=scoring)

model.fit(trainX, trainY)

predictions = model.predict(testX)

print(r2_score(testY, predictions))

rmse = np.sqrt(mean_squared_error(testY, predictions))
rmse = np.sqrt(mean_squared_error(testY, predictions))

rmse
y_pred = model.predict(X_test)

my_submission = pd.DataFrame({'Id': test_id, 'SalePrice': y_pred})

my_submission.to_csv('submission_sales.csv', index=False)