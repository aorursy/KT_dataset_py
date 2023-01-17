import pandas as pd
import numpy as np
from sklearn import preprocessing

#data Preprocessing

dataset= pd.read_csv("../input/train.csv", header = 0, delimiter = ",")
dataset_test= pd.read_csv("../input/test.csv", header = 0, delimiter = ",")

#Getting the sum of TotalBsmtSF, 11stFlrSF and 22ndFlrSF values
df1 = dataset[['1stFlrSF','2ndFlrSF','TotalBsmtSF']]
dataset['Total_usable_area'] = df1.sum(axis=1)
df2 = dataset_test[['1stFlrSF','2ndFlrSF','TotalBsmtSF']]
dataset_test['Total_usable_area'] = df1.sum(axis=1)

#Replace NA values with zeros so it is easy when calculating scores for features on train set
dataset['LotFrontage'].fillna(0, inplace=True)
dataset['MasVnrArea'].fillna(0, inplace=True)
dataset['GarageArea'].fillna(0, inplace=True)
#Replace NA values with zeros so it is easy when calculating scores for features on test set
dataset_test['LotFrontage'].fillna(0, inplace=True)
dataset_test['MasVnrArea'].fillna(0, inplace=True)
dataset_test['GarageArea'].fillna(0, inplace=True)

#converting categorical data to numerical data so it will be much easier when selecting features
dataset['LotShape'] = preprocessing.LabelEncoder().fit_transform(dataset['LotShape'].values)
dataset['HouseStyle'] = preprocessing.LabelEncoder().fit_transform(dataset['HouseStyle'].values)
dataset['OverallCond'] = preprocessing.LabelEncoder().fit_transform(dataset['OverallCond'].values)
dataset['MasVnrType'] = preprocessing.LabelEncoder().fit_transform(dataset['OverallCond'].values)
dataset['Foundation'] = preprocessing.LabelEncoder().fit_transform(dataset['Foundation'].values)
dataset['SaleCondition'] = preprocessing.LabelEncoder().fit_transform(dataset['SaleCondition'].values)

dataset_test['LotShape'] = preprocessing.LabelEncoder().fit_transform(dataset_test['LotShape'].values)
dataset_test['HouseStyle'] = preprocessing.LabelEncoder().fit_transform(dataset_test['HouseStyle'].values)
dataset_test['OverallCond'] = preprocessing.LabelEncoder().fit_transform(dataset_test['OverallCond'].values)
dataset_test['MasVnrType'] = preprocessing.LabelEncoder().fit_transform(dataset_test['OverallCond'].values)
dataset_test['Foundation'] = preprocessing.LabelEncoder().fit_transform(dataset_test['Foundation'].values)
dataset_test['SaleCondition'] = preprocessing.LabelEncoder().fit_transform(dataset_test['SaleCondition'].values)

#removing outliers on both train and test sets
ys = dataset['LotFrontage']
ys_test = dataset_test['LotFrontage']
salesoutliers = dataset['SalePrice']
Total_usable_area_outliers = dataset['Total_usable_area']
Total_usable_area_outliers_test = dataset_test['Total_usable_area']

quartile_1, quartile_3 = np.percentile(ys, [25, 75])
iqr = quartile_3 - quartile_1
lower_bound = quartile_1 - (iqr * 1.5)
upper_bound = quartile_3 + (iqr * 1.5)
dataset = dataset.drop(dataset[(dataset.LotFrontage > upper_bound) | (dataset.LotFrontage < lower_bound)].index)

#quartile_1, quartile_3 = np.percentile(ys_test, [25, 75])
#iqr = quartile_3 - quartile_1
#lower_bound = quartile_1 - (iqr * 1.5)
#upper_bound = quartile_3 + (iqr * 1.5)
#dataset_test = dataset_test.drop(dataset_test[(dataset_test.LotFrontage > upper_bound) | (dataset_test.LotFrontage < lower_bound)].index)


quartile_1, quartile_3 = np.percentile(salesoutliers, [25, 75])
iqr = quartile_3 - quartile_1
lower_bound = quartile_1 - (iqr * 1.5)
upper_bound = quartile_3 + (iqr * 1.5)
dataset = dataset.drop(dataset[(dataset.SalePrice > upper_bound) | (dataset.SalePrice < lower_bound)].index)

quartile_1, quartile_3 = np.percentile(Total_usable_area_outliers, [25, 75])
iqr = quartile_3 - quartile_1
lower_bound = quartile_1 - (iqr * 1.5)
upper_bound = quartile_3 + (iqr * 1.5)
dataset = dataset.drop(dataset[(dataset.Total_usable_area > upper_bound) | (dataset.Total_usable_area < lower_bound)].index)

#quartile_1, quartile_3 = np.percentile(Total_usable_area_outliers_test, [25, 75])
#iqr = quartile_3 - quartile_1
#lower_bound = quartile_1 - (iqr * 1.5)
#upper_bound = quartile_3 + (iqr * 1.5)
#dataset_test = dataset_test.drop(dataset_test[(dataset_test.Total_usable_area > upper_bound) | (dataset_test.Total_usable_area < lower_bound)].index)




#dataset.to_csv('test4.csv', sep=',', encoding='utf-8') #Write the dataframe to a csv file.
#Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.datasets import load_svmlight_file
from array import array
from sklearn import linear_model

model = linear_model.LinearRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 5)
rfe = rfe.fit(dataset[['LotShape','HouseStyle','OverallCond','MasVnrType','Foundation','SaleCondition']], dataset['SalePrice'])
# score the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

model = linear_model.LinearRegression()
rfe = RFE(model, 5)
rfe = rfe.fit(dataset[['LotArea','GrLivArea','MasVnrArea','LotFrontage','Total_usable_area','YearBuilt','GarageArea']], dataset['SalePrice'])
# score the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth = 6, splitter= 'best', criterion='mse',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0)
regressor.fit(dataset[['LotShape', 'OverallCond','MasVnrType', 'Foundation', 'SaleCondition', 'GrLivArea', 'LotFrontage','Total_usable_area','YearBuilt','GarageArea']], dataset['SalePrice'])
data = dataset[['LotShape','OverallCond','MasVnrType', 'Foundation', 'SaleCondition', 'GrLivArea', 'LotFrontage','Total_usable_area','YearBuilt','GarageArea','SalePrice']]
data_converted = np.array(data)
data_converted


data


from sklearn.model_selection import train_test_split
x = data_converted[:,0:10]
y = data_converted[:,10:11]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
y_test
x_train
regressor.fit(x_train,y_train)
prediction = regressor.predict(x_test)
prediction
x_test
y_test.shape
prediction.shape
from sklearn.metrics import mean_absolute_error
error=mean_absolute_error(y_test,prediction)
error
z=(error/y_test.mean())
print(100-100*z)
data_for_the_final_prediction = dataset_test[['LotShape', 'OverallCond','MasVnrType', 'Foundation', 'SaleCondition', 'GrLivArea', 'LotFrontage','Total_usable_area','YearBuilt','GarageArea']]
data_converted_final = np.array(data_for_the_final_prediction)
data_converted_final
data_for_the_final_prediction
x_data = data_converted_final[:,0:10]
final_prediction = regressor.predict(x_data)
final_prediction


submission = pd.DataFrame()

submission
submission['Id'] = dataset_test.Id
final_prediction.shape
submission['Id'].shape
dataset_test.shape
submission['SalePrice'] = final_prediction
submission
submission.to_csv('submission.csv',sep=',', encoding='utf-8')
#'LotShape', 'OverallCond','MasVnrType', 'Foundation', 'SaleCondition', 'GrLivArea', 'LotFrontage','Total_usable_area','YearBuilt','GarageArea'

#linear regression model will be used to show the relationship
#matplotlib to visualize data points on a graph
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#LotShape

points = np.array(dataset)  
x_ax = points[:,7:8]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()
#OverallCond
  
x_ax = points[:,18:19]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()
#MasVnrType

x_ax = points[:,25:26]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()

#Foundation

x_ax = points[:,29:30]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()

#SaleCondition

x_ax = points[:,79:80]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()
#GrLivArea

x_ax = points[:,46:47]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()

#LotFrontage

x_ax = points[:,3:4]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()
#Total_usable_area

x_ax = points[:,81:82]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()
#YearBuilt

x_ax = points[:,19:20]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()

#GarageArea

x_ax = points[:,62:63]
y_ax = points[:,80:81]
plt.xscale("linear")
plt.yscale("linear")
plt.scatter(x_ax,y_ax,color='#003F72')
plt.show()
