#прочитать перед запуском кода. НЕ смог подключить данные (жалуется, что есть дубликат). 
#Рекомендуется скачать файлы test.csv  и train.csv с соревнования 
# ссылка, где можно скачать. https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=test.csv
# Поместить код в одну папку с файлами. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import LabelEncoder

from IPython.display import display
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
df_train = pd.read_csv('train.csv')
del df_train["MoSold"] #удаляем месяц продажи недвижки. над этим можно поработать конешно.
df_train.columns
df_train=df_train.fillna(0)
df_train
q_hi = df_train.LotArea.quantile(0.97)
#q_low = df_train.LotArea.quantile(0.02)
df_filtered = df_train[df_train["LotArea"] < q_hi]
q_hi = df_filtered.GrLivArea.quantile(0.99)
#q_low = df_train.LotArea.quantile(0.02)
df_filtered = df_filtered[df_filtered["GrLivArea"] < q_hi]
q_hi = df_filtered.GarageArea.quantile(0.99)
#q_low = df_train.LotArea.quantile(0.02)
df_filtered = df_filtered[df_filtered["GarageArea"] < q_hi]
q_hi = df_filtered["WoodDeckSF"].quantile(0.99)
df_filtered = df_filtered[df_filtered["WoodDeckSF"] < q_hi]
q_hi = df_filtered["OpenPorchSF"].quantile(0.993)
df_filtered = df_filtered[df_filtered["OpenPorchSF"] < q_hi]
df_filtered = df_filtered.reset_index()
data_le=df_filtered.copy()
le = LabelEncoder() 

# Catg_colmns = set(df_train.columns) - set(df_train.select_dtypes(exclude=['object']).columns)
# Catg_colmns

Catg_colmns = df_train.select_dtypes(include=['object']).columns
Catg_colmns
for col in data_le[Catg_colmns]:
    data_le[col] = data_le[col].astype('category')
    data_le[col] = data_le[col].cat.codes
data_le
# data_le['MSZoning'] = data_le['MSZoning'].astype('category')
# # data_le['cut'] = data_le['cut'].cat.reorder_categories(['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'], ordered=True)
# data_le['MSZoning'] = data_le['MSZoning'].cat.codes

# data_le['Street'] = data_le['Street'].astype('category')
# data_le['Street'] = data_le['Street'].cat.codes

# data_le['Alley'] = data_le['Alley'].astype('category')
# data_le['Alley'] = data_le['Alley'].cat.codes
# data_le['LotShape'] = data_le['LotShape'].astype('category')
# data_le['LotShape'] = data_le['LotShape'].cat.codes

X = data_le[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'YrSold', 'SaleType',
       'SaleCondition']]
y = data_le["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
model = RandomForestRegressor(n_jobs=-1, random_state=21, n_estimators=500)#, max_features=20, min_samples_leaf=5, max_depth=15, criterion='mae', bootstrap=True) 

model.fit(X_train, y_train);
# Use the forest's predict method on the test data
predictions = model.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
len(predictions)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.head()
test_data= pd.read_csv('test.csv')
del test_data["MoSold"] #удаляем месяц продажи недвижки. над этим можно поработать конешно.
test_data=test_data.fillna(0)
test_data.head(10)
data_le_test = test_data.copy()
#Catg_colmns = test_data.select_dtypes(include=['object']).columns

for col in data_le_test[Catg_colmns]:
    data_le_test[col] = data_le_test[col].astype('category')
    data_le_test[col] = data_le_test[col].cat.codes
data_le_test.head()
X_test = data_le_test[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'YrSold', 'SaleType',
       'SaleCondition']]

print('Testing Features Shape:', X_test.shape)
test_predictions = model.predict(X_test)
type(test_predictions)
test_data['SalePrice'] = test_predictions
#test_data
test_data[["Id", "SalePrice"]]
test_data[["Id", "SalePrice"]].to_csv('1st_try.csv', index=False)
# # Import tools needed for visualization
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = rf.estimators_[5]
# # Import tools needed for visualization
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = rf.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')
# q_hi = df_train.BsmtFinSF2.quantile(0.97)
# #q_low = df_train.BsmtFinSF1.quantile(0.02)
# df_filtered = df_filtered[df_filtered["BsmtFinSF2"] < q_hi]
#plt.hist(df_filtered["ScreenPorch"])