%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
input_file = "../input/train.csv"



df_train=pd.read_csv(input_file)

df_train.head()
df_train.shape
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df_train = df_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','Id'], axis=1)
df_train.shape
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] =16.0

fig_size[1] = 4.0



x =df_train['SalePrice']

plt.hist(x, normed=True, bins=400)

plt.ylabel('SalePrice');
def reject_outliers(SalePrice):

    filtered= [e for e in (df_train['SalePrice']) if (e < 500000)]

    return filtered



fig_size = plt.rcParams["figure.figsize"]

fig_size[0] =16.0

fig_size[1] = 4.0



filtered = reject_outliers('SalePrice')

plt.hist(filtered, 50)

fig_size[0]=16.0

fig_size[1]=8.0

plt.show()



df_no_outliers = pd.DataFrame(filtered)

df_no_outliers.shape
df_train = df_train[df_train['SalePrice']<500000]
df_train.head()
X_train = df_train.drop(['SalePrice'], axis=1)
X_train.shape
Y_labels = df_train['SalePrice']
Y_labels.shape
X_train.info()
cat_values= ['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']

len(cat_values)
X_train = X_train.apply(lambda x:x.fillna(x.value_counts().index[0]))

X_train = X_train.fillna(X_train['GarageFinish'].value_counts().index[0]) #fill NaNs with the most frequent value from that column.

X_train = X_train.fillna(X_train['BsmtQual'].value_counts().index[0])

X_train = X_train.fillna(X_train['GarageType'].value_counts().index[0])

X_train = X_train.fillna(X_train['GarageQual'].value_counts().index[0])

X_train = X_train.fillna(X_train['GarageCond'].value_counts().index[0])

X_train = X_train.fillna(X_train['BsmtCond'].value_counts().index[0])

X_train = X_train.fillna(X_train['BsmtExposure'].value_counts().index[0])

X_train = X_train.fillna(X_train['BsmtFinType1'].value_counts().index[0])

X_train = X_train.fillna(X_train['FireplaceQu'].value_counts().index[0])
X_train = pd.get_dummies(X_train, columns=['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])     
X_train.describe()
X_train = X_train.drop(['Condition2_RRAe','Exterior2nd_Other','Condition2_RRAn','Condition2_RRNn','HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc','Heating_Floor','Heating_OthW','Electrical_Mix','GarageQual_Ex', 'Exterior1st_Stone','Utilities_NoSeWa'], axis=1)
X_train.shape
input_file = "../input/test.csv"



df_test=pd.read_csv(input_file)
df_test.head()
df_test.shape
total = df_test.isnull().sum().sort_values(ascending=False)

percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df_test = df_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
df_test.shape
df_test = df_test.apply(lambda x:x.fillna(x.value_counts().index[0])) #= fills every column with its own most frequent value



df_test = df_test.fillna(df_test['GarageFinish'].value_counts().index[0]) #fill NaNs with the most frequent value from that column.

df_test = df_test.fillna(df_test['BsmtQual'].value_counts().index[0])

df_test = df_test.fillna(df_test['FireplaceQu'].value_counts().index[0])

df_test = df_test.fillna(df_test['GarageType'].value_counts().index[0])

df_test = df_test.fillna(df_test['GarageQual'].value_counts().index[0])

df_test = df_test.fillna(df_test['GarageCond'].value_counts().index[0])

df_test = df_test.fillna(df_test['GarageFinish'].value_counts().index[0])

df_test = df_test.fillna(df_test['BsmtCond'].value_counts().index[0])

df_test = df_test.fillna(df_test['BsmtExposure'].value_counts().index[0])

df_test = df_test.fillna(df_test['BsmtFinType1'].value_counts().index[0])

df_test = df_test.fillna(df_test['BsmtFinType2'].value_counts().index[0])

df_test = df_test.fillna(df_test['BsmtUnfSF'].value_counts().index[0])
df_test = pd.get_dummies(df_test, columns=['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])
df_test.head()
X_test = df_test.drop(['Id'], axis=1)
print(X_train.shape,"----",X_test.shape)
from xgboost import XGBRegressor
xgb_clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_clf.fit(X_train, Y_labels)
from sklearn.model_selection import cross_val_score

xgb_clf_cv = cross_val_score(xgb_clf,X_train, Y_labels, cv=10, ) # .911240390855695

print(xgb_clf_cv.mean())## tahmin sonucunu yazacaktır.
xgb_clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)



xgb_clf.fit(X_train, Y_labels)
xgb_predictions_test = xgb_clf.predict(X_test) # shape(1459, 262)



xgb_predictions_test 
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train, Y_labels)

result = model.score(X_train, Y_labels)

print("Accuracy: %.3f%%" % (result*100.0))
submission = pd.DataFrame({

        "Id": df_test["Id"],

        "SalePrice": xgb_predictions_test

    })



submission.to_csv("kaggleXGB_HousePrices.csv", index=False)
import seaborn as sns

corr_matrix = df_train.corr()

f, ax = plt.subplots(figsize=(35, 35))

sns.heatmap(corr_matrix, vmax=1, annot=True, square=True);

plt.show()
corr_matrix = df_train.corr()

top_corr_features = corr_matrix.index[abs(corr_matrix['SalePrice'])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(df_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
print(xgb_clf.feature_importances_)

from matplotlib import pyplot

# plot

f, ax = plt.subplots(figsize=(16, 8))

pyplot.bar(range(len(xgb_clf.feature_importances_)), xgb_clf.feature_importances_)

pyplot.show()
from xgboost import plot_importance

fig_size = plt.rcParams["figure.figsize"] 

fig_size[0]=16.0

fig_size[1]=30.0 # from King County House Prices

plot_importance(xgb_clf)

pyplot.show()