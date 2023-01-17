# Data Loading Code Runs At This Point

import pandas as pd

    

# Load data

train_path = '../input/home-data-for-ml-course/train.csv'

test_path = '../input/home-data-for-ml-course/test.csv'

train_data = pd.read_csv(train_path, index_col='Id')

test_data = pd.read_csv(test_path)
train_data.columns
from sklearn.metrics import mean_squared_log_error

from sklearn.tree import DecisionTreeRegressor



def get_mse(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mse = mean_squared_log_error(val_y, preds_val)

    return(mse)
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('ggplot')

print("Setup Complete")
#sns.regplot(x=train_data['MSSubClass'], y=train_data['SalePrice'])

sns.lmplot(x="MSSubClass", y="SalePrice", hue="MSZoning", data=train_data)

plt.title("Relationships between MSSubClass and SalePrice")
fig, axes = plt.subplots(1, 4)

fig.set_size_inches(18.5, 5.5)



sns.regplot(x='LotFrontage', y='SalePrice', data=train_data, ax=axes[0])

sns.regplot(x='LotArea', y='SalePrice', data=train_data, ax=axes[1])

sns.swarmplot(x='LotShape', y='SalePrice', data=train_data, ax=axes[2])

sns.swarmplot(x='LotConfig', y='SalePrice', data=train_data, ax=axes[3])





plt.show();
sns.swarmplot(x=train_data['Neighborhood'], y=train_data['SalePrice'])
sns.swarmplot(x=train_data['Condition1'], y=train_data['SalePrice'])
sns.swarmplot(x=train_data['BldgType'], y=train_data['SalePrice'])
sns.swarmplot(x=train_data['HouseStyle'], y=train_data['SalePrice'])
# SalePrice = f(OverallQual, OverallCond) 이런 표 만들어서 plot 하나 보자.
fig, axes = plt.subplots(1, 2)

fig.set_size_inches(18.5, 5.5)

sns.regplot(x='YearBuilt', y='SalePrice', data=train_data, ax=axes[0])

sns.regplot(x='YearRemodAdd', y='SalePrice', data=train_data, ax=axes[1])

plt.show()

# 리모델링 여부, built와 remodel간의 시간간격 등 체크해볼만한게 많을 듯
sns.swarmplot(x=train_data['RoofStyle'], y=train_data['SalePrice'])
fig, axes = plt.subplots(1, 2)

fig.set_size_inches(18.5, 5.5)

sns.swarmplot(x=train_data['Exterior1st'], y=train_data['SalePrice'], ax=axes[0])

sns.swarmplot(x=train_data['Exterior2nd'], y=train_data['SalePrice'], ax=axes[1])



plt.show()
sns.lmplot(x="MasVnrArea", y="SalePrice", hue="MasVnrType", data=train_data)
fig, axes = plt.subplots(1, 2)

fig.set_size_inches(18.5, 5.5)

sns.swarmplot(x=train_data['ExterQual'], y=train_data['SalePrice'], ax=axes[0])

sns.swarmplot(x=train_data['ExterCond'], y=train_data['SalePrice'], ax=axes[1])



plt.show()
sns.swarmplot(x=train_data['Foundation'], y=train_data['SalePrice'])
fig, axes = plt.subplots(1, 3)

fig.set_size_inches(18.5, 5.5)

sns.swarmplot(x=train_data['BsmtQual'], y=train_data['SalePrice'], ax=axes[0])

sns.swarmplot(x=train_data['BsmtCond'], y=train_data['SalePrice'], ax=axes[1])

sns.swarmplot(x=train_data['BsmtExposure'], y=train_data['SalePrice'], ax=axes[2])



plt.show()
sns.lmplot(x="BsmtFinSF1", y="SalePrice", hue="BsmtFinType1", data=train_data)
sns.lmplot(x="BsmtFinSF2", y="SalePrice", hue="BsmtFinType2", data=train_data)
fig, axes = plt.subplots(1, 2)

fig.set_size_inches(18.5, 5.5)

sns.swarmplot(x=train_data['BsmtUnfSF'], y=train_data['SalePrice'], ax=axes[0])

sns.swarmplot(x=train_data['TotalBsmtSF'], y=train_data['SalePrice'], ax=axes[1])



plt.show()
fig, axes = plt.subplots(1, 3)

fig.set_size_inches(18.5, 5.5)

sns.swarmplot(x=train_data['HeatingQC'], y=train_data['SalePrice'], ax=axes[0])

sns.swarmplot(x=train_data['CentralAir'], y=train_data['SalePrice'], ax=axes[1])

sns.swarmplot(x=train_data['Electrical'], y=train_data['SalePrice'], ax=axes[2])

plt.show()
fig, axes = plt.subplots(1, 3)

fig.set_size_inches(18.5, 5.5)

sns.regplot(x='1stFlrSF', y='SalePrice', data=train_data, ax=axes[0])

sns.regplot(x='2ndFlrSF', y='SalePrice', data=train_data, ax=axes[1])

sns.regplot(x='GrLivArea', y='SalePrice', data=train_data, ax=axes[2])

plt.show()
fig, axes = plt.subplots(3, 3)

fig.set_size_inches(18.5, 20.5)

sns.swarmplot(x=train_data['BsmtFullBath'], y=train_data['SalePrice'], ax=axes[0, 0])

sns.swarmplot(x=train_data['BsmtHalfBath'], y=train_data['SalePrice'], ax=axes[0,1])

sns.swarmplot(x=train_data['FullBath'], y=train_data['SalePrice'], ax=axes[0,2])

sns.swarmplot(x=train_data['HalfBath'], y=train_data['SalePrice'], ax=axes[1,0])

sns.swarmplot(x=train_data['BedroomAbvGr'], y=train_data['SalePrice'], ax=axes[1,1])

sns.swarmplot(x=train_data['KitchenAbvGr'], y=train_data['SalePrice'], ax=axes[1,2])

sns.swarmplot(x=train_data['TotRmsAbvGrd'], y=train_data['SalePrice'], ax=axes[2,0])

sns.swarmplot(x=train_data['Fireplaces'], y=train_data['SalePrice'], ax=axes[2,1])

plt.show()
fig, axes = plt.subplots(3,3)

fig.set_size_inches(18.5, 20.5)

sns.swarmplot(x=train_data['KitchenQual'], y=train_data['SalePrice'], ax=axes[0,0])

sns.swarmplot(x=train_data['Functional'], y=train_data['SalePrice'], ax=axes[0,1])

sns.swarmplot(x=train_data['FireplaceQu'], y=train_data['SalePrice'], ax=axes[0,2])

sns.swarmplot(x=train_data['GarageType'], y=train_data['SalePrice'], ax=axes[1,0])

sns.swarmplot(x=train_data['GarageYrBlt'], y=train_data['SalePrice'], ax=axes[1,1])

sns.swarmplot(x=train_data['GarageFinish'], y=train_data['SalePrice'], ax=axes[1,2])



sns.swarmplot(x=train_data['GarageCars'], y=train_data['SalePrice'], ax=axes[2,0])

sns.swarmplot(x=train_data['GarageQual'], y=train_data['SalePrice'], ax=axes[2,1])

sns.swarmplot(x=train_data['GarageCond'], y=train_data['SalePrice'], ax=axes[2,2])

plt.show()
fig, axes = plt.subplots(1,2)

fig.set_size_inches(18.5, 5.5)

sns.regplot(x='GarageArea', y='SalePrice', data=train_data, ax=axes[0])

sns.swarmplot(x='PavedDrive', y='SalePrice', data=train_data, ax=axes[1])

plt.show()
fig, axes = plt.subplots(2,3)

fig.set_size_inches(18.5, 12.0)

sns.regplot(x='WoodDeckSF', y='SalePrice', data=train_data, ax=axes[0,0])

sns.regplot(x='OpenPorchSF', y='SalePrice', data=train_data, ax=axes[0,1])

sns.regplot(x='EnclosedPorch', y='SalePrice', data=train_data, ax=axes[0,2])

sns.regplot(x='3SsnPorch', y='SalePrice', data=train_data, ax=axes[1,0])

sns.regplot(x='ScreenPorch', y='SalePrice', data=train_data, ax=axes[1,1])

plt.show()
fig, axes = plt.subplots(1,3)

fig.set_size_inches(18.5, 5.5)

sns.swarmplot(x='PoolArea', y='SalePrice', data=train_data, ax=axes[0])

sns.swarmplot(x='PoolQC', y='SalePrice', data=train_data, ax=axes[1])

sns.swarmplot(x='Fence', y='SalePrice', data=train_data, ax=axes[2])

plt.show()
sns.lmplot(x="MiscVal", y="SalePrice", hue="MiscFeature", data=train_data)
fig, axes = plt.subplots(1,2)

fig.set_size_inches(18.5, 5.5)

sns.swarmplot(x='MoSold', y='SalePrice', data=train_data, ax=axes[0])

sns.swarmplot(x='YrSold', y='SalePrice', data=train_data, ax=axes[1])

plt.show()
fig, axes = plt.subplots(1,2)

fig.set_size_inches(18.5, 5.5)

sns.swarmplot(x='SaleType', y='SalePrice', data=train_data, ax=axes[0])

sns.swarmplot(x='SaleCondition', y='SalePrice', data=train_data, ax=axes[1])

plt.show()
corrs = train_data.corr()

sns.heatmap(corrs, annot=True, fmt='.2f')

fig = plt.gcf()

fig.set_size_inches(30, 30)

plt.show()