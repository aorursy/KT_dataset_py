from fastai.tabular import *
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
dep_var = 'SalePrice'

cat_names = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig',

             'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual',

             'OverallCond','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',

             'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'

             ,'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu'

             ,'GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PavedDrive'

             ,'PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(test, cat_names=cat_names, procs=procs)
doc(TabularList.from_df)
data = (TabularList.from_df(train, path='.', cat_names=cat_names, procs=procs)

                        .split_by_rand_pct(valid_pct = 0.1, seed = 42)

                        .label_from_df(cols = dep_var, label_cls = FloatList, log = True )

                        .add_test(test)

                        .databunch())
data.show_batch(rows=5)
learn = tabular_learner(data,layers=[200,100],metrics=accuracy)
learn.fit_one_cycle(15, max_lr=1e-01)
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

row=test.iloc[0]
learn.predict(row)