from fastai.tabular import *
path = Path('../input/house-prices-advanced-regression-techniques')

output_path = Path('../working')



df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')



len(df), len(test_df)
df.info()
df.describe()
df.head()
dep_var = 'SalePrice'

procs = [FillMissing, Categorify, Normalize]
cont_names = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr',

 'EnclosedPorch', 'Fireplaces', 'FullBath',

 'GarageYrBlt', 'GrLivArea',

 'HalfBath', 'KitchenAbvGr', 

 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea',

 'OpenPorchSF', 'PoolArea', 'ScreenPorch',

 'TotRmsAbvGrd', 'WoodDeckSF']



cat_names = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

           'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',

           'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 

           'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',

           'Electrical', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',

           'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'BsmtQual', 'KitchenQual']

test = TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(df, path=output_path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(list(range(600,800)))

                           #.split_by_rand_pct(0.2)

                           .label_from_df(cols=dep_var, label_cls=FloatList, log=True)

                           .add_test(test)

                           .databunch())
data.show_batch(rows=10)
max_log_y = np.log(np.max(df[dep_var])*1.2)

y_range = torch.tensor([0, max_log_y], device=defaults.device)
learn = tabular_learner(data, layers=[200,100], y_range=y_range, ps=[0.05, 0.1], metrics=exp_rmspe)
learn.lr_find()

learn.recorder.plot()
learn.fit(80, 1e-2)
learn.recorder.plot_losses(skip_start=100)
learn.recorder.plot_metrics(skip_start=200)
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.exp(predictions.data).numpy().T[0]



sub_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': labels})

sub_df.to_csv(output_path/'submission.csv', index=False)