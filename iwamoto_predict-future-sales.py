import numpy as np

import pandas as pd

import sklearn

import xgboost as xgb



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.externals import joblib



import re

from itertools import product

import gc



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
# Check library versions

for p in [np, pd, sklearn, xgb]:

    print (p.__name__, p.__version__)
# Load data from csv files

salesTrain = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32', 'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})

test = pd.read_csv('../input/competitive-data-science-predict-future-sales//test.csv', dtype={'ID': 'int32', 'shop_id': 'int32', 'item_id': 'int32'})

items = pd.read_csv('../input/competitive-data-science-predict-future-sales//items.csv', dtype={'item_name': 'str', 'item_id': 'int32', 'item_category_id': 'int32'})

itemCategories = pd.read_csv('../input/competitive-data-science-predict-future-sales//item_categories.csv', dtype={'item_category_name': 'str', 'item_category_id': 'int32'})

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales//shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
# Join item_category_id to sales_train data

sales = pd.merge(salesTrain, items, on='item_id', how='left')

sales = sales.drop('item_name', axis=1)

sales.head()
# For every month we create a grid from all shops/items combinations from that month

grid = []

for blockNum in sales.date_block_num.unique():

    shopIds = sales.loc[sales.date_block_num == blockNum, 'shop_id'].unique()

    itemIds = sales.loc[sales.date_block_num == blockNum, 'item_id'].unique()

    grid.append(np.array(list(product(*[[blockNum], shopIds, itemIds])), dtype='int32'))

grid = pd.DataFrame(np.vstack(grid), columns=['date_block_num', 'shop_id', 'item_id'], dtype='int32')
grid.head()
# Get aggregated values for (shop_id, item_id, month)

#   The count and average price of sold items in each shop for a month

salesMean = sales.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum', 'item_price':'mean'}).reset_index()

salesMean = pd.merge(grid, salesMean, how='left', on=['date_block_num', 'shop_id', 'item_id']).fillna(0)

salesMean = pd.merge(salesMean, items, how='left', on=['item_id'])

salesMean.columns = ['date_block_num', 'shop_id', 'item_id', 'shop_item_count_sum', 'shop_item_price_mean', 'item_name', 'item_category_id']

salesMean.drop(['item_name'], axis=1, inplace=True)

salesMean.shop_item_count_sum = salesMean.shop_item_count_sum.astype('int32')



salesMean.head(10).T
plt.figure(figsize=(15, 5))

plt.ylim(0,200)

plt.hist(salesMean.shop_item_count_sum, bins=100)

plt.show()
# Clip target values into [0, 20] range

salesMean.shop_item_count_sum = salesMean.shop_item_count_sum.clip(0, 20)



plt.figure(figsize=(15, 5))

plt.ylim(0,100000)

plt.hist(salesMean.shop_item_count_sum, bins=20)

plt.show()
plt.figure(figsize=(15, 5))

plt.ylim(0,10)

# plt.xlim(0, 50000)

plt.hist(salesMean.shop_item_price_mean, bins=100)

plt.show()
# Clip item prices into [0, 40000] range

salesMean.shop_item_price_mean = salesMean.shop_item_price_mean.clip(0, 40000)



plt.figure(figsize=(15, 5))

plt.ylim(0,1000)

plt.hist(salesMean.shop_item_price_mean, bins=20)

plt.show()
def encodeMean(groupColumns, tarnsformColumn, outputColumn):

    gb = salesMean.groupby(groupColumns)

    salesMean[outputColumn + '_mean'] = gb[tarnsformColumn].transform('mean').astype('float32')



def encodeMeanSum(groupColumns, tarnsformColumn, outputColumn):

    gb = salesMean.groupby(groupColumns)

    salesMean[outputColumn + '_mean'] = gb[tarnsformColumn].transform('mean').astype('float32')

    salesMean[outputColumn + '_sum'] = gb[tarnsformColumn].transform('sum').astype('float32')



encodeMean   (['date_block_num', 'shop_id'],          'shop_item_price_mean', 'shop_price')

encodeMeanSum(['date_block_num', 'shop_id'],          'shop_item_count_sum',  'shop_count')



encodeMean   (['date_block_num', 'item_id'],          'shop_item_price_mean', 'item_price')

encodeMeanSum(['date_block_num', 'item_id'],          'shop_item_count_sum',  'item_count')



encodeMean   (['date_block_num', 'item_category_id'], 'shop_item_price_mean', 'category_price')

encodeMeanSum(['date_block_num', 'item_category_id'], 'shop_item_count_sum',  'category_count')



salesMean.head(10).T
lags = [1, 2, 3]



baseColumns = ['date_block_num', 'shop_id', 'item_id', 'item_category_id']

lagColumns = ['shop_item_count_sum', 'shop_item_price_mean', 'shop_price_mean', 'shop_count_mean', 'shop_count_sum', 'item_price_mean',

              'item_count_mean', 'item_count_sum', 'category_price_mean', 'category_count_mean', 'category_count_sum']



def addLags(salesOrigin, salesMerged):

    for lag in lags:

        s = salesOrigin.copy()

        s.date_block_num += lag

        s = s[baseColumns + lagColumns]

        s.columns = baseColumns + [c + '_' + str(lag) for c in lagColumns]

        salesMerged = pd.merge(salesMerged, s, how='left', on=baseColumns)

    return salesMerged
medians = salesMean.median()



def fillOutNan(df):

    for column in df.columns:

        if 'count' in column:

            df[column] = df[column].fillna(0)

        elif 'price' in column:

            c = re.sub(r'_[0-9]+$', "", column)

            df[column] = df[column].fillna(medians[c])

    return df
salesMeanLags = salesMean[baseColumns + ['shop_item_count_sum']]



# Create the lag value of all mean encodings

salesMeanLags = addLags(salesMean, salesMeanLags)



# Remove values having no lag valuses

salesMeanLags = salesMeanLags[salesMeanLags.date_block_num >= max(lags)]



# Fill NaN with zero and median

salesMeanLags = fillOutNan(salesMeanLags)



salesMeanLags.head(10).T
# Show the correlation of all features for shop_item_count_sum

corr = salesMeanLags.corr()

plt.figure(figsize=(25,1))

pc = pd.DataFrame([corr.loc['shop_item_count_sum', :]], columns=corr.index)

sns.heatmap(pc, annot=True)
validBlock = salesMeanLags.date_block_num.max()



xAll = salesMeanLags.loc[:, salesMeanLags.columns != 'shop_item_count_sum']

yAll = salesMeanLags.loc[:, salesMeanLags.columns == 'shop_item_count_sum']



xTrain = xAll.loc[xAll.date_block_num <  validBlock]

xValid = xAll.loc[xAll.date_block_num == validBlock]



yTrain = yAll.loc[xAll.date_block_num <  validBlock]

yValid = yAll.loc[xAll.date_block_num == validBlock]
# Join item_category_id and mean encodings to test data

xTest = pd.merge(test, items, on='item_id', how='left')

xTest = xTest.drop(['ID', 'item_name'], axis=1)

xTest['date_block_num'] = salesMean.date_block_num.max() + 1

xTest.date_block_num = xTest.date_block_num.astype('int32')
xTest = xTest[baseColumns]



# Create the lag value of all mean encodings

xTest = addLags(salesMean[salesMean.date_block_num > salesMean.date_block_num.max() - max(lags)], xTest)



# Fill NaN with zero and median

xTest = fillOutNan(xTest)



xTest.head(10).T
# Check the columns of test data

assert sum(xTrain.columns != xTest.columns) == 0
# If true, eanble Grid Searh

gridSearch = False



# Prediction or Train/Valid

prediction = True
del salesTrain

del items

del itemCategories

del shops

del sales

del grid

del salesMean

del salesMeanLags



# del gb

del corr

del pc



if not prediction:

    del xTrain

    del xValid

    del yTrain

    del yValid



# garbage collect

gc.collect()
if not prediction:

    lr = LinearRegression()

    lr.fit(xTrain.values, yTrain.values)



    predTrainLr = lr.predict(xTrain.values)



    print('Train R-squared for LinearRegression is %f' % r2_score(yTrain, predTrainLr))

    print('Train Mean Squared Error for LinearRegression is %f' % np.sqrt(mean_squared_error(yTrain, predTrainLr)))



    predValidLr = lr.predict(xValid.values)



    print('Valid R-squared for LinearRegression is %f' % r2_score(yValid, predValidLr))

    print('Valid Mean Squared Error for LinearRegression is %f' % np.sqrt(mean_squared_error(yValid, predValidLr)))



    # show coeficients

    plt.figure(figsize=(15,3))

    plt.bar(np.arange(lr.coef_.shape[1]), lr.coef_[0], tick_label=xTrain.columns)

    plt.xticks(rotation='vertical')

    plt.show()
if prediction:

    lr = LinearRegression()

    lr.fit(xAll.values, yAll)



    predAllLr = lr.predict(xAll.values)



    print('All R-squared for LinearRegression is %f' % r2_score(yAll, predAllLr))

    print('All Mean Squared Error for LinearRegression is %f' % np.sqrt(mean_squared_error(yAll, predAllLr)))



    predTestLr = lr.predict(xTest.values)
# show coeficients

plt.figure(figsize=(15,3))

plt.bar(np.arange(lr.coef_.shape[1]), lr.coef_[0], tick_label=xTest.columns)

plt.xticks(rotation='vertical')

plt.show()
params = {'num_round': 100,

          'eta': 0.3,

          'seed': 123,

          'silent': 1,

          'eval_metric': 'rmse'}
if gridSearch:

    gridParams = {'max_depth': [5, 8, 10], 

                  'min_child_weight': [0.5, 0.75, 1],

                  'subsample': [0.5, 0.75, 1]}



    cvCount = 30000 # xValid.shape[0]

    x = xValid.values[: cvCount]

    y = yValid.values[: cvCount, 0]



    gs = GridSearchCV(xgb.XGBClassifier(**params), gridParams, cv=5)

    gs.fit(x, y)



    print('The best score is %f' % gs.best_score_)

    print('The best parameters are %s' % gs.best_params_)



    params.update(gs.best_params_)

else:

    # Use parameters which are searched on the latest calculation

    params.update({'max_depth': 10,

                   'min_child_weight': 0.5,

                   'subsample': 0.5})
if not prediction:

    bst = xgb.XGBClassifier(**params)

    bst.fit(xTrain.values, yTrain.values[:, 0])



    predTrainXgb = bst.predict(xTrain.values)

    

    print('Train R-squared for XGB is %f' % r2_score(yTrain, predTrainXgb))

    print('Train Mean Squared Error for XGB is %f' % np.sqrt(mean_squared_error(yTrain, predTrainXgb)))



    predValidXgb = bst.predict(xValid.values)



    print('Valid R-squared for XGB is %f' % r2_score(yValid, predValidXgb))

    print('Valid Mean Squared Error for XGB is %f' % np.sqrt(mean_squared_error(yValid, predValidXgb)))



    # Show feature importances

    bst.get_fscore()

    mapper = {'f{0}'.format(i): v for i, v in enumerate(xTrain.columns)}

    mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}

    fig, ax = plt.subplots(figsize=(10, 15))

    xgb.plot_importance(mapped, ax=ax)

    plt.show()
if prediction:

    dAll = xgb.DMatrix(xAll.values, label = yAll.values)

    dTest = xgb.DMatrix(xTest.values)

    

#     bst = xgb.train(params, dAll)

    

    # Save model to file

#     joblib.dump(bst, "xgb.dat")

    

    # Load model to file

    bst = joblib.load("../input/xgbdat/xgb.dat")

    

    predAllXgb = bst.predict(dAll)

    

    print('All R-squared for XGB is %f' % r2_score(yAll, predAllXgb))

    print('All Mean Squared Error for XGB is %f' % np.sqrt(mean_squared_error(yAll, predAllXgb)))



    predTestXgb = bst.predict(dTest)
# Show feature importances

bst.get_fscore()

mapper = {'f{0}'.format(i): v for i, v in enumerate(xTrain.columns)}

mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}

fig, ax = plt.subplots(figsize=(10, 15))

xgb.plot_importance(mapped, ax=ax)

plt.show()
if not prediction:

    predTrainLv1 = pd.DataFrame()

    predTrainLv1['lr']  = predTrainLr[:,0]

    predTrainLv1['xgb'] = predTrainXgb



    predValidLv1 = pd.DataFrame()

    predValidLv1['lr']  = predValidLr[:,0]

    predValidLv1['xgb'] = predValidXgb

    

    predTrainLv1.head(20).T

else:

    predAllLv1 = pd.DataFrame()

    predAllLv1['lr']  = predAllLr[:,0]

    predAllLv1['xgb'] = predAllXgb



    predTestLv1 = pd.DataFrame()

    predTestLv1['lr']  = predTestLr[:,0]

    predTestLv1['xgb'] = predTestXgb



    predTestLv1.head(20).T
lrLv2 = LinearRegression()



if not prediction:

    lrLv2.fit(predTrainLv1, yTrain.values)



    predTrainLv2 = lrLv2.predict(predTrainLv1)

    predValidLv2 = lrLv2.predict(predValidLv1)



    print('Train R-squared for Ensembling is %f' % r2_score(yTrain, predTrainLv2))

    print('Train Mean Squared Error for Ensembling is %f' % np.sqrt(mean_squared_error(yTrain, predTrainLv2)))



    print('Valid R-squared for Ensembling is %f' % r2_score(yValid, predValidLv2))

    print('Valid Mean Squared Error for Ensembling is %f' % np.sqrt(mean_squared_error(yValid, predValidLv2)))
if not prediction:

    # Show coefitients

    plt.figure(figsize=(5,3))

    plt.bar(np.arange(lrLv2.coef_.shape[1]), lrLv2.coef_[0], tick_label=predTrainLv1.columns)

    plt.xticks(rotation='vertical')

    plt.show()
if prediction:

    lrLv2.fit(predAllLv1, yAll.values)



    predAllLv2 = lrLv2.predict(predAllLv1)



    print('All R-squared for Ensembling is %f' % r2_score(yAll, predAllLv2))

    print('All Mean Squared Error for Ensembling is %f' % np.sqrt(mean_squared_error(yAll, predAllLv2)))



    predTestLv2 = lrLv2.predict(predTestLv1)
if prediction:

    # Show coefitients

    plt.figure(figsize=(5,3))

    plt.bar(np.arange(lrLv2.coef_.shape[1]), lrLv2.coef_[0], tick_label=predTestLv1.columns)

    plt.xticks(rotation='vertical')

    plt.show()
if prediction:

#     pred = predTestLv2[:,0]

    pred = predTestXgb

    

    pred = pred.clip(0, 20)

    submission = pd.DataFrame({'ID': test.index, 'item_cnt_month': pred})

    submission.to_csv('submission.csv',index=False)



    print(submission.head())
if prediction:

    print(submission.describe())