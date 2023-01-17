import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import matplotlib.pylab as plt
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
items.tail()
categories = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
categories.tail()
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
shops
sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
sales.tail()
sales_test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
sales_test.tail()
print(sum(items.duplicated(['item_name'])))
print(sum(categories.duplicated(['item_category_name'])))
print(sum(shops.duplicated(['shop_name'])))
# We can see that the names of shops 10 and 11 differ only by one letter. It is probably the same shop.  
# Also 0 and 57, 1 and 58, ?39 and 40?
# Let's find out if shops 10,11,0,57,1,58 present in the dataframe for forecasting.
uniq_shops = sales_test['shop_id'].unique()
for shop in list([10,11,0,57,1,58]):
    print(shop, shop in uniq_shops)
new_shop_id = {11: 10, 0: 57, 1: 58}
shops['shop_id'] = shops['shop_id'].apply(lambda x: new_shop_id[x] if x in new_shop_id.keys() else x)
sales['shop_id'] = sales['shop_id'].apply(lambda x: new_shop_id[x] if x in new_shop_id.keys() else x)
sales = pd.merge(sales_test, sales, on = ('shop_id', 'item_id'), how = 'left')
sales.tail()
print(sum(sales.duplicated()))
# drop the duplicate rows from sales
sales = sales.drop_duplicates()
sales.shape
print(sum(sales.duplicated(['ID','date','date_block_num','item_price'])))
print(sum(sales.duplicated(['ID','date','date_block_num','item_cnt_day'])))
sales[sales.duplicated(['ID','date','date_block_num','item_cnt_day'])]
sales[sales.duplicated(['ID','date','date_block_num','item_cnt_day'], keep = 'last')]
# We should think carefully which row should be dropped. Price will help us. But now we will just keep first duplicate and drop later.
sales = sales.drop_duplicates(['date','date_block_num','shop_id','item_id','item_cnt_day'])
sales.shape
sales[sales.duplicated(['ID','date','date_block_num'])]
sales[sales.duplicated(['ID','date','date_block_num'], keep = 'last')]
sales = sales.drop_duplicates(['ID','date','date_block_num'], keep = 'last')
sales.shape
print(items.isnull().sum().sum())
print(categories.isnull().sum().sum())
print(shops.isnull().sum().sum())
print(sales.isnull().sum().sum())
# There are missing values in the data. Most of them corresponds to IDs from the forcast set that doesn't represent in training set.
sales.describe()
# It is possible that item_price and item_cnt_day has outliers (max >> 0.75-quantile), and item_cnt_day has wrong values (min < 0)
sum(sales.item_cnt_day < 0)
# change a sign of negative values
sales.loc[sales.item_cnt_day < 0, 'item_cnt_day'] = -1. * sales.loc[sales.item_cnt_day < 0, 'item_cnt_day']
ax = sns.boxplot(x = sales.item_cnt_day)
ax = sns.boxplot(x = sales.loc[sales.item_id == sales.item_id[sales.item_cnt_day.idxmax()],'item_cnt_day'])
ax = sns.boxplot(x = sales.item_price)
sales.loc[sales.item_price > 35000,'item_id'].unique()
items.loc[items.item_id == 13403,:]
sales_month = sales.sort_values('date_block_num').groupby(['ID', 'date_block_num'], as_index = False).agg({'item_cnt_day': ['sum'], 'item_price': ['mean']})
sales_month.columns = ['ID', 'date_block_num', 'item_cnt_month', 'item_price']
sales_month.sample(10)
# after we grouped and aggregate data we delete all rows corresponding to IDs that don't present in train data set (and preset just in forcasting set)
sales_month.describe()
sns.distplot(sales_month.loc[:,'item_cnt_month'])
sns.distplot(sales_month.loc[:,'item_price'], kde=False)
plt.scatter(sales_month.item_cnt_month, sales_month.item_price)
ax = plt.subplots(figsize=(25, 12))
ax = sns.heatmap(sales.pivot_table(index = 'date_block_num', columns = 'shop_id', values = 'item_cnt_day', aggfunc = 'sum'), cmap="YlGnBu")
plt.title("Items sold by each shop per month")
plt.show()
ax = plt.subplots(figsize=(15, 5))
plt.plot(sales_month.groupby(['date_block_num'], as_index = False).agg({'ID':'count'}).iloc[:,1], 'o')
plt.title("Number of the unique IDs over months")
plt.xlabel("date_block_num")
plt.show()
# We have over 100,000 unique IDs but less then 30,000 of them was sold in any month
ax = plt.subplots(figsize=(25, 5))
plt.plot(sales_month.groupby(['ID'], as_index = False).agg({'date_block_num':'count'}).iloc[:,1], '.')
plt.title("Number of the month in which each ID was sold")
plt.xlabel("unique ID")
plt.show()
# The most of ID has information about sales less then for 10 months.
def to_IDs(np_data, col_ID):
    # np_data - sales converted to numpy array
    # col_ID - name of ID column
    sales_by_ID = list() #dict()
    IDs = np.unique(np_data[:,col_ID])
    #IDs = np_data[col_ID].unique()
    for i in IDs:
        positions = np_data[:,col_ID] == i
        sales_ID = np_data[positions,:]
        #positions = np_data[col_ID] == i
        #sales_by_ID[i] = np_data.loc[positions,:]
        sales_by_ID.append(sales_ID)
    return sales_by_ID
sales_by_ID = to_IDs(sales_month.values,0)
#sales_by_ID = to_IDs(sales_month,'ID')
print(len(sales_by_ID))
# to decrease calculation time during a code debugging we remove IDs that don't have observtions for last 6 months
def remove_ID_nan_last_year(np_data):
    N_IDs = len(np_data)
    col_date = 1
    clear_data = list()
    cut_month = 33 - 6
    for i in range(N_IDs):
        ID_data = np_data[i]
        if len(ID_data[ID_data[:,col_date] >= cut_month,2]) != 0:
            clear_data.append(ID_data)
    return clear_data
#sales_by_ID = remove_ID_nan_last_year(sales_by_ID)
#len(sales_by_ID)
#def split_train_test(np_data, col_date = 'date_block_num', last_month = 33):
def split_train_test(np_data, col_date = 1, last_month = 33):
    col_TS = 2 # numbe of item_cnt_month column
    N_IDs = len(np_data)
    train = list()
    test = list()
    empty_train, empty_test = 0, 0 # we will count train and test sets that have zero length
    #for ID_data in np_data.values():
    for i in range(N_IDs):
        ID_data = np.array(np_data[i])
        # ID_data = np.array(ID_data)
        train_rows = ID_data[ID_data[:,col_date] < last_month, :]
        test_response = ID_data[ID_data[:,col_date] >= last_month, col_TS]
        #train_rows = ID_data.loc[ID_data[col_date] < last_month, :]
        #test_rows = ID_data.loc[ID_data[col_date] >= last_month, :]
        if len(train_rows) == 0: #or (len(train_rows) == 1 and len(test_response) == 0):
            empty_train = 1 + empty_train
            continue
        if len(test_response) == 0:
            empty_test = 1 + empty_test
            test.append(np.nan)
        else:
            test.append(test_response[0])
        train.append(train_rows)
    print(empty_train,' IDs do not have observations in TRAIN set')
    print(empty_test,' IDs do not have observations in TEST set')
    return train, np.array(test)
train, test_actual = split_train_test(sales_by_ID, 1, 33)
print(len(test_actual), 'IDs will be used for modeling')
print(len(train))
# We have a lot of IDs that don't have observations for last month, 
# so these are useless for a metric calculating but we keep it for modeling.
test_actual = np.nan_to_num(test_actual, nan = 0)
# Let's fill the missing date_block_num by NaN for paticular ID
def missing_months(np_data, col_date, col_TS, N_months = 33):
    # col_date - index of date_block_num column
    # col_TS - index of item_price column and item_cnt_month column
    # at first fill time series by NaN for all months
    series = [np.nan for _ in range(N_months)]
    for i in range(len(np_data)):
        position = int(np_data[i, col_date] - 1)
        # fill positions that present in data
        series[position] = np_data[i, col_TS]
    return series
# Plot time series for particular ID to find out missing months
def plot_TS(np_data, n_vars = 1, N_months = 33, flag = 0):
    # n_vars = 1 or 2 (plot item_cnt OR item_cnt and item_price)
    plt.figure()
    if flag == 1:
        TSs = to_fill_missing(np_data, N_months)
    for i in range(n_vars):
        col_plot = i + 2 # index of column to plot
        if flag == 1:
            series = TSs[:,col_plot]
        else:
            series = missing_months(np_data, 1, col_plot, N_months)
        ax = plt.subplot(n_vars, 1, i+1)
        plt.plot(series, 'o')
        plt.plot(series)
    plt.show()
plot_TS(train[10],2,33, flag = 0)
# in the up plot there is represented the item_cnt_month
# in the low -- item_price for ID=11
# We can use an interpolation to fill missing value for some IDs but not for all
plot_TS(train[1563],2,33)
# In this case we can fill the data by 0 or 1
plt.scatter(train[1563][:,2], train[1563][:,3])
plot_TS(train[80059],2,33)
# In this case it seems resonable to fill by 0 (because of high price)
plot_TS(train[30111],2,33)
# Let's fill the missing item_cnt_month and item_price for particular ID
def to_fill_missing(np_data, N_months = 33):
    date_month = pd.DataFrame(range(N_months),columns = ['date_block_num'])
    col = ['ID','date_block_num','item_cnt_month','item_price']
    sales_ID = pd.DataFrame(np_data, columns = col)
    #sales_ID['missing_flag'] = 1
    if sales_ID.shape[0] < N_months:
        sales_ID = pd.merge(date_month, sales_ID, on = ('date_block_num'), how = 'left')
        sales_ID = sales_ID.reindex(columns = col)
     #   sales_ID['missing_flag'] = sales_ID['item_cnt_month'].isnull().astype('uint')
        sales_ID['ID'] = sales_ID['ID'].fillna(sales_ID['ID'].loc[sales_ID['ID'].first_valid_index()])
        sales_ID['item_cnt_month'] = sales_ID['item_cnt_month'].fillna(0)
        sales_ID['item_price'] = sales_ID['item_price'].interpolate(method ='linear', limit_direction ='both')
    return sales_ID.to_numpy()
plot_TS(train[30111],2,33,flag = 0)
plot_TS(train[30111],2,33,flag = 1)
plt.scatter(train[30111][:,2], train[30111][:,3])
plot_TS(train[5127],2,33,flag = 0)
plot_TS(train[5127],2,33,flag = 1)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
def plot_acf_pacf(np_data, n_vars = 1, N_months = 33):
    # n_vars = 1 or 2 (plot item_cnt OR item_cnt and item_price)
    plt.figure()
    col_plot = 2 # index of column to plot
    for i in range(1, 2*n_vars, 2): 
        series = to_fill_missing(np_data, N_months)
        series = series[:,col_plot]
        ax = plt.subplot(n_vars, 2, i)
        plot_acf(series, ax = ax)
        ax = plt.subplot(n_vars, 2, i+1)
        plot_pacf(series, ax = ax)
        col_plot = col_plot + 1
    plt.show()
plot_acf_pacf(train[61559],1,33)
plot_acf_pacf(train[30112])
# Let's modify the data of particular ID
def to_make_lag_features(np_data, n_lag = 2, N_months=33):
    # in_out = np.empty((N_months-n_lag, 2*n_lag+1))
    ID_TS = to_fill_missing(np_data, N_months)
    N_col = ID_TS.shape[1]
    in_out = np.empty((N_months-n_lag, n_lag+N_col))
    count_TS = ID_TS[:,2]
    for i in range(n_lag, N_months):
        # input features: n_lags of item_cnt_month and n_lags of item_price
        # output: item_cnt_month
        # in_out[i-n_lag,:] = np.concatenate([count_TS[i-n_lag:i], np.array([count_TS[i]])]) # price_TS[i-n_lag:i],  np.array([count_TS[i]])])
        in_out[i-n_lag,:] = np.concatenate([np.delete(ID_TS[i,:],2,axis=0), count_TS[i-n_lag:i],  np.array([count_TS[i]])])
    # the last array contains n_lags of item_cnt_month and n_lags of item_price that will be features for prediction of 34th month
    test_df = np.concatenate([np.array([ID_TS[i,0], N_months, ID_TS[i,3]]), count_TS[i+1-n_lag:i+1]])
    # test_df = np.concatenate([count_TS[i+1-n_lag:i+1], price_TS[i+1-n_lag:i+1]])
    # test_df = count_TS[i+1-n_lag:i+1]
    return in_out, test_df
def data_preparation(data_IDs, N_months = 33, n_lag = 2):
    # data_IDs - list of train data, each element of list contains train data for particular ID
    N_IDs = len(data_IDs)
    N_col = data_IDs[0].shape[1]
    N_rows = N_months-n_lag
    train_data = np.empty((N_IDs*N_rows, n_lag+N_col)) # 2*n_lag+1
    test_data = np.empty((N_IDs, n_lag+N_col-1)) # 2*n_lag
    list_IDs = np.empty((1,N_IDs))
    # col_TS - index of item_cnt_month column
    col_TS = 2
    for i in range(N_IDs): # each particular ID
        ID_data = data_IDs[i]
        train, test = to_make_lag_features(ID_data, n_lag, N_months)
        train_data[i*N_rows:(i+1)*N_rows,:] = train
        test_data[i,:] = test
       # list_IDs[0,i] = ID_data[0,0]
    return np.array(train_data), np.array(test_data)#, list_IDs
# Let's fit model for item_cnt_month of all IDs
def to_fit_model(model, train_df):
    features, label = train_df.drop('item_cnt_month', axis = 1), train_df.loc[:,'item_cnt_month']
    model.fit(features, label)
    return model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
n_lag = 2
tr, ts = data_preparation(train, n_lag = n_lag)
train_features_label = pd.DataFrame(tr, columns = ['ID', 'date_block_num', 'item_price', 
                                                   'item_cnt_month_lag-2', 'item_cnt_month_lag-1', 'item_cnt_month'])
test_features = pd.DataFrame(ts, columns = ['ID', 'date_block_num', 'item_price', 
                                                   'item_cnt_month_lag-2', 'item_cnt_month_lag-1'])
train_features_label.head()
train_features_label = pd.merge(train_features_label, sales_test.loc[:,['ID', 'shop_id', 'item_id']], on = ('ID'), how = 'left')
train_features_label = pd.merge(train_features_label, items.loc[:,['item_id', 'item_category_id']], on = ('item_id'), how = 'left')
#aa = aa.drop(aa[(aa['item_cnt_month_lag-2'] == 0) & (aa['item_cnt_month_lag-1'] == 0) & (aa['item_cnt_month'] == 0)].index)
train_features_label = train_features_label.astype({'ID': 'uint32', 'date_block_num': 'uint8', 'item_cnt_month_lag-2': 'uint16', 'item_cnt_month_lag-1': 'uint16',        'item_cnt_month': 'uint16'})
test_features = pd.merge(test_features, sales_test.loc[:,['ID', 'shop_id', 'item_id']], on = ('ID'), how = 'left')
test_features = pd.merge(test_features, items.loc[:,['item_id', 'item_category_id']], on = ('item_id'), how = 'left')
test_features = test_features.astype({'ID': 'uint32', 'date_block_num': 'uint8', 'item_cnt_month_lag-2': 'uint16', 'item_cnt_month_lag-1': 'uint16'})
best_random = {'n_estimators': 94, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True}
#{'n_estimators': 1000, 'min_samples_split': 50, 'max_features': 'auto'} # score = 0.6
model = RandomForestRegressor(n_estimators = 500, min_samples_split = 10, criterion = "mse", bootstrap = True, verbose = 1)
fitt = to_fit_model(model, train_features_label)
predictions = fitt.predict(test_features)
RMSE = metrics.mean_squared_error(test_actual, predictions) #to_calculate_RMSE(test_actual, predictions)
print(RMSE)
predictions[range(20)]
test_actual[range(20)]
importance = fitt.feature_importances_
N_features = len(importance)
importance
plt.figure(figsize=(15,5))
plt.title("Feature importance")
bars = plt.bar(range(N_features), importance, align="center")
x = plt.xticks(range(N_features), list(test_features.columns))
# concatenate the columns
features_last_month = test_features.copy()
features_last_month['date_block_num'] = 34
features_last_month['item_cnt_month_lag-2'] = test_features['item_cnt_month_lag-1']
features_last_month['item_cnt_month_lag-1'] = test_actual
features_last_month.head()
last_month_predict = fitt.predict(features_last_month)
submission = pd.DataFrame({
        'ID': features_last_month['ID'],
        'item_cnt_month': last_month_predict
    })
submission.head()
submission = pd.merge(sales_test.ID, submission, on = ('ID'), how = 'left')
submission = submission.fillna(0)
submission.tail()
submission.to_csv('submission.csv', index=False)

