# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

import matplotlib.pyplot as plt

import seaborn as sns



# import scientific computing library

import statsmodels.api as sm



# import xgboost model class

import xgboost as xgb



# import sklearn model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



# import sklearn model evaluation regression metrics

from sklearn.metrics import mean_squared_error
# acquiring training and testing data

df_train = pd.read_csv('../input/train.csv', parse_dates=['date'], index_col='date')

df_test = pd.read_csv('../input/test.csv', parse_dates=['date'], index_col='date')
# visualize head of the training data

df_train.head(n=5)
# visualize tail of the testing data

df_test.tail(n=5)
# combine training and testing dataframe

df_train['datatype'], df_test['datatype'] = 'training', 'testing'

df_train.insert(0, 'id', 0)

df_test.insert(df_test.shape[1] - 1, 'sales', np.nan)

df_data = pd.concat([df_train, df_test], ignore_index=False)
def scatterplot(numerical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a scatter plot applied for numerical variable in x-axis vs numerical variable in y-axis.

    

    Args:

        numerical_x (list or str): The numerical variable in x-axis.

        numerical_y (list or str): The numerical variable in y-axis.

        data (pd.DataFrame): The data to plot.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    numerical_x, numerical_y = [numerical_x] if type(numerical_x) == str else numerical_x, [numerical_y] if type(numerical_y) == str else numerical_y

    if nrows is None: nrows = (len(numerical_x)*len(numerical_y) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [sns.scatterplot(x=vj, y=vi, data=data, ax=axes[i*len(numerical_x) + j], rasterized=True) for i, vi in enumerate(numerical_y) for j, vj in enumerate(numerical_x)]

    return fig
# describe training and testing data

df_data.describe(include='all')
# feature exploration: histogram of all numeric features

_ = df_data.hist(bins=20, figsize=(10, 6))
# feature exploration: season for store 1 to 10 and item 1

for i in range(1, 11):

    fig, axes = plt.subplots(figsize=(20, 3))

    _ = df_data.loc[(df_data['store'] == i) & (df_data['item'] == 1) & (df_data['datatype'] == 'training'), 'sales'].plot()

    axes.set_title('store %d, item %d' %(i, 1))
# feature exploration: seasonal decompose for store 5 and item 1

seasonal = sm.tsa.seasonal_decompose(df_data.loc[(df_data['store'] == 5) & (df_data['item'] == 1) & (df_data['datatype'] == 'training'), 'sales']).plot()

seasonal.set_figwidth(20)

seasonal.set_figheight(15)

plt.tight_layout(); plt.show()
# feature extraction: combination of keyword date

df_data['date'] = df_data.index

df_data['year'] = df_data['date'].dt.year - 2000

df_data['quarter'] = df_data['date'].dt.quarter

df_data['month'] = df_data['date'].dt.month

df_data['weekofyear'] = df_data['date'].dt.weekofyear

df_data['dayofweek'] = df_data['date'].dt.dayofweek
# feature extraction: statistic features for store, item and quarter

df_data['item_quarter_mean'] = df_data.groupby(['quarter', 'item'])['sales'].transform('mean')

df_data['store_quarter_mean'] = df_data.groupby(['quarter', 'store'])['sales'].transform('mean')

df_data['store_item_quarter_mean'] = df_data.groupby(['quarter', 'store', 'item'])['sales'].transform('mean')
# feature extraction: statistic features for store, item and month

df_data['item_month_mean'] = df_data.groupby(['month', 'item'])['sales'].transform('mean')

df_data['store_month_mean'] = df_data.groupby(['month', 'store'])['sales'].transform('mean')

df_data['store_item_month_mean'] = df_data.groupby(['month', 'store', 'item'])['sales'].transform('mean')
# feature extraction: statistic features for store, item and weekofyear

df_data['item_weekofyear_mean'] = df_data.groupby(['weekofyear', 'item'])['sales'].transform('mean')

df_data['store_weekofyear_mean'] = df_data.groupby(['weekofyear', 'store'])['sales'].transform('mean')

df_data['store_item_weekofyear_mean'] = df_data.groupby(['weekofyear', 'store', 'item'])['sales'].transform('mean')
# feature extraction: statistic features for store, item and dayofweek

df_data['item_dayofweek_mean'] = df_data.groupby(['dayofweek', 'item'])['sales'].transform('mean')

df_data['store_dayofweek_mean'] = df_data.groupby(['dayofweek', 'store'])['sales'].transform('mean')

df_data['store_item_dayofweek_mean'] = df_data.groupby(['dayofweek', 'store', 'item'])['sales'].transform('mean')
# feature extraction: shifted features for store, item and weekofyear shift 90 days

df_data['store_item_shift90'] = df_data.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(90))

df_data['item_weekofyear_shift90_mean'] = df_data.groupby(['weekofyear', 'item'])['sales'].transform(lambda x: x.shift(13).mean())

df_data['store_weekofyear_shift90_mean'] = df_data.groupby(['weekofyear', 'store'])['sales'].transform(lambda x: x.shift(13).mean())
# feature extraction: shifted features for store, item and weekofyear shift 180 days

df_data['store_item_shift180'] = df_data.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(180))

df_data['item_weekofyear_shift180_mean'] = df_data.groupby(['weekofyear', 'item'])['sales'].transform(lambda x: x.shift(26).mean())

df_data['store_weekofyear_shift180_mean'] = df_data.groupby(['weekofyear', 'store'])['sales'].transform(lambda x: x.shift(26).mean())
# feature extraction: shifted features for store, item and weekofyear shift 270 days

df_data['store_item_shift270'] = df_data.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(270))

df_data['item_weekofyear_shift270_mean'] = df_data.groupby(['weekofyear', 'item'])['sales'].transform(lambda x: x.shift(39).mean())

df_data['store_weekofyear_shift270_mean'] = df_data.groupby(['weekofyear', 'store'])['sales'].transform(lambda x: x.shift(39).mean())
# feature extraction: shifted features for store, item and weekofyear shift 365 days

df_data['store_item_shift365'] = df_data.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(365))

df_data['item_weekofyear_shift365_mean'] = df_data.groupby(['weekofyear', 'item'])['sales'].transform(lambda x: x.shift(52).mean())

df_data['store_weekofyear_shift365_mean'] = df_data.groupby(['weekofyear', 'store'])['sales'].transform(lambda x: x.shift(52).mean())
# feature extraction: fillna with 0

col_fillnas = ['store_item_shift90', 'store_item_shift180', 'store_item_shift270', 'store_item_shift365']

df_data[col_fillnas] = df_data[col_fillnas].fillna(0)
# feature exploration: sales

col_number = df_data.select_dtypes(include=['number']).columns.drop(['id']).tolist()

_ = scatterplot(col_number, 'sales', df_data[df_data['datatype'] == 'training'])
# feature extraction: fillna with 0

df_data['sales'] = df_data['sales'].fillna(0)
# convert category codes for data dataframe

df_data = pd.get_dummies(df_data, columns=None, drop_first=True)
# describe data dataframe

df_data.describe(include='all')
# verify dtypes object

df_data.info()
# compute pairwise correlation of columns, excluding NA/null values and present through heat map

corr = df_data[df_data['datatype_training'] == 1].corr()

fig, axes = plt.subplots(figsize=(200, 150))

heatmap = sns.heatmap(corr, annot=True, cmap=plt.cm.RdBu, fmt='.1f', square=True, vmin=-0.8, vmax=0.8)
def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Return the symmetric mean absolute percentage error (mape).

    

    Args:

        y_true (np.ndarray): The ground truth (correct) labels.

        y_pred (np.ndarray): The predicted labels.

    

    Returns:

        float: The symmetric mean absolute percentage error.

    """

    

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    condition = (y_true > 0) & (y_pred > 0)

    return np.mean(2 * np.abs((y_pred[condition] - y_true[condition])) / (np.abs(y_pred[condition]) + np.abs(y_true[condition]))) * 100
def symmetric_mean_absolute_percentage_error_scoring(model: object, x: np.ndarray, y: np.ndarray) -> float:

    """ Return the symmetric mean absolute percentage error (mape) scoring.

    

    Args:

        y_true (np.ndarray): The ground truth (correct) labels.

        y_pred (np.ndarray): The predicted labels.

    

    Returns:

        float: The symmetric mean absolute percentage error scoringg.

    """

    

    y_pred = model.predict(x)

    return symmetric_mean_absolute_percentage_error(y, y_pred)
# select all features

x = df_data[df_data['datatype_training'] == 1].drop(['id', 'sales', 'date', 'datatype_training'], axis=1)

y = df_data.loc[df_data['datatype_training'] == 1]['sales']
# perform train-test (validate) split

x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=58, test_size=0.25)
# xgboost regression model setup

model_xgbreg = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=1000, objective='reg:linear', booster='gbtree', gamma=0, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.9, random_state=58)



# xgboost regression model fit

model_xgbreg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_validate, y_validate)], early_stopping_rounds=50, verbose=False, callbacks=[xgb.callback.print_evaluation(period=50)])



# xgboost regression model prediction

model_xgbreg_ypredict = model_xgbreg.predict(x_validate)



# xgboost regression model metrics

model_xgbreg_mape = symmetric_mean_absolute_percentage_error(y_validate, model_xgbreg_ypredict)

print('xgboost regression\n  symmetric mean absolute percentaged error: %0.4f' %model_xgbreg_mape)
# model selection

final_model = model_xgbreg



# prepare testing data and compute the observed value

x_test = df_data[df_data['datatype_training'] == 0].drop(['id', 'sales', 'date', 'datatype_training'], axis=1)

y_test = pd.DataFrame(final_model.predict(x_test), columns=['sales'], index=df_data.loc[df_data['datatype_training'] == 0, 'id'])
# submit the results

out = pd.DataFrame({'id': y_test.index, 'sales': y_test['sales']})

out.to_csv('submission.csv', index=False)