import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from sklearn.linear_model import LinearRegression



raw_mcr = pd.read_csv('../input/multipleChoiceResponses.csv',encoding="ISO-8859-1", low_memory=False)

print(raw_mcr.shape)

raw_mcr.head().T
print("CompensationAmount will be our target column (filtering to dollars for simplicity)")

raw_mcr.columns[raw_mcr.columns.str.contains('compensation', flags=re.IGNORECASE)]
print("If you filter for 70%+ non-null columns, you lose most of your data already")

mcr = raw_mcr[raw_mcr.CompensationCurrency == 'USD'].copy()

print("Shape before filter: {}".format(mcr.shape))

mcr = mcr.loc[:, mcr.isnull().mean() < .7]

assert (mcr.columns == ('CompensationAmount')).any()

mcr['CompensationAmount'] = pd.to_numeric(mcr['CompensationAmount'].str.replace(r'\D',''))

print("Shape after filter: {}".format(mcr.shape))
print("Even for that small subset of columns,"

      " if you drop rows with any nulls you are left with ZERO data")

print("Original shape: {}".format(mcr.shape))

print("Shape if nulls dropped across all columns: {}".format(mcr.dropna(axis=0).shape))
print("For simplicity, we'll focus on predictions using the 'Time Spent' columns,"

      " like 'TimeGatheringData'. Using just this one variable won't give great"

      " predictions, but it will be a good example of the topic")
def simple_plot(y_pred, y_all):

    plt.scatter(y_pred, y_all, s=.2, alpha=.2)

    plt.plot([.75e5, 1.5e5], [.75e5, 1.5e5], linestyle='--', c='k', alpha=.25)

    ax = plt.gca()

    f = plt.gcf()

    f.set_figheight(5)

    f.set_figwidth(7)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('${:0,.0f}').format(y)))

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('${:0,.0f}').format(y)))

    ax.set_xlim(0, 2e5)

    ax.set_ylim(0, 2e5)

    ax.set_title("Relationship between Predicted and actual Salary (Full Data)");

    ax.set_xlabel("Salary (Predicted)");

    ax.set_ylabel("Salary (Actual)");

    return ax
print("Because none of the Time data is null, we can get a perfect picture of how good it should be"

      " as a predictor of salary. It's pretty terrible, but works as an example.")

filtered_data = mcr[['TimeGatheringData', 'TimeModelBuilding', 'TimeProduction', 'TimeVisualizing', 'CompensationAmount']].dropna(axis=0, how='any')

lr_all = LinearRegression()

X_all = filtered_data[['TimeGatheringData']].values

y_all = filtered_data['CompensationAmount'].values

lr_all.fit(X_all, y_all)

y_pred = lr_all.predict(X_all)



ax = simple_plot(y_pred, y_all);
print("Now let's introduce some missing values randomly, and try to fix them")

print("If you just impute the mean like normal, you get a weird cluster of data")



# Creating the nulls

with_nulls = filtered_data.copy()

with_nulls.loc[np.random.choice(with_nulls.index, size=int(with_nulls.index.size/3)), 

              'TimeGatheringData'] = np.nan



# Fixing the nulls

with_nulls['TimeGatheringData'] = with_nulls['TimeGatheringData'].fillna(

                                  with_nulls['TimeGatheringData'].mean())





ax = with_nulls['TimeGatheringData'].hist()

ax.set_title("Histogram of Time Gathering Data Values");

ax.set_xlabel("Time Gathering Data (Value)");

ax.set_ylabel("Number of data points");
print("The prediction is now even more wrong, and the tight range of predicted values could give"

      " the impression of better accuracy than you have")

lr_all = LinearRegression()

X_all = with_nulls[['TimeGatheringData']].values

y_all = with_nulls['CompensationAmount'].values

lr_all.fit(X_all, y_all)

y_pred = lr_all.predict(X_all)

simple_plot(y_pred, y_all);
print("If you just bootstrap, at least you don't get that same cluster")



# Creating the nulls

with_nulls = filtered_data.copy()

with_nulls.loc[np.random.choice(with_nulls.index, size=int(with_nulls.index.size/3)), 

              'TimeGatheringData'] = np.nan



# Fixing the nulls

null_ix = with_nulls.loc[:, 'TimeGatheringData'][lambda x: np.isnan(x)].index

notnull_ix = with_nulls.index.difference(null_ix)

with_nulls.loc[null_ix, 'TimeGatheringData'] = np.random.choice(

                                                with_nulls.loc[notnull_ix, 'TimeGatheringData'],

                                                size = len(null_ix),

                                                replace=True)





with_nulls['TimeGatheringData'].hist();
print("Now the prediction looks roughly like the prediction with full data.")

lr_all = LinearRegression()

X_all = with_nulls[['TimeGatheringData']].values

y_all = with_nulls['CompensationAmount'].values

lr_all.fit(X_all, y_all)

y_pred = lr_all.predict(X_all)

simple_plot(y_pred, y_all)
""" Iterative regression imputation"""

print("A more advanced technique is to iteratively fit the data with linear regressions"

     " The disadvantage here is that the values will be drawn towards the mean, as the"

     " linear regression doesn't capture all of the variance of the original data.")





# Creating the nulls, now across all columns

with_nulls = filtered_data.copy()

nullcnt = int(filtered_data.index.size/3)

for col in filtered_data.columns:

    if col == 'CompensationAmount': continue

    null_ix = np.random.choice(with_nulls.index, size=nullcnt)

    with_nulls.loc[null_ix, col] = np.nan



# Fix nulls

def fill_with_mean(col):

    return col.fillna(col.mean())



def fill_with_resample(col):

    col.loc[col.isnull()] = np.random.choice(col.loc[col.notnull()].values, size=col.isnull().sum())

    return col



null_mask = with_nulls.isnull()

with_nulls.apply(fill_with_resample)

all_x_cols = filtered_data.columns.difference(['CompensationAmount'])

for i in range(10):

    for col in filtered_data.columns:

        lr = LinearRegression()

        y_col = filtered_data[col]

        x_col = filtered_data[all_x_cols.difference([col])]

        col_pred = lr.fit(x_col, y_col).predict(x_col)

        filtered_data.loc[:, col] = filtered_data[col].where(null_mask[col], col_pred)





filtered_data['TimeGatheringData'].hist();
print("I'm not going to pretend that this is a great prediction,"

      " but it avoids the artifacts from the other approaches.")

lr_all = LinearRegression()

X_all = with_nulls[['TimeGatheringData']].values

y_all = with_nulls['CompensationAmount'].values

lr_all.fit(X_all, y_all)

y_pred = lr_all.predict(X_all)

simple_plot(y_pred, y_all)