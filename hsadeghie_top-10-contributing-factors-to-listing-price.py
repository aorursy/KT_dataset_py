from glob import glob
import pandas as pd
import numpy as np
import seaborn
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
path = '../input/seattle/'
df = pd.read_csv(path + 'listings.csv')
calendar = pd.read_csv(path + 'calendar.csv')
reviews = pd.read_csv(path + 'reviews.csv')
def convert_time(dataframe, name):
    """
    This function takes a dataframe as an input, plus the name of the date column. Then splits the date by '-', 
    and converts to year, month, and day. It also extracts day of the week and week of the year as features.
    
    Args:
    dataframe (pandas.DataFrame): the input dataframe
    name (str): the name of the column to be treated as date string
    
    Returns:
    pd.DataFrame: with new columns for year, month, day, dayofweek, and weekofyear
    """
    to_numeric = lambda x: float(x.lstrip('$').replace(',', ''))
    suffixes = ['y', 'm', 'd']
    date = pd.to_datetime(dataframe[name])
    dataframe['dayofweek'] = date.dt.dayofweek
    dataframe['weekofyear'] = date.dt.week
    split_df = dataframe[name].str.split('-', expand=True).rename(columns={idx: '%s_%s' % (name, suffix) for idx, suffix in enumerate(suffixes)}).applymap(float)
    dataframe = pd.concat((dataframe.drop([name], axis=1), split_df), axis=1)
    dataframe['price'] = dataframe.price.apply(to_numeric)
    return dataframe
    
calendar.dropna(inplace=True)
df = df.applymap(lambda x: str.lower(x) if isinstance(x, str) else x)

calendar = convert_time(calendar, 'date')
df = convert_time(df, 'host_since')
city_data = df.groupby('city')['price']
city_data.mean()
df[[x for x in df.columns if 'neighb' in x]]
ax = seaborn.boxplot(data=df, x='neighbourhood_group_cleansed', y='price', fliersize=0.1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylim(0, 400);
relevant_data = df.groupby('neighbourhood_group_cleansed')[['price', 'longitude', 'latitude']].mean()
plt.figure(figsize=(10, 10))
seaborn.scatterplot(data=df, x='longitude', y='latitude', size='price', marker='o', color='skyblue')
seaborn.scatterplot(data=relevant_data, x='longitude', y='latitude', size='price', marker='o', color='red')
data = df[[x for x in df.columns if 'host' in x] + ['price']].drop(['host_id'], axis=1)
seaborn.heatmap(data.corr(), fmt='.2f', annot=True)
df.groupby('host_since_y')['price'].mean().plot.bar(title='Price by year since the host joined', yerr=df.groupby('host_since_y')['price'].std())
seaborn.heatmap(calendar.drop('listing_id', axis=1).corr(), fmt='.2f', annot=True)
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
seaborn.boxplot(data=calendar, x='date_m', y='price', ax=axs[0], fliersize=0)
seaborn.boxplot(data=calendar, x='dayofweek', y='price', ax=axs[1], fliersize=0)
seaborn.boxplot(data=calendar, x='date_y', y='price', ax=axs[2], fliersize=0)
for ax in axs:
    ax.set_ylim(0, 350)
def group_columns(df):
    """
    This function takes a dataframe and groups the columns by their characteristics.
    This is relatively a general function and reusable on other datasets
    
    Args:
    df (pandas.DataFrame): The input data frame
    
    Returns:
    dict: A dictionary of column types and list of columns
    """
    columns = df.columns
    columns_list = {'categorical': [], 'binary': [], 'text': [], 'number': [], 'drop': []}
    for col in columns:
        if col == 'id':
            continue
        elif 'id' in col or 'url' in col:
            columns_list['drop'].append(col)
        else:
            if df[col].dtype == np.dtype('O'):
                n = len(np.unique(df[col][df[col].notnull()]))
                if n == 1:
                    columns_list['drop'].append(col)
                elif n < 20:
                    columns_list['categorical'].append(col)
                else:
                    columns_list['text'].append(col)
            else:
                columns_list['number'].append(col)
    return columns_list

def clean_data(df):
    """
    This function cleans the input data frame by removing unwanted columns, turning categorical variables to 
    one-hot numbers, and dropping columns or filling NAN values based on the frequency of NAN in the columns.
    """
    df.drop(['city'], axis=1, inplace=True)
    columns_list = group_columns(df)

    df = df.drop(columns_list['drop'], axis=1)
    for col in columns_list['categorical']:
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=True)
        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        columns_list['number'].extend(dummies.columns.tolist())
        assert(col not in df.columns)
    for col in columns_list['text']:
        df[col] = df[col].fillna('na')

    df_new = df[columns_list['number'] + ['id']]
    
    df_nan = df.isnull().mean()
    for col in df_new.columns:
        if df_nan[col] > 0.2:
            df_new.drop(col, axis=1, inplace=True)
        elif col in df_nan.index:
            df_new.fillna({col: df_new[col].mean()}, inplace=True)
    return df_new

df_new = clean_data(df)
df = pd.merge(calendar.drop('available', axis=1), df_new, left_on='listing_id', right_on='id')
X = df.drop(['price_x', 'price_y', 'listing_id', 'id'], axis=1)
drop_more = [x for x in X.columns if X[x].std() <= 1e-6]
X.drop(drop_more, axis=1, inplace=True)
y = df['price_x']
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(xtrain, ytrain)

ypred = model.predict(xtrain)
ypredtest = model.predict(xtest)

r2 = r2_score(ytrain, ypred)
r2test = r2_score(ytest, ypredtest)

print("Training r2-score: %.4f,  Test r2-score: %.4f" % (r2, r2test))
inds = np.argsort(-model.feature_importances_)
importance_ds = pd.Series(model.feature_importances_[inds][:20], index=X.columns[inds][:20])
cols = X.columns[inds][:10]
new_data = pd.concat((X[cols], y), axis=1)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
seaborn.heatmap(new_data.corr(), fmt='.2f', annot=True, ax=axs[0])
importance_ds.plot.bar(title='Top 20 important features', ax=axs[1])