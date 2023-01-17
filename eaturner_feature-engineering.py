# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



for col in train.columns:

    print (col)
train[['pickup_datetime']].head(5)
def toDateTime( df ):

    

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    

    df['month'] = df['pickup_datetime'].dt.month

    df['hour'] = df['pickup_datetime'].dt.hour

    df['day_week'] = df['pickup_datetime'].dt.weekday_name

    

    return df
train = toDateTime(train)

test = toDateTime(test)
train['trip_duration'] = np.log1p(train['trip_duration'])
train.groupby('month')['trip_duration'].describe()
train.groupby('hour')['trip_duration'].describe()
train.groupby('day_week')['trip_duration'].describe()
def violinPlot( df, by_col ):

    import seaborn as sns

    

    sns.violinplot(x = by_col, y = 'trip_duration', data = df)
violinPlot(train, 'month')
violinPlot(train, 'day_week')
violinPlot(train, 'hour')
def locationFeatures( df ):

    #displacement

    df['y_dis'] = df['pickup_longitude'] - df['dropoff_longitude']

    df['x_dis'] = df['pickup_latitude'] - df['dropoff_latitude']

    

    #square distance

    df['dist_sq'] = (df['y_dis'] ** 2) + (df['x_dis'] ** 2)

    

    #distance

    df['dist_sqrt'] = df['dist_sq'] ** 0.5

    

    return df
train = locationFeatures(train)

test = locationFeatures(test)
train.plot(x = 'trip_duration', y = 'y_dis', kind = 'scatter')
train.plot(x = 'trip_duration', y = 'x_dis', kind = 'scatter')
train.plot(x = 'trip_duration', y = 'dist_sq', kind = 'scatter')
train.plot(x = 'trip_duration', y = 'dist_sqrt', kind = 'scatter')