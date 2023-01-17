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
col_date = ['date_time', 'srch_ci', 'srch_co']
train = pd.read_csv('../input/train.csv',
                    dtype={'date_time':np.datetime64, 'srch_ci':np.datetime64, 'srch_co':np.datetime64,
                           'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    usecols=['date_time', 'srch_ci', 'srch_co',
                             'srch_destination_id','is_booking','hotel_cluster'],
                    parse_dates = col_date,
                    chunksize=1000000)
train

for chunk in train:
    for col in col_date:
    # Convert into date-time
    

        col_year = col + "_year"
        col_month = col + "_month"
        col_day = col + "_day"     

        output[col_year] = pd.DatetimeIndex(output[col]).year
        output[col_month] = pd.DatetimeIndex(output[col]).month
        output[col_day] = pd.DatetimeIndex(output[col]).day

        ### Get lag betweend data
        for col1 in col_date:
            for col2 in col_date:
                if col1 is not col2:
                    col_name = "lag_" + col1 + "-" + col2
                    chunk[col_name] = chunk[col1] - chunk[col2]
                    
train[0].head()

aggs = []
print('-'*38)
for chunk in train:
    agg = chunk.groupby(['srch_destination_id',
                         'hotel_cluster'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)
    print('.',end='')
print('')
aggs = pd.concat(aggs, axis=0)
aggs.head()
CLICK_WEIGHT = 0.05
agg = aggs.groupby(['srch_destination_id','hotel_cluster']).sum().reset_index()
agg['count'] -= agg['sum']
agg = agg.rename(columns={'sum':'bookings','count':'clicks'})
agg['relevance'] = agg['bookings'] + CLICK_WEIGHT * agg['clicks']
agg.head()
def most_popular(group, n_max=5):
    ind = group.relevance.nlargest(n_max).index
    most_popular = group.hotel_cluster[ind].values
    return np.array_str(most_popular)[1:-1] # remove square brackets
most_pop = agg.groupby(['srch_destination_id']).apply(most_popular)
most_pop = pd.DataFrame(most_pop).rename(columns={0:'hotel_cluster'})
most_pop.head()
test = pd.read_csv('../input/test.csv',
                    dtype={'srch_destination_id':np.int32},
                    usecols=['srch_destination_id'],)
test = test.merge(most_pop, how='left',left_on='srch_destination_id',right_index=True)
test.head()
test.hotel_cluster.isnull().sum()
most_pop_all = agg.groupby('hotel_cluster')['relevance'].sum().nlargest(5).index
most_pop_all = np.array_str(most_pop_all)[1:-1]
most_pop_all
test.hotel_cluster.fillna(most_pop_all,inplace=True)
test.hotel_cluster.to_csv('predicted_with_pandas.csv',header=True, index_label='id')
