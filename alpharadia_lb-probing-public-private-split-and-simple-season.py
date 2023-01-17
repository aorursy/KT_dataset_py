from pandas import read_csv



sample_submission = read_csv('../input/sample_submission.csv')



only0s = sample_submission.assign(item_cnt_month = 0)

only1s = sample_submission.assign(item_cnt_month = 1)



only0s.head()
only1s.head()
del only0s, only1s



train = read_csv('../input/sales_train.csv')

train_by_month = train.groupby(['date_block_num', 'shop_id', 'item_id'])[['item_cnt_day']].sum().clip(0, 20)

train_by_month.columns = ['item_cnt_month']

train_by_month = train_by_month.reset_index()

del train



train_by_month.head()
train_by_month.groupby('date_block_num')['item_cnt_month'].mean().tail()
print('About %.2f%% of train values are 0s' % (train_by_month[train_by_month['item_cnt_month'] == 0].shape[0] * 100 / train_by_month.shape[0]))
test = read_csv('../input/test.csv')

len(test.shop_id.unique()), len(test.item_id.unique()), len(test)
from itertools import product

from pandas import DataFrame



pairs = DataFrame(list(product(list(range(34)), test.shop_id.unique(), test.item_id.unique())), columns = ['date_block_num', 'shop_id', 'item_id'])

pairs.head()
pairs.shape
def displayWithSize(df):

    print('Shape : %i x %i' % df.shape)

    m = df.memory_usage().sum()

    if m >= 1000000000:

        print('Total memory usage : %.2f Go' % (m / 1000000000))

    else:

        print('Total memory usage : %.2f Mo' % (m / 1000000))

    return df.head()
from numpy import uint8, uint16, float16



pairs_red = pairs.assign(date_block_num = pairs['date_block_num'].astype(uint8))

pairs_red = pairs_red.assign(shop_id = pairs['shop_id'].astype(uint8))

pairs_red = pairs_red.assign(item_id = pairs['item_id'].astype(uint16))

del pairs



inflated_train = pairs_red.merge(train_by_month, on=['date_block_num', 'shop_id', 'item_id'], how='left')

inflated_train.fillna(0.0, inplace=True)

del pairs_red



displayWithSize(inflated_train)
inflated_train.dtypes
from numpy import finfo



finfo(float16)
from numpy import float64



inflated_train[inflated_train['date_block_num'] == 33]['item_cnt_month'].astype(float64).clip(0, 20).mean()
from matplotlib.pyplot import plot

%matplotlib inline



sales_by_month = inflated_train.groupby('date_block_num')['item_cnt_month'].sum().tolist()

plot(sales_by_month)
from statsmodels.tsa.seasonal import seasonal_decompose

from matplotlib.pyplot import figure



decomposition = seasonal_decompose(sales_by_month, freq=12, model='multiplicative')

fig = figure()  

fig = decomposition.plot()  

fig.set_size_inches(15, 8)
def rolling_mean(data, timespan):

    n = len(data)

    output = []

    for i in range(n):

        maxWindow = min(i - max(0, i-timespan//2), min(n, i+timespan//2) - i)

        output.append(data[i-maxWindow:i+maxWindow+1])

    return list(map(lambda x: sum(x) / len(x), output))



rmean = rolling_mean(sales_by_month, 12)

plot(rmean)
from numpy import mean



incr_step = mean(list(map(lambda x: x[0] - x[1], zip(rmean[22:28], rmean[21:27]))))

incr_step
rmean_corrected = rmean[:28]

current = rmean_corrected[-1]



for _ in range(6):

    current += incr_step

    rmean_corrected.append(current)



plot(rmean_corrected)
forecast = list(map(lambda x: x[0] * x[1], zip(rmean_corrected, decomposition.seasonal)))



plot(sales_by_month)

plot(forecast)
forecast.append((rmean_corrected[-1] + incr_step) * decomposition.seasonal[-12])



plot(sales_by_month)

plot(forecast)
month_n = 32

month_n_plus_1 = 33



item_cnt_month_n = inflated_train[inflated_train['date_block_num'] == month_n]['item_cnt_month']



C0 = sum(item_cnt_month_n == 0)

C = len(item_cnt_month_n)

mt = inflated_train[inflated_train['date_block_num'] == month_n_plus_1]['item_cnt_month'].mean()



item_cnt_month_n_plus_1 = item_cnt_month_n / sales_by_month[month_n]

item_cnt_month_n_plus_1 = (item_cnt_month_n_plus_1 * forecast[month_n_plus_1]).clip(0, 20)

item_cnt_month_n_plus_1[item_cnt_month_n_plus_1 == 0] = (C / C0) * (mt - item_cnt_month_n_plus_1.mean())



item_cnt_month_n_plus_1.mean(), mt
from sklearn.metrics import mean_squared_error

from numpy import sqrt



def rmse(y, y_pred):

    return sqrt(mean_squared_error(y, y_pred))



rmse(item_cnt_month_n_plus_1, inflated_train[inflated_train['date_block_num'] == month_n_plus_1]['item_cnt_month'])
month_n = 33

month_n_plus_1 = 34



item_cnt_month_n = inflated_train[inflated_train['date_block_num'] == month_n]['item_cnt_month']



C0 = sum(item_cnt_month_n == 0)

C = len(item_cnt_month_n)

mt = 0.284



item_cnt_month_n_plus_1 = item_cnt_month_n / sales_by_month[month_n]

item_cnt_month_n_plus_1 = (item_cnt_month_n_plus_1 * forecast[month_n_plus_1]).clip(0, 20)

item_cnt_month_n_plus_1[item_cnt_month_n_plus_1 == 0] = max(0, min(20, (C / C0) * (mt - item_cnt_month_n_plus_1.mean())))



item_cnt_month_n_plus_1.mean(), mt
DataFrame(list(zip(range(len(item_cnt_month_n_plus_1)), item_cnt_month_n_plus_1)), columns=['ID', 'item_cnt_month']).to_csv('submission.csv', index=False)