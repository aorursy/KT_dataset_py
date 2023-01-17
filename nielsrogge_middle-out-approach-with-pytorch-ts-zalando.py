!pip install pytorchts
import pandas as pd

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
sales_train.date = pd.to_datetime(sales_train.date)
sales_train.head()
sales_train.describe()
sales_train.shop_id.nunique()
sales_train = sales_train[sales_train['item_price'] > 0]
sales_train.loc[sales_train['item_cnt_day'] < 0,'item_cnt_day'] = 0
import matplotlib.pyplot as plt
import seaborn as sns

f, (ax1, ax2) = plt.subplots(figsize=(14,4), nrows=1, ncols=2)
sns.boxplot(sales_train['item_price'].dropna(), ax=ax1)
sns.boxplot(sales_train['item_cnt_day'].dropna(), ax=ax2)
f.suptitle('Distribution of item price and item sales per day')
sales_train = sales_train[sales_train.item_price < 100000]
sales_train = sales_train[sales_train.item_cnt_day < 1001]
per_shop = sales_train.groupby(['shop_id','date'], as_index=False)['item_cnt_day'].sum()
per_shop
data = {}
date_indices = pd.date_range(
        start=per_shop.date.min(),
        end=per_shop.date.max(),
        freq='D'
    )
target = pd.DataFrame(data, index=date_indices)

for shop_id in range(per_shop.shop_id.nunique()):
    daily_sales = per_shop.loc[per_shop['shop_id'] == shop_id]
    daily_sales = daily_sales.drop(["shop_id"], axis=1).set_index('date')
    daily_sales = daily_sales.rename(columns={"item_cnt_day": shop_id})
    # append to dataframe
    target[shop_id] = daily_sales

target.head()
target.tail()
# let's plot the first 20 shops
for c in range(0,20):
    fig = plt.figure(figsize=(12,3))
    plt.plot(target[c])
    plt.title('shop_id: %d' %(c))
test_set = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
test_set.head()
test_set.tail()
test_set.shop_id.nunique()
test_set.shop_id.value_counts().index.sort_values()
targets_shops_test_set = target.iloc[:, test_set.shop_id.value_counts().index.sort_values()]
targets_shops_test_set
targets_shops_test_set.isnull().sum()
targets_shops_test_set = targets_shops_test_set.fillna(0)
targets_shops_test_set
# let's plot the first 20
for c in test_set.shop_id.value_counts().index.sort_values()[:20]:
    fig = plt.figure(figsize=(12,3))
    plt.plot(targets_shops_test_set[c])
    plt.title('shop_id: %d' %(c))
from pts.dataset import ListDataset, FieldName

start = targets_shops_test_set.index[0]
num_series = targets_shops_test_set.shape[1]
prediction_length = 31
freq = "D"

data = targets_shops_test_set.T # shape (10, 1913)

train_ds = ListDataset([{FieldName.TARGET: target,
                         FieldName.START: start}
                        for (target, start) in zip(data.values[:, :-prediction_length], # shape (42, 1003) 
                                                   [pd.Timestamp(start, freq=freq) for _ in range(num_series)])
                        ],
                        freq=freq)

test_ds = ListDataset([{FieldName.TARGET: target,
                        FieldName.START: start}
                       for (target, start) in zip(data.values, # shape (42, 1034)
                                                  [pd.Timestamp(start, freq=freq) for _ in range(num_series)])
                      ],
                        freq=freq)
from pts.dataset import to_pandas

# print out the first time series of the training set
train_entry = next(iter(train_ds))
train_series = to_pandas(train_entry)
train_series
# print out the first time series of the test set (which is the same as the one above, but a bit longer)
test_entry = next(iter(test_ds))
test_series = to_pandas(test_entry)
test_series
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

train_series.plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].legend(["train series"], loc="upper left")

test_series.plot(ax=ax[1])
ax[1].axvline(train_series.index[-1], color='r') # end of train dataset
ax[1].grid(which="both")
ax[1].legend(["test series", "end of train series"], loc="upper left")

plt.show()
from pts.dataset import to_pandas, MultivariateGrouper, TrainDatasets

train_grouper = MultivariateGrouper(max_target_dim=int(len(train_ds)))

test_grouper = MultivariateGrouper(num_test_dates=int(len(test_ds)/len(train_ds)), 
                                   max_target_dim=int(len(train_ds)))

dataset_train = train_grouper(train_ds)
dataset_test = test_grouper(test_ds)
import torch
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

estimator = TransformerTempFlowEstimator(
    d_model=16,
    num_heads=4,
    input_size=42,
    target_dim=len(train_ds),
    prediction_length=prediction_length,
    flow_type='MAF',
    dequantize=True,
    freq=freq,
    trainer=Trainer(
        device=device,
        epochs=14,
        learning_rate=1e-3,
        num_batches_per_epoch=100,
        batch_size=64,
    )
)

predictor = estimator.train(training_data=dataset_train)
# add id of test set to training set
result = pd.merge(sales_train, test_set, how='left', on=['shop_id','item_id'])
result.head()
result.loc[result.ID == 0].shape[0]
test_set.shop_id.value_counts()
for shop_id in test_set.shop_id:
    print(shop_id, result.loc[result.shop_id == shop_id].shape[0])
result.loc[(result['shop_id']==5) & (result['item_id'] == 5037)]
result.loc[(result['shop_id'] == 25) & (result['item_id'] == 2552)]
result.ID.isna().sum()
result.ID.nunique()
test_set.ID.nunique()
