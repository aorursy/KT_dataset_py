import numpy as np
import pandas as pd
# Seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# Plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
init_notebook_mode(connected=True)

# Read data from kernel (see above)
path = "../input/parse-json-v2-without-hits-column/"
train = pd.read_pickle(path + 'train_v2_clean.pkl')
test = pd.read_pickle(path + 'test_v2_clean.pkl')

# Unique visitors
unique_vis_train = train.fullVisitorId.nunique()
unique_vis_test = test.fullVisitorId.nunique()
print("Unique fullVisitorId - train: {}, test: {}".format(unique_vis_train, unique_vis_test))

# Print shapes and first 5 rows
print("Train shape: {}, test shape: {}".format(train.shape, test.shape))
train.head()
plt.figure(figsize=(10,4))
plt.title("Train logn scale - transactionsRevenue vs totalTransactionRevenue")
for col in ['totals_totalTransactionRevenue', 'totals_transactionRevenue']:
    train[col] = train[col].astype('float64')
    revenue = train[train[col] > 0][col].dropna()
    ax1 = sns.kdeplot(np.log(revenue))
    
plt.figure(figsize=(10,4))
plt.title("Test logn scale - transactionsRevenue vs totalTransactionRevenue")
for col in ['totals_totalTransactionRevenue', 'totals_transactionRevenue']:
    test[col] = test[col].astype('float64')
    revenue = test[test[col] > 0][col].dropna()
    ax1 = sns.kdeplot(np.log(revenue))
# Revenue by time
train_date_sum = train.groupby('date')['totals_transactionRevenue'].sum().to_frame().reset_index()
test_date_sum = test.groupby('date')['totals_transactionRevenue'].sum().to_frame().reset_index()
# Plot
trace_train = go.Scatter(x = pd.to_datetime(train_date_sum.date.astype(str)),
                        y=train_date_sum['totals_transactionRevenue'].apply(lambda x: np.log(x)),
                         opacity=0.8, name='Train')

trace_test = go.Scatter(x = pd.to_datetime(test_date_sum.date.astype(str)),
                        y=test_date_sum['totals_transactionRevenue'].apply(lambda x: np.log(x)),
                        opacity=0.8, name='Test')
layout = dict(
    title= "Log transactionRevenue by date",
    xaxis=dict(rangeslider=dict(visible=True), type='date')
)
fig = dict(data= [trace_train, trace_test], layout=layout)
iplot(fig)
def train_test_distribution(col, dtype='float64'):
    """Plot a single numerical column distribution in linear and log scale."""
    fig, axis = plt.subplots(1, 2, figsize=(12,4))
    axis[0].set_title("Linear scale")
    axis[1].set_title("Log scale")
    
    train[col], test[col] = train[col].astype(dtype), test[col].astype(dtype)
    ax1 = sns.kdeplot(train[col].dropna(), label='train', ax=axis[0])
    ax2 = sns.kdeplot(test[col].dropna(), label='test', ax=axis[0])
    ax3 = sns.kdeplot(np.log(train[col].dropna()), label='train', ax=axis[1])
    ax4 = sns.kdeplot(np.log(test[col].dropna()), label='test', ax=axis[1])
print(train.totals_transactions.value_counts(dropna=False).head())
train_test_distribution('totals_transactions')
train_test_distribution('totals_sessionQualityDim')
train_test_distribution('totals_timeOnSite')
counts = train.customDimensions_value.value_counts(dropna=False).to_frame().reset_index()
plt.figure(figsize=(8,4))
g = sns.barplot(x='index', y='customDimensions_value', data=counts)
unique_test = test.fullVisitorId.unique()
print("There are", len(unique_test), "unique users (fullVisitorId) in test")
rev = test.groupby('fullVisitorId')['totals_transactionRevenue'].sum()
print("There are", len(rev[rev > 0]), "unique users with revenue (greater than 0) in test")
print("So only {:.1f}% of users have revenue".format(100 * len(rev[rev > 0]) / len(unique_test)))
plt.figure(figsize=(8,4))
plt.title("Distribution of revenue per user (only users with revenue > 0)")
ax = sns.kdeplot(np.log(rev[rev > 0]), label='Log transactionRevenue')
users_period = [
    (20160801, 20170115),
    (20170115, 20170630),
    (20170701, 20171215),
    (20171216, 20180601),
    # Using the same months!
    (20170501, 20171015),
]

predict_period = [
    (20170301, 20170430),
    (20170715, 20170915),
    (20180201, 20180331),
    (20180715, 20180915),
    # Using the same months!
    (20171201, 20180131),
]

# Join train and test
df = pd.concat([train, test])
# Save revenue for plot
revenues_list = []

for i in range(5):
    print("\nPeriod", i+1)
    a, b = users_period[i]
    batch = df[(df.date >= a) & (df.date <= b)]
    batch_visitors = batch.fullVisitorId.unique()
    print("There are", len(batch_visitors), "visitors in 5.5 months")
    
    c, d = predict_period[i]
    pred = df[(df.date >= c) & (df.date <= d)]
    pred_visitors = pred.fullVisitorId.unique()
    print("There are", len(pred_visitors), "visitors in 2 months")
    # Returning visitors
    same_visitors = np.intersect1d(batch_visitors, pred_visitors)
    print("{} visitors returned or {:.2f}%".format(len(same_visitors), 100*len(same_visitors)/len(batch_visitors)))
    # Returning visitors revenue
    with_rev = pred[(pred.fullVisitorId.isin(same_visitors)) & (pred.totals_transactionRevenue > 0)]
    print("And only {} returning visitors have revenue or {:.2f}% from total".format(len(with_rev), 100*len(with_rev)/len(batch_visitors)))
    print("The total revenue for this users is U$$ {:.2f}".format(with_rev.totals_transactionRevenue.sum()/1000000))
    revenues_list.append(with_rev.copy(deep=True))
plt.figure(figsize=(8,4))
plt.title("Distribution of revenue for returning customers")
ax = sns.kdeplot(np.log(revenues_list[-1].totals_transactionRevenue), label='Log transactionRevenue')