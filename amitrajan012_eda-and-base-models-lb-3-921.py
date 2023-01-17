import pandas as pd
import plotly
import cufflinks as cf
import plotly.graph_objs as go
import plotly.plotly as py
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plotly.offline.init_notebook_mode()
cf.set_config_file(world_readable=True,offline=True)
merchants = pd.read_csv("../input/merchants.csv")
merchants.head()
# Merchants by Subsector
s = merchants.groupby(['subsector_id'])['merchant_id'].count()
df = pd.DataFrame({'subsector_id':s.index, 'merchant_count':s.values})
df.loc[-1] = ['Others', df.loc[df['merchant_count'] < 1000, 'merchant_count'].sum()]
df = df[df['merchant_count'] >= 1000]
df.iplot(kind='pie',labels='subsector_id',values='merchant_count', 
         title='Merchants by Subsector (Subsector with count less than 1000 grouped together)')
# Merchants by Category
s = merchants.groupby(['merchant_category_id'])['merchant_id'].count()
df = pd.DataFrame({'merchant_category_id':s.index, 'merchant_count':s.values})
df.loc[-1] = ['Others', df.loc[df['merchant_count'] < 2000, 'merchant_count'].sum()]
df = df[df['merchant_count'] >= 2000]
df.iplot(kind='pie',labels='merchant_category_id',values='merchant_count', 
         title='Merchants by Category (Category with count less than 2000 grouped together)')
# Merchants by State
s = merchants.groupby(['state_id'])['merchant_id'].count()
df = pd.DataFrame({'state_id':s.index, 'merchant_count':s.values})
df.iplot(kind='pie',labels='state_id',values='merchant_count', title='Merchants by State')
# Merchants by City
s = merchants.groupby(['city_id'])['merchant_id'].count()
df = pd.DataFrame({'city_id':s.index, 'merchant_count':s.values})
df.loc[-1] = ['Others', df.loc[df['merchant_count'] < 1000, 'merchant_count'].sum()]
df = df[df['merchant_count'] >= 1000]
df.iplot(kind='pie',labels='city_id',values='merchant_count', 
         title='Merchants by City (City with count less than 1000 grouped together)')
# Merchants by category_2
s = merchants.groupby(['category_2'])['merchant_id'].count()
df = pd.DataFrame({'category_2':s.index, 'merchant_count':s.values})
df.iplot(kind='pie',labels='category_2',values='merchant_count', 
         title='Merchants by Category 2')
# Merchants by category_4
s = merchants.groupby(['category_4'])['merchant_id'].count()
df = pd.DataFrame({'category_4':s.index, 'merchant_count':s.values})
df.iplot(kind='pie',labels='category_4',values='merchant_count', 
         title='Merchants by Category 4')
# Merchants by most_recent_sales_range
s = merchants.groupby(['most_recent_sales_range'])['merchant_id'].count()
df = pd.DataFrame({'most_recent_sales_range':s.index, 'merchant_count':s.values})
df.iplot(kind='pie',labels='most_recent_sales_range',values='merchant_count', 
         title='Merchants by Range of Revenue in the last active month (A > B > C > D > E)')
# Merchants by most_recent_purchases_range
s = merchants.groupby(['most_recent_purchases_range'])['merchant_id'].count()
df = pd.DataFrame({'most_recent_purchases_range':s.index, 'merchant_count':s.values})
df.iplot(kind='pie',labels='most_recent_purchases_range',values='merchant_count', 
         title='Merchants by Range of quantity of Transactions in the last active month (A > B > C > D > E)')
print("Percentage of merchants who are in the same range of Revenue and Quantity of Transaction is: "
     + str(merchants.loc[merchants['most_recent_sales_range'] == merchants['most_recent_purchases_range']]
           ['merchant_id'].count() / merchants['merchant_id'].count() * 100))
grouped = merchants.groupby(["subsector_id", "most_recent_sales_range"])['merchant_id'].count().reset_index()

traceA = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'A']['subsector_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'A']['merchant_id'],
    name='Range A'
)

traceB = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'B']['subsector_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'B']['merchant_id'],
    name='Range B'
)

traceC = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'C']['subsector_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'C']['merchant_id'],
    name='Range C'
)

traceD = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'D']['subsector_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'D']['merchant_id'],
    name='Range D'
)

traceE = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'E']['subsector_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'E']['merchant_id'],
    name='Range E'
)

data = [traceA, traceB, traceC, traceD, traceE]
layout = go.Layout(
    barmode='stack',
    title='Merchant Counts (Revenue ranges) for Subsectors',
    xaxis=dict(
        title='Subsector ID'
    ),
    yaxis=dict(
        title='Merchant Counts'
    )
)

fig = go.Figure(data=data, layout=layout,)
plotly.offline.iplot(fig)
grouped = merchants.groupby(["subsector_id", "most_recent_purchases_range"])['merchant_id'].count().reset_index()

traceA = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'A']['subsector_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'A']['merchant_id'],
    name='Range A'
)

traceB = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'B']['subsector_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'B']['merchant_id'],
    name='Range B'
)

traceC = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'C']['subsector_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'C']['merchant_id'],
    name='Range C'
)

traceD = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'D']['subsector_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'D']['merchant_id'],
    name='Range D'
)

traceE = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'E']['subsector_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'E']['merchant_id'],
    name='Range E'
)

data = [traceA, traceB, traceC, traceD, traceE]
layout = go.Layout(
    barmode='stack',
    title='Merchant Counts (Quantity of Transactions ranges) for Subsectors',
    xaxis=dict(
        title='Subsector ID'
    ),
    yaxis=dict(
        title='Merchant Counts'
    )
)

fig = go.Figure(data=data, layout=layout,)
plotly.offline.iplot(fig)
grouped = merchants.groupby(["state_id", "most_recent_sales_range"])['merchant_id'].count().reset_index()

traceA = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'A']['state_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'A']['merchant_id'],
    name='Range A'
)

traceB = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'B']['state_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'B']['merchant_id'],
    name='Range B'
)

traceC = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'C']['state_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'C']['merchant_id'],
    name='Range C'
)

traceD = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'D']['state_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'D']['merchant_id'],
    name='Range D'
)

traceE = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'E']['state_id'],
    y=grouped[grouped['most_recent_sales_range'] == 'E']['merchant_id'],
    name='Range E'
)

data = [traceA, traceB, traceC, traceD, traceE]
layout = go.Layout(
    barmode='stack',
    title='Merchant Counts (Revenue ranges) for States',
    xaxis=dict(
        title='State ID'
    ),
    yaxis=dict(
        title='Merchant Counts'
    )
)

fig = go.Figure(data=data, layout=layout,)
plotly.offline.iplot(fig)
grouped = merchants.groupby(["state_id", "most_recent_purchases_range"])['merchant_id'].count().reset_index()

traceA = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'A']['state_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'A']['merchant_id'],
    name='Range A'
)

traceB = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'B']['state_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'B']['merchant_id'],
    name='Range B'
)

traceC = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'C']['state_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'C']['merchant_id'],
    name='Range C'
)

traceD = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'D']['state_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'D']['merchant_id'],
    name='Range D'
)

traceE = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'E']['state_id'],
    y=grouped[grouped['most_recent_purchases_range'] == 'E']['merchant_id'],
    name='Range E'
)

data = [traceA, traceB, traceC, traceD, traceE]
layout = go.Layout(
    barmode='stack',
    title='Merchant Counts (Quantity of Transactions ranges) for States',
    xaxis=dict(
        title='State ID'
    ),
    yaxis=dict(
        title='Merchant Counts'
    )
)

fig = go.Figure(data=data, layout=layout,)
plotly.offline.iplot(fig)
grouped = merchants.groupby(["category_2", "most_recent_sales_range"])['merchant_id'].count().reset_index()

traceA = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'A']['category_2'],
    y=grouped[grouped['most_recent_sales_range'] == 'A']['merchant_id'],
    name='Range A'
)

traceB = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'B']['category_2'],
    y=grouped[grouped['most_recent_sales_range'] == 'B']['merchant_id'],
    name='Range B'
)

traceC = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'C']['category_2'],
    y=grouped[grouped['most_recent_sales_range'] == 'C']['merchant_id'],
    name='Range C'
)

traceD = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'D']['category_2'],
    y=grouped[grouped['most_recent_sales_range'] == 'D']['merchant_id'],
    name='Range D'
)

traceE = go.Bar(
    x=grouped[grouped['most_recent_sales_range'] == 'E']['category_2'],
    y=grouped[grouped['most_recent_sales_range'] == 'E']['merchant_id'],
    name='Range E'
)

data = [traceA, traceB, traceC, traceD, traceE]
layout = go.Layout(
    barmode='stack',
    title='Merchant Counts (Revenue ranges) for Category 2',
    xaxis=dict(
        title='Category 2'
    ),
    yaxis=dict(
        title='Merchant Counts'
    )
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
grouped = merchants.groupby(["category_2", "most_recent_purchases_range"])['merchant_id'].count().reset_index()

traceA = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'A']['category_2'],
    y=grouped[grouped['most_recent_purchases_range'] == 'A']['merchant_id'],
    name='Range A'
)

traceB = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'B']['category_2'],
    y=grouped[grouped['most_recent_purchases_range'] == 'B']['merchant_id'],
    name='Range B'
)

traceC = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'C']['category_2'],
    y=grouped[grouped['most_recent_purchases_range'] == 'C']['merchant_id'],
    name='Range C'
)

traceD = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'D']['category_2'],
    y=grouped[grouped['most_recent_purchases_range'] == 'D']['merchant_id'],
    name='Range D'
)

traceE = go.Bar(
    x=grouped[grouped['most_recent_purchases_range'] == 'E']['category_2'],
    y=grouped[grouped['most_recent_purchases_range'] == 'E']['merchant_id'],
    name='Range E'
)

data = [traceA, traceB, traceC, traceD, traceE]
layout = go.Layout(
    barmode='stack',
    title='Merchant Counts (Quantity of Transactions ranges) for Category 2',
    xaxis=dict(
        title='Category 2'
    ),
    yaxis=dict(
        title='Merchant Counts'
    )
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
print("The correlation coefficient between numerical_1 and numerical_2 is: " + 
     str(merchants['numerical_1'].corr(merchants['numerical_2'])))
fig = plt.figure(figsize=(15,16))
ax = fig.add_subplot(221)
sns.scatterplot(x="most_recent_sales_range", y="numerical_1", data=merchants, color='red')
ax.set_xlabel('Revenue Range')
ax.set_ylabel('Numerical 1')
ax.set_title('Numerical 1 by Revenue Range')
ax.grid()

ax = fig.add_subplot(222)
sns.scatterplot(x="most_recent_purchases_range", y="numerical_1", data=merchants, color='red')
ax.set_xlabel('Quantity of Transactions Range')
ax.set_ylabel('Numerical 1')
ax.set_title('Numerical 1 by Quantity of Transactions Range')
ax.grid()

ax = fig.add_subplot(223)
sns.scatterplot(x="most_recent_sales_range", y="numerical_2", data=merchants, color='green')
ax.set_xlabel('Revenue Range')
ax.set_ylabel('Numerical 2')
ax.set_title('Numerical 2 by Revenue Range')
ax.grid()

ax = fig.add_subplot(224)
sns.scatterplot(x="most_recent_purchases_range", y="numerical_2", data=merchants, color='green')
ax.set_xlabel('Quantity of Transactions Range')
ax.set_ylabel('Numerical 2')
ax.set_title('Numerical 2 by Quantity of Transactions Range')
ax.grid()

plt.show()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121)
sns.scatterplot(x="category_2", y="numerical_1", data=merchants, color='green')
ax.set_xlabel('Category 2')
ax.set_ylabel('Numerical 1')
ax.set_title('Numerical 1 by Category 2')
ax.grid()

ax = fig.add_subplot(122)
sns.scatterplot(x="state_id", y="numerical_1", data=merchants, color='green')
ax.set_xlabel('State ID')
ax.set_ylabel('Numerical 1')
ax.set_title('Numerical 1 by State ID')
ax.grid()
grouped_12 = merchants.groupby(['most_recent_sales_range','active_months_lag12'])['merchant_id'].count().groupby(
    level=[0]).apply(lambda x: x *100 / x.sum()).reset_index()
grouped_12.columns = ['Sales Range', 'Active Month', 'Percent of Merchants']
grouped_12 = grouped_12[grouped_12['Percent of Merchants'] > 80]

grouped_6 = merchants.groupby(['most_recent_sales_range','active_months_lag6'])['merchant_id'].count().groupby(
    level=[0]).apply(lambda x: x *100 / x.sum()).reset_index()
grouped_6.columns = ['Sales Range', 'Active Month', 'Percent of Merchants']
grouped_6 = grouped_6[grouped_6['Percent of Merchants'] > 80]

grouped_3 = merchants.groupby(['most_recent_sales_range','active_months_lag3'])['merchant_id'].count().groupby(
    level=[0]).apply(lambda x: x *100 / x.sum()).reset_index()
grouped_3.columns = ['Sales Range', 'Active Month', 'Percent of Merchants']
grouped_3 = grouped_3[grouped_3['Percent of Merchants'] > 80]

df = pd.concat([grouped_12, grouped_6, grouped_3],ignore_index=True)

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.stripplot(x="Sales Range", y="Percent of Merchants", hue='Active Month', data=df, size=10.0, jitter=False)
ax.set_xlabel('Revenue Range')
ax.set_ylabel('Percentage of Merchant (Active for the entire period)')
ax.set_title('Percentage of Merchant (Active for the entire period) vs Revenue Range')
ax.grid()
fig = plt.figure(figsize=(15,12))
ax = fig.add_subplot(311)

sns.distplot(merchants.dropna()['avg_sales_lag3'], hist=False, color='r', label='based on Last 3 months')
sns.distplot(merchants.dropna()['avg_sales_lag6'], hist=False, color='g', label='based on Last 6 months')
sns.distplot(merchants.dropna()['avg_sales_lag12'], hist=False, color='b', label='based on Last 12 months')

ax.set_xlabel('Average Sales Lag')
ax.set_title('Distribution Plot (Average Sales Lag)')
ax.grid()
ax.legend()

ax = fig.add_subplot(312)

sns.distplot(merchants.dropna()['avg_purchases_lag3'], hist=False, color='r', label='based on Last 3 months')
sns.distplot(merchants.dropna()['avg_purchases_lag6'], hist=False, color='g', label='based on Last 6 months')
sns.distplot(merchants.dropna()['avg_purchases_lag12'], hist=False, color='b', label='based on Last 12 months')

ax.set_xlabel('Average Purchase Lag')
ax.set_title('Distribution Plot (Average Purchase Lag)')
ax.grid()
ax.legend()

ax = fig.add_subplot(313)

sns.distplot(merchants.dropna()[merchants['avg_purchases_lag3'] <= 200]['avg_purchases_lag3'], hist=False, color='r', label='based on Last 3 months')
sns.distplot(merchants.dropna()[merchants['avg_purchases_lag6'] <= 200]['avg_purchases_lag6'], hist=False, color='g', label='based on Last 6 months')
sns.distplot(merchants.dropna()[merchants['avg_purchases_lag12'] <= 200]['avg_purchases_lag12'], hist=False, color='b', label='based on Last 12 months')

ax.set_xlabel('Average Purchase Lag')
ax.set_title('Distribution Plot (Average Purchase Lag after removing outliers)')
ax.grid()
ax.legend()

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(15,12))

q = merchants["avg_sales_lag12"].quantile(0.99)
df = merchants[merchants["avg_sales_lag12"] < q]

q = df["avg_purchases_lag12"].quantile(0.99)
df = df[df["avg_purchases_lag12"] < q]

ax = fig.add_subplot(311)

sns.scatterplot(x="avg_sales_lag12", y="avg_purchases_lag12", hue="most_recent_sales_range", data=df)

ax.set_xlabel('Average Sales Lag (12 months)')
ax.set_ylabel('Average Purchase Lag (12 months)')
ax.set_title('Sales Lag vs Purchase Lag (12 months)')
ax.grid()

q = merchants["avg_sales_lag6"].quantile(0.99)
df = merchants[merchants["avg_sales_lag6"] < q]

q = df["avg_purchases_lag6"].quantile(0.99)
df = df[df["avg_purchases_lag6"] < q]

ax = fig.add_subplot(312)

sns.scatterplot(x="avg_sales_lag6", y="avg_purchases_lag6", hue="most_recent_sales_range", data=df)

ax.set_xlabel('Average Sales Lag (6 months)')
ax.set_ylabel('Average Purchase Lag (6 months)')
ax.set_title('Sales Lag vs Purchase Lag (6 months)')
ax.grid()

q = merchants["avg_sales_lag3"].quantile(0.99)
df = merchants[merchants["avg_sales_lag3"] < q]

q = df["avg_purchases_lag3"].quantile(0.99)
df = df[df["avg_purchases_lag3"] < q]

ax = fig.add_subplot(313)

sns.scatterplot(x="avg_sales_lag3", y="avg_purchases_lag3", hue="most_recent_sales_range", data=df)

ax.set_xlabel('Average Sales Lag (3 months)')
ax.set_ylabel('Average Purchase Lag (3 months)')
ax.set_title('Sales Lag vs Purchase Lag (3 months)')
ax.grid()

plt.tight_layout()
plt.show()
print("The correlation coefficient between avg_sales_lag3 and avg_purchases_lag3 is: " + 
     str(merchants['avg_sales_lag3'].corr(merchants['avg_purchases_lag3'])))
print("The correlation coefficient between avg_sales_lag6 and avg_purchases_lag6 is: " + 
     str(merchants['avg_sales_lag6'].corr(merchants['avg_purchases_lag6'])))
print("The correlation coefficient between avg_sales_lag12 and avg_purchases_lag12 is: " + 
     str(merchants['avg_sales_lag12'].corr(merchants['avg_purchases_lag12'])))
new_merchant_transactions = pd.read_csv("../input/new_merchant_transactions.csv")
new_merchant_transactions.head()
new_merchant_transactions['category_3'] = new_merchant_transactions['category_3'].fillna('UKWN')
df = new_merchant_transactions[['category_3', 'merchant_id']].drop_duplicates()
s = df.groupby(['category_3'])['merchant_id'].count()
df1 = pd.DataFrame({'category_3':s.index, 'merchant_count':s.values})
df1.iplot(kind='pie',labels='category_3',values='merchant_count', 
         title='Merchants by Category 3 (NaN replaced with UKWN)')
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(211)

ax = sns.kdeplot(new_merchant_transactions['purchase_amount'], shade=True, color="b")
ax.set_xlabel('Purchase Amount')
ax.set_title('Distribution Plot (Purchase Amount)')
ax.grid()


q = new_merchant_transactions["purchase_amount"].quantile(0.99)
df = new_merchant_transactions[new_merchant_transactions["purchase_amount"] < q]
ax = fig.add_subplot(212)

ax = sns.kdeplot(df['purchase_amount'], shade=True, color="b")
ax.set_xlabel('Purchase Amount')
ax.set_title('Distribution Plot (Purchase Amount after removing outliers)')
ax.grid()

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(211)

ax = sns.kdeplot(new_merchant_transactions[new_merchant_transactions['category_3'] == 'A']['purchase_amount'],
                 shade=True, color="b", label='Category 3: A')
ax = sns.kdeplot(new_merchant_transactions[new_merchant_transactions['category_3'] == 'UKWN']['purchase_amount'],
                 shade=True, color="r", label='Category 3: UKWN')
ax.set_xlabel('Purchase Amount')
ax.set_title('Distribution Plot (Purchase Amount-Category 3: A, UKWN)')
ax.grid()
ax.legend()

ax = fig.add_subplot(212)
ax = sns.kdeplot(new_merchant_transactions[new_merchant_transactions['category_3'] == 'B']['purchase_amount'],
                 shade=True, color="b", label='Category 3: B')
ax = sns.kdeplot(new_merchant_transactions[new_merchant_transactions['category_3'] == 'C']['purchase_amount'],
                 shade=True, color="r", label='Category 3: C')
ax.set_xlabel('Purchase Amount')
ax.set_title('Distribution Plot (Purchase Amount-Category 3: B, C)')
ax.grid()
ax.legend()

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.boxplot(x="subsector_id", y="purchase_amount", data=new_merchant_transactions,
                showfliers=False)

ax.set_xlabel('Subsector ID')
ax.set_ylabel('Purchase Amount')
ax.set_title('Box Plot (Purchase Amount vs Subsectors)')

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.boxplot(x="state_id", y="purchase_amount", data=new_merchant_transactions,
                showfliers=False)

ax.set_xlabel('State ID')
ax.set_ylabel('Purchase Amount')
ax.set_title('Box Plot (Purchase Amount vs States)')

plt.tight_layout()
plt.show()
s = df.groupby(['installments'])['merchant_id'].count()
df1 = pd.DataFrame({'installments':s.index, 'transaction_count':s.values})
df1.iplot(kind='pie',labels='installments',values='transaction_count', 
         title='Transaction Counts by Installments')
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.boxplot(x="installments", y="purchase_amount", data=new_merchant_transactions,
                showfliers=False)

ax.set_xlabel('Installment Counts')
ax.set_ylabel('Purchase Amount')
ax.set_title('Box Plot (Purchase Amount vs Installment Counts)')

plt.tight_layout()
plt.show()
# Convert Purchase time to DateTime
new_merchant_transactions['purchase_date'] = pd.to_datetime(new_merchant_transactions['purchase_date'],
                                                         format='%Y%m%d %H:%M:%S')
new_merchant_transactions['purchase_hour'] = new_merchant_transactions['purchase_date'].dt.hour
new_merchant_transactions['purchase_day'] = new_merchant_transactions['purchase_date'].dt.day_name()
new_merchant_transactions['purchase_date_only'] = new_merchant_transactions['purchase_date'].dt.day
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.boxplot(x="purchase_date_only", y="purchase_amount", data=new_merchant_transactions,
                showfliers=False)

ax.set_xlabel('Date of the Month')
ax.set_ylabel('Purchase Amount')
ax.set_title('Box Plot (Purchase Amount vs Date of the month)')

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.boxplot(x="purchase_hour", y="purchase_amount", data=new_merchant_transactions,
                showfliers=False)

ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Purchase Amount')
ax.set_title('Box Plot (Purchase Amount vs Hour of the Day)')

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.boxplot(x="purchase_day", y="purchase_amount", data=new_merchant_transactions,
                showfliers=False, 
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

ax.set_xlabel('Day of the Week')
ax.set_ylabel('Purchase Amount')
ax.set_title('Box Plot (Purchase Amount vs Day of the Week)')

plt.tight_layout()
plt.show()
df1 = new_merchant_transactions[['merchant_id', 'card_id', 'category_3', 'purchase_amount', 'purchase_day', 
                                 'purchase_hour', 'purchase_date_only']]
df2 = merchants[['merchant_id', 'numerical_1', 'numerical_2', 'most_recent_sales_range', 'most_recent_purchases_range'
                 ,'category_4']]
df = df1.merge(df2, on='merchant_id', how='left')
df.head()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.boxplot(x="most_recent_sales_range", y="purchase_amount", data=df,
                showfliers=False, 
            order=['A', 'B', 'C', 'D', 'E'])

ax.set_xlabel('Revenue Range (A > B > C > D > E)')
ax.set_ylabel('Purchase Amount')
ax.set_title('Box Plot (Purchase Amount vs Revenue Range)')

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
sns.boxplot(x="most_recent_purchases_range", y="purchase_amount", data=df,
                showfliers=False, 
            order=['A', 'B', 'C', 'D', 'E'])

ax.set_xlabel('Quantity of Transaction (A > B > C > D > E)')
ax.set_ylabel('Purchase Amount')
ax.set_title('Box Plot (Purchase Amount vs Quantity of Transaction)')

plt.tight_layout()
plt.show()
train = pd.read_csv("../input/train.csv")
train.describe()
train.head()
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(211)

ax = sns.kdeplot(train['target'], shade=True, color="b")
ax.set_xlabel('Target')
ax.set_title('Distribution Plot (Target)')
ax.grid()


q = train["target"].quantile(0.99)
df = train[train["target"] < q]
ax = fig.add_subplot(212)

ax = sns.kdeplot(df['target'], shade=True, color="b")
ax.set_xlabel('Target')
ax.set_title('Distribution Plot (Target after removing outliers)')
ax.grid()

plt.tight_layout()
plt.show()
s = train.groupby(['feature_1'])['card_id'].count()
df = pd.DataFrame({'feature_1':s.index, 'count':s.values})
df.iplot(kind='pie',labels='feature_1',values='count', 
         title='Count for Feature 1')
s = train.groupby(['feature_2'])['card_id'].count()
df = pd.DataFrame({'feature_2':s.index, 'count':s.values})
df.iplot(kind='pie',labels='feature_2',values='count', 
         title='Count for Feature 2')
s = train.groupby(['feature_3'])['card_id'].count()
df = pd.DataFrame({'feature_3':s.index, 'count':s.values})
df.iplot(kind='pie',labels='feature_3',values='count', 
         title='Count for Feature 3')
fig = plt.figure(figsize=(15,24))

ax = fig.add_subplot(311)
ax = sns.violinplot(x="feature_1", y="target", data=train, palette="muted")
ax.set_xlabel('Feature 1')
ax.set_title('Distribution (Target vs Feature 1)')
ax.grid()

ax = fig.add_subplot(312)
ax = sns.violinplot(x="feature_2", y="target", data=train, palette="muted")
ax.set_xlabel('Feature 2')
ax.set_title('Distribution (Target vs Feature 2)')
ax.grid()

ax = fig.add_subplot(313)
ax = sns.violinplot(x="feature_3", y="target", data=train, palette="muted")
ax.set_xlabel('Feature 3')
ax.set_title('Distribution (Target vs Feature 3)')
ax.grid()

plt.tight_layout()
plt.show()
feature_1 = [1,2,3,4,5]
feature_2 = [1,2,3]

rows = ['Feature 1: {}'.format(row) for row in feature_1]
cols = ['Feature 2: {}'.format(col) for col in feature_2]


fig = plt.figure(figsize=(15,24))

for f1 in feature_1:
    for f2 in feature_2:
        df = train[(train['feature_1'] == f1) & (train['feature_2'] == f2)]
        if(len(df) > 0):
            ax = fig.add_subplot(5,3,3*(f1-1)+f2)
            ax = sns.violinplot(x="feature_3", y="target", data=df, palette="muted")
            ax.set_xlabel('Feature 3')
            ax.set_title(cols[f2-1])
            ax.set_ylabel(rows[f1-1])
            ax.grid()

plt.tight_layout()
plt.show()
test = pd.read_csv("../input/test.csv")

feature_1 = [1,2,3,4,5]
feature_2 = [1,2,3]

rows = ['Feature 1: {}'.format(row) for row in feature_1]
cols = ['Feature 2: {}'.format(col) for col in feature_2]


fig = plt.figure(figsize=(15,24))

for f1 in feature_1:
    for f2 in feature_2:
        df = test[(test['feature_1'] == f1) & (test['feature_2'] == f2)]
        if(len(df) > 0):
            ax = fig.add_subplot(5,3,3*(f1-1)+f2)
            ax = sns.countplot(x="feature_3", data=df, palette="muted")
            ax.set_xlabel('Feature 3')
            ax.set_title(cols[f2-1])
            ax.set_ylabel(rows[f1-1])
            ax.grid()

plt.tight_layout()
plt.show()
# Convert first active month into datetime and extract year and month
train['first_active_month'] = pd.to_datetime(train['first_active_month'], format='%Y-%m')
train['first_active_year'] = train['first_active_month'].dt.year
train['first_active_month'] = train['first_active_month'].dt.month
train.head()
fig = plt.figure(figsize=(15,24))

ax = fig.add_subplot(311)
ax = sns.violinplot(x="first_active_year", y="target", data=train, palette="muted")
ax.set_xlabel('First Active Year')
ax.set_title('Distribution (Target vs First active year)')
ax.grid()
fig = plt.figure(figsize=(15,24))

ax = fig.add_subplot(311)
ax = sns.violinplot(x="first_active_month", y="target", data=train, palette="muted")
ax.set_xlabel('First Active Month')
ax.set_title('Distribution (Target vs First active month)')
ax.grid()
train = train.join(pd.get_dummies(train['feature_1'], prefix='Feature_1'))
train = train.drop('feature_1', axis=1)
train = train.join(pd.get_dummies(train['feature_2'], prefix='Feature_2'))
train = train.drop('feature_2', axis=1)
train = train.join(pd.get_dummies(train['feature_3'], prefix='Feature_3'))
train = train.drop('feature_3', axis=1)
train = train.join(pd.get_dummies(train['first_active_month'], prefix='first_active_month'))
train = train.drop('first_active_month', axis=1)
train = train.join(pd.get_dummies(train['first_active_year'], prefix='first_active_year'))
train = train.drop('first_active_year', axis=1)
X_train = train.drop(['card_id', 'target'], axis=1)
y_train = train['target']
# Decision Tree Regressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

parameters = {'max_depth':range(1,30),
             'max_features': [0.5, 0.6, 0.7, 0.8, 0.9]}

# Best Model
# {'max_depth': 5, 'max_features': 0.9}
parameters_best_model = {'max_depth':[5],
             'max_features': [0.9]}
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False) 

clf = GridSearchCV(tree.DecisionTreeRegressor(random_state=1), parameters_best_model, n_jobs=4, cv=10, 
                   scoring=mse_scorer)
clf.fit(X=X_train, y=y_train)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 
# Preprocessing of test data
# Convert first active month into datetime and extract year and month
test['first_active_month'] = test['first_active_month'].fillna('2018-01')
test['first_active_month'] = pd.to_datetime(test['first_active_month'], format='%Y-%m')
test['first_active_year'] = test['first_active_month'].dt.year
test['first_active_month'] = test['first_active_month'].dt.month.astype(int)
test.head()
# Preprocessing of test data
test = test.join(pd.get_dummies(test['feature_1'], prefix='Feature_1'))
test = test.drop('feature_1', axis=1)
test = test.join(pd.get_dummies(test['feature_2'], prefix='Feature_2'))
test = test.drop('feature_2', axis=1)
test = test.join(pd.get_dummies(test['feature_3'], prefix='Feature_3'))
test = test.drop('feature_3', axis=1)
test = test.join(pd.get_dummies(test['first_active_month'], prefix='first_active_month'))
test = test.drop('first_active_month', axis=1)
test = test.join(pd.get_dummies(test['first_active_year'], prefix='first_active_year'))
test = test.drop('first_active_year', axis=1)
test.head()
X_test = test.drop(['card_id'], axis=1)
p = tree_model.predict(X_test)
# Submission File
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = p
sub.to_csv("submission_decisiontree.csv", index=False)
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor

lgbm_parameters = {'learning_rate': [0.1, 0.01, 0.005],
                   'bagging_freq': [1, 2],
                   'bagging_fraction': [0.8, 0.9],
                   'feature_fraction': [0.8, 0.9],
                   'lambda_l1' : [0.1, 0.2]}

# Best Model
# {'bagging_fraction': 0.8, 'bagging_freq': 2, 'feature_fraction': 0.9, 'lambda_l1': 0.2, 'learning_rate': 0.01}
lgbm_parameters_best_model = {'bagging_fraction': [0.8], 'bagging_freq': [2], 'feature_fraction': [0.9],
                              'lambda_l1': [0.2], 'learning_rate': [0.01]}
reg = GridSearchCV(LGBMRegressor(objective='regression', boosting_type='gbdt', metric='rmse', random_state=1),
                   lgbm_parameters_best_model, n_jobs=1, cv=10)
reg.fit(X=X_train, y=y_train)
model = reg.best_estimator_
print (reg.best_score_, reg.best_params_) 
# Submission File
p = model.predict(X_test)
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = p
sub.to_csv("submission_lgbm.csv", index=False)
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

xgboost_parameters = {'eta': [0.1, 0.01, 0.05], 
                      'max_depth': [6, 7, 8], 
                      'subsample': [0.8, 1.0],
                      'colsample_bytree': [0.7, 0.8]}

# Best model parameters
# {'colsample_bytree': 0.7, 'eta': 0.1, 'max_depth': 6, 'subsample': 1.0}
xgboost_parameters_best_model = {'colsample_bytree': [0.7], 'eta': [0.1], 'max_depth': [6], 'subsample': [1.0]}
xgboost_reg = GridSearchCV(XGBRegressor(objective='reg:linear', booster='gbtree', eval_metric='rmse', random_state=1),
                   xgboost_parameters_best_model, n_jobs=10, cv=5)
xgboost_reg.fit(X=X_train, y=y_train)
model = xgboost_reg.best_estimator_
print (xgboost_reg.best_score_, xgboost_reg.best_params_)
# Submission File
p = model.predict(X_test)
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = p
sub.to_csv("submission_xgboost.csv", index=False)