import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix, lil_matrix
from scipy import spatial
from scipy import stats

%matplotlib inline
data = pd.read_csv('../input/data.csv', encoding='ISO-8859-1', engine='c')
data.info()
data.head()
def count_na(data):
    na_count = pd.DataFrame(data.isnull().sum(), columns=['Count']).sort_values(by=['Count'], ascending=False)
    return na_count
count_na(data)
data.InvoiceDate = pd.to_datetime(data.InvoiceDate, format="%m/%d/%Y %H:%M")
data["CustomerID"] = data["CustomerID"].fillna(-9999)
data["CustomerID"] = data["CustomerID"].apply(lambda x: str(x)[:-2])
data["Description"] = data["Description"].fillna("Unkown")
count_na(data)
data.describe()
print(data.shape)

data = data[ np.abs((data['UnitPrice']-data['UnitPrice'].mean())/data['UnitPrice'].std()) <= 3]
data = data[ np.abs((data['Quantity']-data['Quantity'].mean())/data['Quantity'].std()) <= 3]

data = data[data["UnitPrice"] >= 0 ]
data = data[data["InvoiceNo"].astype(str).str[0] != "C"]
data = data[data["InvoiceNo"].astype(str).str[0] != "A"]
data = data[data["Quantity"] > 0 ]

print(data.shape)
data.describe()
data['Revenue'] = data['Quantity']*data['UnitPrice']

data['Date'] = data['InvoiceDate'].dt.date
data['Day'] = data['InvoiceDate'].dt.day
data['Month'] = data['InvoiceDate'].dt.month
data['Year'] = data['InvoiceDate'].dt.year
data['Hour'] = data['InvoiceDate'].dt.hour
data['Week'] = data['InvoiceDate'].dt.week
data['Minute'] = data['InvoiceDate'].dt.minute
data.head()
fig, ax = plt.subplots(2,1, figsize=(9, 9))

sales = data[['Year', 'Month', 'Revenue']].groupby(['Year', 'Month']).sum().reset_index()
sales['Day'] = 1
sales['Date'] = pd.to_datetime(sales[['Year', 'Month', 'Day']])
sales = sales.set_index('Date')
sales = sales.drop(['Year', 'Month', 'Day'], axis=1)

ax[0].plot(sales)
ax[0].set_title('Monthly revenue')        


sales_top = data[['Revenue', 'Country']].groupby(['Country']).sum().reset_index().sort_values(by='Revenue', ascending=False)['Country'][1:6]

for c in sales_top:
    sales = data[data['Country'] == c]
    sales = sales[['Year', 'Month', 'Revenue']].groupby(['Year', 'Month']).sum().reset_index()
    sales['Day'] = 1
    sales['Date'] = pd.to_datetime(sales[['Year', 'Month', 'Day']])
    sales = sales.set_index('Date')
    sales = sales.drop(['Year', 'Month', 'Day'], axis=1)
    ax[1].plot(sales.cumsum(), label=c)    
    ax[1].legend()
    ax[1].set_title('Top countries by revenue')            
    
plt.show()
grps = np.array([['Month', 'Week'], ['Hour', 'Minute']])

fig, ax = plt.subplots(grps.shape[0],grps.shape[1], figsize=(13, 13))

for i in range(0, grps.shape[0]):
    for j in range(0, grps.shape[1]):
        grp_data = data[[grps[i,j], 'InvoiceNo']].groupby([grps[i,j]]).count()
        ax[i,j].bar(x=grp_data.index, height='InvoiceNo', data=grp_data)
        ax[i,j].set_ylabel('Amount of transactions')
        ax[i,j].set_xlabel(grps[i,j])
        
plt.show()
ctry = np.array([['United Kingdom', 'Japan'], ['Germany', 'France']])
fltr = ['DOT', 'POST', 'M']
top_n = 10

fig, ax = plt.subplots(grps.shape[0],grps.shape[1], figsize=(14, 14))

for i in range(0, ctry.shape[0]):
    for j in range(0, ctry.shape[1]):
        grp_data = data[data['Country'] == ctry[i,j]]
        grp_data = grp_data[~grp_data['StockCode'].isin(fltr)]
        grp_data = grp_data[['StockCode', 'Revenue']].groupby(['StockCode']).sum().sort_values(by='Revenue', ascending=False)        
        grp_data = grp_data[0:top_n]    
        grp_data = grp_data.reset_index()
        
        ax[i,j].barh(y=grp_data.index, width='Revenue', data=grp_data)
        ax[i,j].invert_yaxis()
        ax[i,j].set_yticks(range(0,top_n))
        ax[i,j].set_yticklabels(grp_data['StockCode'].tolist())
        ax[i,j].set_ylabel('Stock code')        
        ax[i,j].set_xlabel('Revenue')                
        ax[i,j].set_title('Top 10 ' + ctry[i,j])        
        
plt.show()
stock_top = data[['Revenue', 'StockCode']].groupby(['StockCode']).sum().reset_index().sort_values(by='Revenue', ascending=False)['StockCode'][0:5]

fig, ax = plt.subplots(2,1, figsize=(12, 9))

for c in stock_top:
    sales = data[data['StockCode'] == c]
    sales = sales[['Year', 'Month', 'Revenue']].groupby(['Year', 'Month']).sum().reset_index()
    sales['Day'] = 1
    sales['Date'] = pd.to_datetime(sales[['Year', 'Month', 'Day']])
    sales = sales.set_index('Date')
    sales = sales.drop(['Year', 'Month', 'Day'], axis=1)
    ax[0].plot(sales.Revenue, label=c)
    ax[0].legend()
    ax[0].set_title('Monthly revenue')            

for c in stock_top:
    sales = data[data['StockCode'] == c]
    sales = sales[['Year', 'Month', 'Revenue']].groupby(['Year', 'Month']).sum().reset_index()
    sales['Day'] = 1
    sales['Date'] = pd.to_datetime(sales[['Year', 'Month', 'Day']])
    sales = sales.set_index('Date')
    sales = sales.drop(['Year', 'Month', 'Day'], axis=1)
    ax[1].plot(sales.cumsum(), label=c)
    ax[1].legend()
    ax[1].set_title('Cumulative revenue')        
def top_stock(stockcode, top_n):
    """ Plot cumulative sales in pound sterling, 
    clients who bought the most, distribution of order size
    and pie chart with country % of buyers.    
    Parameters:
    stockcode: string with stock id
    top_n: how many bars etc to plot"""

    stock = data[data['StockCode'] == stockcode].sort_values(['InvoiceDate'])
    stock['ValueCum'] = stock['Revenue'].cumsum()

    top_buyers = stock[['Revenue', 'CustomerID']].groupby(['CustomerID']).sum().sort_values(by='Revenue', ascending=False).reset_index()
    top_buyers = top_buyers[0:top_n]

    ctry = stock[['Country', 'Revenue']].groupby(['Country']).sum().sort_values(by='Revenue', ascending=False)
    ctry = ((ctry/ctry.sum())*100).round(2)
    ctry = ctry[0:int(top_n/2)]
    ctry = ctry.append(pd.DataFrame(data=[100-ctry.sum()], columns=['Revenue'], index=['Other'] ))

    plt.clf()

    fig, ax = plt.subplots(grps.shape[0],grps.shape[1], figsize=(14, 14))
    
    ax[0,0].plot(stock['InvoiceDate'].dt.date, stock['ValueCum'])
    ax[0,0].set_ylabel('Cumulative revenue')        
    ax[0,0].set_title(stock['Description'].iloc[0])    

    ax[0,1].invert_yaxis()
    ax[0,1].barh(y=top_buyers.index, width='Revenue', data=top_buyers)
    ax[0,1].set_yticks(range(0,top_n))
    ax[0,1].set_yticklabels(top_buyers['CustomerID'].tolist())
    ax[0,1].set_ylabel('Customer ID')        
    ax[0,1].set_xlabel('Revenue')      
    ax[0,1].set_title("Clients who bought the most")        

    ax[1,0].hist(stock['Quantity'], bins=range(0,stock['Quantity'].max(), top_n) )
    ax[1,0].set_ylabel('Count')        
    ax[1,0].set_xlabel('Order size')     

    ax[1,1].pie(ctry['Revenue'], labels=ctry.index,  autopct='%1.0f%%', pctdistance=0.85)

    plt.show()
    
top_stock(stockcode="22629", top_n = 10)    
def stalk_customer(customer_id):        
    c_data = data[data['CustomerID'] == customer_id ]

    most_ordered = c_data[['StockCode', 'Revenue']].groupby(['StockCode']).sum().sort_values(by='Revenue', ascending=False)
    most_ordered = most_ordered[0:10]

    last_inv = c_data[['InvoiceNo', 'Revenue', 'Date']].groupby(['InvoiceNo', 'Date'], as_index='False').sum().sort_values(by='Revenue')
    last_inv = last_inv.reset_index()
    last_inv = last_inv.set_index('Date').sort_index(ascending=False)
    last_inv = last_inv[0:10]

    inv_diff = data[data['CustomerID'] == customer_id][['InvoiceNo', 'Revenue', 'Date']]
    inv_diff = inv_diff[['Revenue', 'Date']].groupby(['Date'], as_index=False).sum()
    inv_diff = inv_diff.sort_values(by='Date', ascending=False)
    inv_diff['DateDiff'] = (inv_diff['Date'] - inv_diff['Date'].shift(-1) ).dt.days
    inv_diff = inv_diff.fillna(0, axis=1)

    fig, ax = plt.subplots(2,2, figsize=(10, 10))

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    ax[0,0].barh(y=most_ordered.index, width="Revenue", data=most_ordered)
    ax[0,0].set_ylabel('Stock code')        
    ax[0,0].set_xlabel('Rev')     
    ax[0,0].set_title('Most ordered products of customer ' + str(customer_id))    

    ax[0,1].bar(x=last_inv.index, height='Revenue', data=last_inv)
    ax[0,1].set_title('Last 10 invoices')    

    ax[1,0].hist(inv_diff['DateDiff'])
    ax[1,0].set_ylabel('Count')        
    ax[1,0].set_xlabel('Days between invoices')     
    ax[1,0].set_title('Invoice inter-arrival distribution')    

    plt.sca(ax[0,1])
    plt.xticks(rotation=90)

    plt.show()
    
stalk_customer(customer_id = '17841')
def interarrival(cid):
    s = data[data['CustomerID'] == str(cid)].sort_values(by=['Date'])['Date']    
    x = (s - s.shift(1)).dropna()
    
    if x.empty:
        return 0
    
    else:
        return x.dt.days.mean()
c_data = data[['CustomerID', 'InvoiceDate']].groupby('CustomerID', as_index=False)['InvoiceDate'].min()
c_data.columns = ['CustomerID', 'i_first']

c_data['i_first'] = data[['CustomerID', 'InvoiceDate']].groupby('CustomerID', as_index=False)['InvoiceDate'].min()['InvoiceDate']
c_data['i_last'] = data[['CustomerID', 'InvoiceDate']].groupby('CustomerID', as_index=False)['InvoiceDate'].max()['InvoiceDate']

c_data['d_as_customer']  = (max(c_data['i_last']) - c_data['i_first']).dt.days
c_data['d_last_invoice']  = (max(c_data['i_last']) - c_data['i_last'] ).dt.days

c_data['i_last_30D']  = c_data['d_last_invoice'].apply(lambda x: 1 if x < 30 else 0)
c_data['i_last_90D']  = c_data['d_last_invoice'].apply(lambda x: 1 if x >= 30 and x < 90 else 0)
c_data['i_last_180D'] = c_data['d_last_invoice'].apply(lambda x: 1 if x >= 90 and x < 180 else 0)
c_data['i_over_180D'] = c_data['d_last_invoice'].apply(lambda x: 1 if x >= 180 else 0)

c_data['i_arrival_rate'] = c_data['CustomerID'].apply(interarrival)

c_data['t_invoices'] = data[['CustomerID', 'InvoiceDate']].groupby('CustomerID', as_index=False)['InvoiceDate'].count()['InvoiceDate']
c_data['t_units'] = data[['CustomerID', 'Quantity']].groupby('CustomerID', as_index=False)['Quantity'].sum()['Quantity']
c_data['t_rev'] = data[['CustomerID', 'Revenue']].groupby('CustomerID', as_index=False)['Revenue'].sum()['Revenue']

c_data['u_orders'] = data[['CustomerID', 'StockCode']].groupby('CustomerID', as_index=False).nunique()['StockCode']
c_data['pct_rev_total'] = (c_data['t_rev']/data['Revenue'].sum())*100
c_data['avg_rev'] = data[['CustomerID', 'Revenue']].groupby('CustomerID', as_index=False)['Revenue'].mean()['Revenue']

c_data['a_rate_week'] = c_data['i_arrival_rate'].apply(lambda x:  1 if x <  7  else 0)
c_data['a_rate_2week'] = c_data['i_arrival_rate'].apply(lambda x: 1 if x >= 7  and x < 14 else 0)
c_data['a_rate_month'] = c_data['i_arrival_rate'].apply(lambda x: 1 if x >= 14 and x < 30 else 0)
c_data['a_rate_small'] = c_data['i_arrival_rate'].apply(lambda x:  1 if x >= 30 else 0)

c_data = c_data[c_data['CustomerID'] != str(-9999)]
c_data.round(2).head()
fig, ax = plt.subplots(2,2, figsize=(10, 10))

plt.subplots_adjust(hspace=0.3, wspace=0.3)

ax[0,0].hist(c_data['d_as_customer'], bins=50)
ax[0,0].set_title('Days as customer')        

ax[1,0].bar(x=['i_last_30D', 'i_last_90D', 'i_last_180D', 'i_over_180D'], height=c_data[['i_last_30D', 'i_last_90D', 'i_last_180D', 'i_over_180D']].sum())
ax[1,0].set_title('No of clients grouped by days since last invoice')        

ax[0,1].hist(c_data['i_first'].dt.month)
ax[0,1].set_title('New clients (first invoice) by month')        

ax[1,1].hist(c_data['i_arrival_rate'].apply(lambda x: np.log(x+1)), bins=30)
ax[1,1].set_title('Log of days between invoices')        


plt.show()
c_data_corr = c_data.drop(['i_last_30D', 'i_last_90D', 'i_last_180D', 'i_over_180D', 
                           'a_rate_week', 'a_rate_2week', 'a_rate_month', 'a_rate_small'], axis=1).corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(c_data_corr, cmap=cmap)
plt.show()
rk = c_data.copy()

rk['t_rev'] = rk['t_rev'].apply(lambda x: np.log(x))
rk['t_units'] = rk['t_units'].apply(lambda x: np.log(x))

cols = ['d_as_customer', 't_invoices', 't_rev', 't_units', 'avg_rev', 'i_arrival_rate']

for c in cols:
    rk.loc[:,str(c)] = rk[str(c)].apply(lambda x: (x-rk[str(c)].min())/(rk[str(c)].max()-rk[str(c)].min()))
    
rk['Active'] = (3*rk['i_last_30D'] + 2*rk['i_last_90D'] + rk['i_last_180D'] - 4*rk['i_over_180D'])/10
cols.append('Active')
cols.append('CustomerID')
rk = rk[cols]
rk['Score'] = rk.iloc[:,1:rk.shape[1]].sum(axis=1)
rk = rk.loc[:,['CustomerID', 'Score']]
rk = rk.sort_values(by=['Score'], ascending=False)
rk = rk.reset_index(drop=True)

fig, ax = plt.subplots(1,2, figsize=(14, 7))

ax[0].hist(rk['Score'].values)
ax[0].set_title('Score distribution')    

ax[1].bar(x=np.arange(10), height=rk['Score'][0:10])
ax[1].set_xticks(np.arange(10))
ax[1].set_xticklabels(rk['CustomerID'][0:10].tolist())
ax[1].set_title("Top ten ranked clients")    

plt.show()
def to_sparse(data):
    d = data[['StockCode', 'CustomerID', 'Revenue']].groupby(['StockCode', 'CustomerID']).count()#[1:10000]
    d = d.reset_index()
    d = d[d['CustomerID'] != '-9999']
    piv = pd.pivot_table(d, index='CustomerID', columns='StockCode', values='Revenue')
    piv = piv.dropna(axis=1, how='all')
    print(piv.iloc[0:10,0:5])
    cols = piv.columns
    piv = piv.fillna(0)
    piv = lil_matrix(piv)
    return piv,cols
piv,cols = to_sparse(data)
def get_sims(matrix, col_index):
    
    cosine = lambda x,y: 1-spatial.distance.cosine(x,y)
    euclidean = lambda x,y: spatial.distance.euclidean(x,y)    
    pearson = lambda x,y: float(stats.pearsonr(x,y)[0])
    
    corrs = []
    col = matrix[:,col_index].todense()    
    col_nums = range(0, piv.shape[1])
    
    def calculate(i):    
        tmp_col = matrix[:,i].todense()            
        return [i, euclidean(col,tmp_col), cosine(col, tmp_col), pearson(col, tmp_col)]
        
    sims = list(map(calculate, col_nums))
        
    return pd.DataFrame(data= sims, index=cols, columns=['ColNo', 'Euclidean', 'Cosine', 'Pearson'])

product = list(cols).index('23240')
most_similar = get_sims(matrix=piv, col_index=product)
most_similar = most_similar.join(data[['StockCode', 'Description']].groupby(['StockCode', 'Description']).count())
most_similar.reset_index(inplace=True)
most_similar.sort_values(by=['Cosine'], ascending=False)[0:20]