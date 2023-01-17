import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import copy
import plotly.graph_objects as go
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
orderdetails = pd.read_csv('../input/ecommerce-data/Order Details.csv')
orderlist = pd.read_csv('../input/ecommerce-data/List of Orders.csv')
target = pd.read_csv('../input/ecommerce-data/Sales target.csv')
orderdetails.head()
orderdetails.isnull().sum()
print('Orderdetails info')
print(orderdetails.info())
print('Orderdetails describe')
print(orderdetails.describe())
orderlist.head()
print('Orderlist info')
print(orderlist.info())
print('Orderlist describe')
print(orderlist.describe())
orderlist.isnull().sum()
cols = orderlist.columns
df=orderlist[cols].isnull().replace({True:1, False:0})
fig = px.imshow(df,x=df.columns, y=orderlist.index, labels=dict(x="Column", y="Row", color="Missing indicator"))
fig.update_layout(title = 'Missing values map (yellow for missing value cells and blue for filled cells)')
fig.show()
orderlist = orderlist.dropna()
orders = orderdetails.merge(orderlist[['Order ID','Order Date']], left_on='Order ID',right_on='Order ID', how='inner')
orders.head()
orders_copy = copy.deepcopy(orders)
orders_copy['Date'] = orders_copy['Order Date'].apply(lambda x:x[6::]+'-'+x[3:5])
orderssub = orders_copy.groupby(['Sub-Category', 'Date']).Profit.sum().unstack().reset_index()
orderssub.head()
orderssub['Profit Summary'] = orderssub.iloc[:,1:14].sum(axis=1)
orderssub
orderssub.loc[orderssub['Profit Summary'] < 0]['Sub-Category'].unique()
fig = go.Figure(data=[go.Bar(x=orderssub[orderssub['Profit Summary'] < 0]['Sub-Category'], y=orderssub[orderssub['Profit Summary'] < 0]['Profit Summary'])])
# Customize aspect
fig.update_traces(marker_color='rgb(255,0,17)', marker_line_color='rgb(173,0,0)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Non profitable sub-categories: Loss')
fig.show()
orderssub[orderssub['Profit Summary'] < 0]
orderssub = orderssub.loc[orderssub['Profit Summary'] > 0]
orderssub = orderssub.reset_index(drop=True)
orderssub['Profit share'] = orderssub['Profit Summary']/orderssub['Profit Summary'].values.sum()*100
orderssub = orderssub.sort_values('Profit share', ascending = False)
orderssub['Cumulative share'] = orderssub['Profit share'].cumsum()
orderssub
fig = px.bar(orderssub, y='Profit Summary', x='Sub-Category', title='Profit by sub-category', color='Cumulative share')
fig.show()
orderssub['ABC'] = 0
orderssub.loc[(orderssub['Cumulative share'] <= 83), 'ABC'] = 'A'
orderssub.loc[(orderssub['Cumulative share'] <= 95)&(orderssub['Cumulative share'] >= 83), 'ABC'] = 'B'
orderssub.loc[(orderssub['Cumulative share'] >= 95), 'ABC'] = 'C'
orderssub
fig = px.pie(orderssub.groupby('ABC')['Profit Summary'].sum(), values='Profit Summary', names = orderssub.groupby('ABC')['Profit Summary'].sum().index, title = 'Profit by ABC')
fig.show()
ordersubb = orders_copy.groupby(['Sub-Category', 'Date']).Amount.sum().unstack().reset_index()
ordersubb.head()
ordersubb = ordersubb[ordersubb['Sub-Category'].isin(orderssub.loc[orderssub['Profit Summary'] > 0]['Sub-Category'].unique())]
ordersubb['Summary'] = ordersubb.iloc[:,1:14].sum(axis=1)
ordersubb = ordersubb.sort_values('Summary', ascending = False)
ordersubb
ordersubb['variation'] = 0
ordersubb['variation'] = ordersubb.iloc[:,1:13].apply(lambda x: np.std(x)/np.mean(x)*100, axis=1)
ordersubb.sort_values('variation')
fig = px.bar(ordersubb, y='Summary', x='Sub-Category', title='Sales by sub-category', color='variation')
fig.show()
import plotly.graph_objects as go
fig = go.Figure()
for i in range(ordersubb['Sub-Category'].nunique()):
    x = ordersubb.iloc[:, 1:13].columns
    fig.add_trace(go.Scatter(x = x, y = ordersubb.iloc[i, 1:13], mode = 'lines+markers', name = ordersubb['Sub-Category'].unique()[i]))
    i+=1  
fig.update_layout(title = 'Monthly Sub-categories sales')
fig.show()
ordersubb['XYZ'] = 0
ordersubb.loc[(ordersubb['variation'] <= 50), 'XYZ'] = 'X'
ordersubb.loc[(ordersubb['variation'] <= 71)&(ordersubb['variation'] >= 50), 'XYZ'] = 'Y'
ordersubb.loc[(ordersubb['variation'] >= 71), 'XYZ'] = 'Z'
ordersubb
subb = orderssub[['Sub-Category','ABC']].merge(ordersubb[['Sub-Category','XYZ']], right_on = 'Sub-Category', left_on = 'Sub-Category', how = 'inner')
subb
table = subb.groupby(['ABC','XYZ'])['Sub-Category'].unique().unstack(level=-1)
table
