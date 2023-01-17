from datetime import datetime, timedelta

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



import chart_studio.plotly as py

import plotly.offline as pyoff

import plotly.graph_objs as go
pyoff.init_notebook_mode()


tx_data = pd.read_csv("../input/online-retail-ii-uci/online_retail_II.csv")
tx_data = tx_data.rename(columns={"Customer ID": "CustomerID"})
##tx_data.head(10)
##tx_data.info()
##tx_data.shape
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
##tx_data['InvoiceDate'].describe()
tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100*date.year + date.month)
##tx_data.head(10)
tx_data['Revenue'] = tx_data['Price'] * tx_data['Quantity']
##tx_data.groupby('InvoiceYearMonth')['Revenue'].sum()
tx_revenue = tx_data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()
##tx_revenue
plot_data = [

    go.Scatter(

        x=tx_revenue['InvoiceYearMonth'],

        y=tx_revenue['Revenue'],

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Montly Revenue'

    )
fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
tx_revenue['MonthlyGrowth'] = tx_revenue['Revenue'].pct_change()
##tx_revenue.head()
plot_data = [

    go.Scatter(

        x=tx_revenue.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],

        y=tx_revenue.query("InvoiceYearMonth < 201112")['MonthlyGrowth'],

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Montly Growth Rate'

    )



fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
##tx_data.groupby('Country')['Revenue'].sum().sort_values(ascending=False).astype(int)
tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
##tx_uk.head()
tx_monthly_active = tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()
##tx_monthly_active
plot_data = [

    go.Bar(

        x=tx_monthly_active['InvoiceYearMonth'],

        y=tx_monthly_active['CustomerID'],

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Monthly Active Customers'

    )
fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
##tx_monthly_active['CustomerID'].mean()
tx_monthly_sales = tx_uk.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()
##tx_monthly_sales
plot_data = [

    go.Bar(

        x=tx_monthly_sales['InvoiceYearMonth'],

        y=tx_monthly_sales['Quantity'],

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Monthly Total # of Order'

    )
fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
##tx_monthly_sales['Quantity'].mean()
tx_monthly_order_avg = tx_uk.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()
##tx_monthly_order_avg
plot_data = [

    go.Bar(

        x=tx_monthly_order_avg['InvoiceYearMonth'],

        y=tx_monthly_order_avg['Revenue'],

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Monthly Order Average'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
##tx_monthly_order_avg.Revenue.mean()
##tx_uk.info()
tx_min_purchase = tx_uk.groupby('CustomerID').InvoiceDate.min().reset_index()
tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']
tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)



##tx_min_purchase
tx_uk = pd.merge(tx_uk, tx_min_purchase, on='CustomerID')
##tx_uk.head()
tx_uk['UserType'] = 'New'

tx_uk.loc[tx_uk['InvoiceYearMonth']>tx_uk['MinPurchaseYearMonth'],'UserType'] = 'Existing'
tx_uk.UserType.value_counts()
##tx_uk.head()
tx_user_type_revenue = tx_uk.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()
##tx_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")
tx_user_type_revenue = tx_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")
plot_data = [

    go.Scatter(

        x=tx_user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],

        y=tx_user_type_revenue.query("UserType == 'Existing'")['Revenue'],

        name = 'Existing'

    ),

    go.Scatter(

        x=tx_user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],

        y=tx_user_type_revenue.query("UserType == 'New'")['Revenue'],

        name = 'New'

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='New vs Existing'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
tx_user_ratio = tx_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()/tx_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique() 

tx_user_ratio = tx_user_ratio.reset_index()

tx_user_ratio = tx_user_ratio.dropna()

##tx_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()
##tx_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()
plot_data = [

    go.Bar(

        x=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['InvoiceYearMonth'],

        y=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['CustomerID'],

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='New Customer Ratio'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
##tx_min_purchase.head()
unq_month_year =  tx_min_purchase.MinPurchaseYearMonth.unique()
##unq_month_year
def generate_signup_date(year_month):

    signup_date = [el for el in unq_month_year if year_month >= el]

    return np.random.choice(signup_date)
tx_min_purchase['SignupYearMonth'] = tx_min_purchase.apply(lambda row: generate_signup_date(row['MinPurchaseYearMonth']),axis=1)





tx_min_purchase['InstallYearMonth'] = tx_min_purchase.apply(lambda row: generate_signup_date(row['SignupYearMonth']),axis=1)

##tx_min_purchase.head()
channels = ['organic','inorganic','referral']
tx_min_purchase['AcqChannel'] = tx_min_purchase.apply(lambda x: np.random.choice(channels),axis=1)

tx_activation = tx_min_purchase[tx_min_purchase['MinPurchaseYearMonth'] == tx_min_purchase['SignupYearMonth']].groupby('SignupYearMonth').CustomerID.count()/tx_min_purchase.groupby('SignupYearMonth').CustomerID.count()

tx_activation = tx_activation.reset_index()

plot_data = [

    go.Bar(

        x=tx_activation.query("SignupYearMonth>201101 and SignupYearMonth<201109")['SignupYearMonth'],

        y=tx_activation.query("SignupYearMonth>201101 and SignupYearMonth<201109")['CustomerID'],

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Monthly Activation Rate'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
tx_activation_ch = tx_min_purchase[tx_min_purchase['MinPurchaseYearMonth'] == tx_min_purchase['SignupYearMonth']].groupby(['SignupYearMonth','AcqChannel']).CustomerID.count()/tx_min_purchase.groupby(['SignupYearMonth','AcqChannel']).CustomerID.count()

tx_activation_ch = tx_activation_ch.reset_index()

plot_data = [

    go.Scatter(

        x=tx_activation_ch.query("SignupYearMonth>201001 and SignupYearMonth<201012 and AcqChannel == 'organic'")['SignupYearMonth'],

        y=tx_activation_ch.query("SignupYearMonth>201001 and SignupYearMonth<201012 and AcqChannel == 'organic'")['CustomerID'],

        name="organic"

    ),

    go.Scatter(

        x=tx_activation_ch.query("SignupYearMonth>201001 and SignupYearMonth<201012 and AcqChannel == 'inorganic'")['SignupYearMonth'],

        y=tx_activation_ch.query("SignupYearMonth>201001 and SignupYearMonth<201012 and AcqChannel == 'inorganic'")['CustomerID'],

        name="inorganic"

    ),

    go.Scatter(

        x=tx_activation_ch.query("SignupYearMonth>201001 and SignupYearMonth<201012 and AcqChannel == 'referral'")['SignupYearMonth'],

        y=tx_activation_ch.query("SignupYearMonth>201001 and SignupYearMonth<201012 and AcqChannel == 'referral'")['CustomerID'],

        name="referral"

    )

    

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Monthly Activation Rate - Channel Based'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
##tx_uk.head()
df_monthly_active = tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()
tx_user_purchase = tx_uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().astype(int).reset_index()
##tx_user_purchase
##tx_user_purchase.Revenue.sum()
tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()

##tx_retention.head()
months = tx_retention.columns[2:]
##months
retention_array = []

for i in range(len(months)-1):

    retention_data = {}

    selected_month = months[i+1]

    prev_month = months[i]

    retention_data['InvoiceYearMonth'] = int(selected_month)

    retention_data['TotalUserCount'] = tx_retention[selected_month].sum()

    retention_data['RetainedUserCount'] = tx_retention[(tx_retention[selected_month]>0) & (tx_retention[prev_month]>0)][selected_month].sum()

    retention_array.append(retention_data)

    
tx_retention = pd.DataFrame(retention_array)
tx_retention.head()
tx_retention['RetentionRate'] = tx_retention['RetainedUserCount']/tx_retention['TotalUserCount']
##tx_retention
plot_data = [

    go.Scatter(

        x=tx_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],

        y=tx_retention.query("InvoiceYearMonth<201112")['RetentionRate'],

        name="organic"

    )

    

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Monthly Retention Rate'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
tx_retention['ChurnRate'] =  1- tx_retention['RetentionRate']
plot_data = [

    go.Scatter(

        x=tx_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],

        y=tx_retention.query("InvoiceYearMonth<201112")['ChurnRate'],

        name="organic"

    )

    

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Monthly Churn Rate'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()

##tx_retention.head()
new_column_names = [ 'm_' + str(column) for column in tx_retention.columns]
tx_retention.columns = new_column_names
##months
retention_array = []

for i in range(len(months)):

    retention_data = {}

    selected_month = months[i]

    prev_months = months[:i]

    next_months = months[i+1:]

    for prev_month in prev_months:

        retention_data[prev_month] = np.nan

        

    total_user_count =  retention_data['TotalUserCount'] = tx_retention['m_' + str(selected_month)].sum()

    retention_data[selected_month] = 1 

    

    query = "{} > 0".format('m_' + str(selected_month))

    



    for next_month in next_months:

        query = query + " and {} > 0".format(str('m_' + str(next_month)))

        retention_data[next_month] = np.round(tx_retention.query(query)['m_' + str(next_month)].sum()/total_user_count,2)

    retention_array.append(retention_data)

    

    
tx_retention = pd.DataFrame(retention_array)
len(months)
tx_retention.index = months
tx_retention