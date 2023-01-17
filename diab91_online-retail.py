# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.getcwd()
!pip install chart_studio
from __future__ import division

from datetime import datetime, timedelta

%matplotlib inline

import matplotlib.pyplot as plt



import seaborn as sns





import chart_studio.plotly as py

import plotly.offline as pyoff

import plotly.graph_objs as go



#initiate visualization library for jupyter notebook 

pyoff.init_notebook_mode()



tx_data=pd.read_csv(r"../input/onlineretail/OnlineRetail.csv", encoding="cp1252")



tx_data.head(10)
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
tx_data['InvoiceDate'].head()
tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100*date.year + date.month)
tx_data['InvoiceYearMonth'].head()
tx_data['Revenue'] = tx_data['UnitPrice'] * tx_data['Quantity']

tx_revenue = tx_data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()

tx_revenue
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



#showing first 5 rows

tx_revenue.head()



#visualization - line graph

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
tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)



#creating monthly active customers dataframe by counting unique Customer IDs

tx_monthly_active = tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()



#print the dataframe

tx_monthly_active



#plotting the output

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
tx_monthly_active
tx_monthly_sales = tx_uk.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()



#print the dataframe

tx_monthly_sales



#plot

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
tx_monthly_sales
tx_monthly_order_avg = tx_uk.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()



#print the dataframe

tx_monthly_order_avg



#plot the bar chart

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
tx_monthly_order_avg
tx_min_purchase = tx_uk.groupby('CustomerID').InvoiceDate.min().reset_index()

tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']

tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)



#merge first purchase date column to our main dataframe (tx_uk)

tx_uk = pd.merge(tx_uk, tx_min_purchase, on='CustomerID')



tx_uk.head()



#create a column called User Type and assign Existing 

#if User's First Purchase Year Month before the selected Invoice Year Month

tx_uk['UserType'] = 'New'

tx_uk.loc[tx_uk['InvoiceYearMonth']>tx_uk['MinPurchaseYearMonth'],'UserType'] = 'Existing'



#calculate the Revenue per month for each user type

tx_user_type_revenue = tx_uk.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()



#filtering the dates and plot the result

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



#print the dafaframe

tx_user_ratio



#plot the result



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
tx_user_purchase = tx_uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().reset_index()



#create retention matrix with crosstab

tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()



tx_retention.head()



#create an array of dictionary which keeps Retained & Total User count for each month

months = tx_retention.columns[2:]

retention_array = []

for i in range(len(months)-1):

    retention_data = {}

    selected_month = months[i+1]

    prev_month = months[i]

    retention_data['InvoiceYearMonth'] = int(selected_month)

    retention_data['TotalUserCount'] = tx_retention[selected_month].sum()

    retention_data['RetainedUserCount'] = tx_retention[(tx_retention[selected_month]>0) & (tx_retention[prev_month]>0)][selected_month].sum()

    retention_array.append(retention_data)

    

#convert the array to dataframe and calculate Retention Rate

tx_retention = pd.DataFrame(retention_array)

tx_retention['RetentionRate'] = tx_retention['RetainedUserCount']/tx_retention['TotalUserCount']



#plot the retention rate graph

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



    
tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()

new_column_names = [ 'm_' + str(column) for column in tx_retention.columns]

tx_retention.columns = new_column_names



#create the array of Retained users for each cohort monthly

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

tx_retention.index = months



#showing new cohort based retention table

tx_retention
