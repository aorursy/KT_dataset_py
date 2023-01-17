import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import datetime as dt
df = pd.read_csv('../input/dataset/Online Retail.csv')
df
df.info()
df.isnull().sum()[df.isnull().sum() != 0]
df.nunique()
df.describe()
df.InvoiceNo = df.InvoiceNo.astype(str)

df.StockCode = df.StockCode.astype(str)

df.Description = df.Description.astype(str)

df.InvoiceDate = df.InvoiceDate.apply(lambda x: dt.datetime.strptime(str(x), '%m/%d/%Y %H:%M').date())

df.CustomerID = df.CustomerID.fillna(-1).astype(int).astype(str)

df.Country = df.Country.astype(str)
df["Amount"] = df.Quantity * df.UnitPrice
df # Data Overview
df.rename({"InvoiceDate":"Date"},axis=1,inplace=True)
df["Month"] = df.Date.apply(lambda x : x.replace(day=1))
MonthlyDF = df.groupby(by='Month').agg({'Amount' : 'sum', 'CustomerID' : 'nunique', 'InvoiceNo' : 'nunique'})

MonthlyDF.rename({"CustomerID":"TotalCustomer","InvoiceNo":"TotalTicket"},axis=1,inplace=True)
MonthlyDF["TicketSize"] = MonthlyDF.Amount / MonthlyDF.TotalTicket
MonthlyDF.reset_index(inplace=True)
MonthlyDF
def MonthlySalesPlot(ax = None, month=None):

    Data = MonthlyDF if month == None else MonthlyDF.tail(month)

    sns.lineplot(x='Month', y='Amount', data=MonthlyDF, label='Amount', ax=ax).set_title("Monthly Sales Amount" + ("" if month == None else f" for last {month} months"))

plt.figure(figsize=(12,8))

MonthlySalesPlot()
def MonthlyCustomerPlot(ax=None, month=None):

    Data = MonthlyDF if month == None else MonthlyDF.tail(month)

    sns.lineplot(x='Month', y='TotalCustomer', data=MonthlyDF, label='Total Customer', ax=ax).set_title("Monthly Customer Amount" + ("" if month == None else f" for last {month} months"))

plt.figure(figsize=(12,8))

MonthlyCustomerPlot()
def MonthlyTicketSizePlot(ax = None,month=None):

    Data = MonthlyDF if month == None else MonthlyDF.tail(month)

    plt.figure(figsize=(12,8))

    sns.lineplot(x='Month', y='TicketSize', data=MonthlyDF, label='Ticket Size',ax=ax).set_title("Monthly Ticket Size" + ("" if month == None else f" for last {month} months"))

MonthlyTicketSizePlot(ax = None)
CountrySalesDF = df.groupby(by=['Country'])[['Amount']].sum().reset_index().sort_values(by='Amount', ascending=False)
CountrySalesDF
plt.figure(figsize=(12,8))

sns.barplot(y = 'Country', x='Amount', data = CountrySalesDF).set_title("Country Sales")
def CountrySalesPlot(ax=None,top=None): # Without UK actually

    Data = CountrySalesDF.iloc[1:] if top == None else CountrySalesDF.iloc[1:].head(top)

    sns.barplot(y = 'Country', x='Amount', data = Data,ax=ax).set_title("Country Sales (excluding UK)") # Remove the first index (UK) to view the rest of the relationship

plt.figure(figsize=(12,8))

CountrySalesPlot()
ProductSalesDF = df.groupby(by=['Description'])[['Amount','Quantity']].sum() # Same Description = Same Product
ProductSalesDF
def ProductSalesPlot(ax=None):

    sns.scatterplot(x='Quantity', y='Amount', data=ProductSalesDF,ax=ax).set_title("Product Sales: Amount vs Quantity")

plt.figure(figsize=(12,8))

ProductSalesPlot()
fig, ax = plt.subplots(3, 2, figsize=(18,20))

gs = ax[1, 1].get_gridspec()

for a in ax[1:, -1]:

    a.remove()

axbig = fig.add_subplot(gs[1:, -1])



MonthlySalesPlot(ax=ax[0,0], month=12)

MonthlyCustomerPlot(ax=ax[1,0], month=12)

MonthlyTicketSizePlot(ax=ax[2,0], month=12)



ProductSalesPlot(ax=ax[0,1])

CountrySalesPlot(ax=axbig, top=15)



plt.tight_layout(pad=3.0)

fig.suptitle('\nOnline Retail Dashboard', fontsize=20)

fig.subplots_adjust(top=0.88)

sns.despine(bottom = True, left = True)



fig.savefig("Figure.png")

fig.show()