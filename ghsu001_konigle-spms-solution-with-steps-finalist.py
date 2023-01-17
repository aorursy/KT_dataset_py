import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.metrics import mean_squared_error

from fbprophet import Prophet



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train_data.csv") 

pd.set_option('display.max_columns', None)
print('Columns with null values: \n', df.isnull().sum())
df['CustomerID'] = df['CustomerID'].replace(np.nan, 'Unknown', regex=True)

df.head()
df['Error'] = df['StockCode'].astype(str).str[0]



e = set()

for i in df['Error']:

    e.add(i)

print(sorted(e, reverse=True))



df = df[df.Error.apply(lambda x: x.isnumeric())]

df['StockCode'] = df['StockCode'].str.upper() 
p = set()

for i in df['UnitPrice']:

    p.add(i)

print(sorted(p, reverse=True))



df = df[(df != 0).all(axis = 1)]
q = set()

for i in df['Quantity']:

    q.add(i)

print(sorted(q, reverse=True))
df['AbsQ'] = df['Quantity'].abs()

# df = df.sort_values(['AbsQ','CustomerID','StockCode','InvoiceDate'],axis = 0, ascending = [False,True,True,True],ignore_index = True)

df = df.sort_values(['AbsQ','CustomerID','StockCode','InvoiceDate'],axis = 0, ascending = [False,True,True,True])

df = df.reset_index()



cancel=[]

for i in range(df.shape[0]):

    if df.loc[i,"Quantity"] < 0:

        j = -1

        while df.loc[i,"CustomerID"] == df.loc[i+j,"CustomerID"]:

            if i+j not in cancel:

                if df.loc[i,"StockCode"] == df.loc[i+j,"StockCode"]:

                    if df.loc[i,"Quantity"] == 0-df.loc[i+j,"Quantity"]:

                        cancel.append(i)

                        cancel.append(i+j)

                        break

            j +=1

            if i+j == df.shape[0]:

                break

df = df.drop(cancel,axis = 0)
q = set()

for i in df['Quantity']:

    q.add(i)

print(sorted(q, reverse=True))



print(df['Quantity'].quantile([0.05, 0.95]))



df = df[df['Quantity'] >= 0]

df = df[df['Quantity'] <= 30]
# df = df.sort_values(['InvoiceDate','Country','CustomerID','StockCode'],axis = 0, ascending = [True,True,True,True],ignore_index = True)

df = df.sort_values(['InvoiceDate','Country','CustomerID','StockCode'],axis = 0, ascending = [True,True,True,True])

df = df.reset_index()
dslist = []

for i in range(df.shape[0]):

    year = df.loc[i,'InvoiceDate'].split('-')[0]

    month = df.loc[i,'InvoiceDate'].split('-')[1]

    day = df.loc[i,'InvoiceDate'].split('-')[2].split(' ')[0]

    dslist.append('%s-%s-%s' % (year,month,day))

df['ds'] = dslist



df = df.drop(columns = ['Unnamed: 0',"Description","InvoiceDate","Error","AbsQ"])



print(df)
dfday = df[['Quantity', 'ds']]

dfday = dfday.groupby('ds',as_index=False)

dfday = dfday.agg({'Quantity':['sum']})

dfday.columns = ['ds','y']



print(dfday)
dfcom = dfday.copy()



dfcom['ds'] =  pd.to_datetime(dfcom['ds'], format='%Y-%m-%d')

dfcom.set_index('ds', inplace=True)

dfcom = dfcom.resample('D').asfreq().reset_index()
print(dfcom.loc[290:315,'y'])



dfcom.loc[295,'y'] = None

dfcom.loc[302,'y'] = None

dfcom.loc[309,'y'] = None



print(dfcom.loc[290:315,'y'])
print(dfcom.loc[290:315,'y'])
model = Prophet(interval_width = 0.95, weekly_seasonality = 3, yearly_seasonality = 10)

model.fit(dfcom)

dates = model.make_future_dataframe(periods = 21)

pred = model.predict(dates)

print(pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21))
train = dfcom.loc[0:347,:]

test = dfcom.loc[347:352,:]

Y_test = test.loc[:,'y']

X_test = test.loc[:,'ds']



keep = []

for i in [1,2,3]:

    for j in range(20,30):

        model = Prophet(interval_width = 0.95, weekly_seasonality = i, yearly_seasonality = j)

        model.fit(train)

        dates = model.make_future_dataframe(periods = 21)

        pred = model.predict(dates)

        error = np.sqrt(mean_squared_error(pred.loc[347:352,'yhat'],Y_test))

        keep.append([error,i,j])
keep.sort(key = lambda x: x[0])

print(keep)
model = Prophet(interval_width = 0.95, weekly_seasonality = 3, yearly_seasonality = 20)

model.fit(dfcom)

dates = model.make_future_dataframe(periods = 21)

pred = model.predict(dates)

print(pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21))
model = Prophet(interval_width = 0.95, weekly_seasonality = 1, yearly_seasonality = 26)

model.fit(dfcom)

dates = model.make_future_dataframe(periods = 21)

pred = model.predict(dates)

print(pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21))
date = pred.loc[353:373,'ds'].astype(str).tolist()

predictions = pred.loc[353:373,'yhat'].astype(int).tolist()



Submission = pd.DataFrame({ 'Date': date,

                            'Quantity': predictions })

Submission.to_csv("SubmissionCheckerRe.csv", index=False)