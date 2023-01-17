import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sales = pd.read_csv('/kaggle/input/supermarket_sales.csv')

sales.head()
sales['date'] = pd.to_datetime(sales['Date'])

sales['date'].dtype

type(sales['date'])

sales['date'] = pd.to_datetime(sales['date'])



sales['day'] = (sales['date']).dt.day

sales['month'] = (sales['date']).dt.month

sales['year'] = (sales['date']).dt.year



sales['Time'] = pd.to_datetime(sales['Time'])

sales['Hour'] = (sales['Time']).dt.hour    #type(sales['Time'])



sales['Hour'].unique()



sales.describe()
sales.head()
import plotly.express as px

import plotly.graph_objects as go



Gender = sales.groupby('Gender', as_index=False).agg({"Quantity":"count"})



fig = px.bar(Gender, x='Gender', y='Quantity', color='Gender',

             labels={'Quantity':'Jumlah Pembelian'}, height=400)



fig = go.Figure(

    fig,

    layout_title_text="Perbedaan Jumlah Pembelian antara Female dan Male"

)

fig.show()
import plotly.express as px



fig = px.box(sales, x="Branch", y="Rating", color="Branch",notched=True, title="Rating di tiap Branch")

fig.show()
genderCount  = sns.lineplot(x="Hour",  y = 'Quantity', data =sales).set_title("Product Sales per Hour")

import plotly.graph_objects as go



#revenue = sales.groupby('Product line', as_index=False).agg({"gross income":"sum"})

#revenue.head()

labels = ['Electronic accessories', ['Fashion accessories'], ['Food and beverages'], ['Health and beauty'], ['Home and lifestyle']]

values = [2587.5015, 2585.9950, 2673.5640, 2342.5590, 2564.8530]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)], layout_title_text="Revenue per Product")

fig.show()
sns.boxenplot(y = 'Product line', x = 'Quantity', hue = 'month', data=sales, palette="Set1").set_title("Penjualan Product tiap Branch")
sns.countplot(x="Payment", hue = "Branch", data =sales).set_title("Tipe Pembayaran tiap Branch")