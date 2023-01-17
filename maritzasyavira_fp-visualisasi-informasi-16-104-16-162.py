import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



df = pd.read_csv('../input/sample-sales-data/sales_data_sample.csv', encoding = 'unicode_escape')

prod=df.groupby(['PRODUCTLINE']).agg({'SALES': lambda x: x.sum() })

df1=pd.DataFrame(data = prod)

df1=df1.reset_index()

names=df1['PRODUCTLINE']

size=df1['SALES']

my_circle=plt.Circle( (0,0), 0.7, color='white')

fig, ax = plt.subplots(figsize=(14,10))

ax.pie(size, labels=names, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' }, textprops={'fontsize': 14})

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
dfclone = df

dfclone = dfclone.rename(columns={"SALES": "Total Pembelian"})

df2=pd.DataFrame(data = dfclone.groupby(['CUSTOMERNAME'])['Total Pembelian'].sum().sort_values(ascending=False))

df2=df2.reset_index()

dfbar = df2.head(10)

dfbar=dfbar.round(2)



y_pos = np.arange(len(dfbar['CUSTOMERNAME']))



fig, ax = plt.subplots(figsize=(14,10))

rects1 = ax.bar(y_pos, dfbar["Total Pembelian"], color=plt.cm.Paired(np.arange(len(dfbar))))

plt.xticks(fontsize=13, rotation=90)

plt.yticks(fontsize=13)







ax.set_ylabel('Total Pembelian', fontsize=13)

ax.set_xlabel('Nama Customer', fontsize=13)



ax.set_xticks(y_pos)

ax.set_xticklabels(dfbar['CUSTOMERNAME'])





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)



plt.show()

dfclone1=df

dfclone1=dfclone1.rename(columns={"YEAR_ID": "TAHUN"})



plt.figure(figsize=(9,6))



monthly_revenue = dfclone1.groupby(['TAHUN','MONTH_ID'])['SALES'].sum().reset_index()

monthly_revenue

sns.lineplot(x="MONTH_ID", y="SALES",hue="TAHUN", data=monthly_revenue)

plt.xlabel('Bulan')

plt.ylabel("Total Penjualan ($)")

plt.show()
df['COUNTRY'] = df['COUNTRY'].replace(['UK'], 'United Kingdom')

df['COUNTRY'] = df['COUNTRY'].replace(['USA'], 'United States of America')

count2 = df.groupby(['COUNTRY'])['SALES'].agg('sum') #sales per country

df4=pd.DataFrame(data = count2)

df4=df4.reset_index()

code = ['AUS', 'AUT', 'BEL', 'CAN', 'DNK', 'FIN', 'FRA', 'DEU', 'IRL', 'ITA', 'JPN', 'NOR', 'PHL', 'SGP', 'ESP', 'SWE', 'CHE', 'GBR', 'USA'] 

  

# Using 'Address' as the column name 

# and equating it to the list 

df4['Code'] = code 

import plotly.graph_objects as go

fig = go.Figure(data=go.Choropleth(

    locations = df4['Code'],

    z = df4['SALES'],

    text = df4['COUNTRY'],

    colorscale = 'YlOrRd',

    autocolorscale=False,

    reversescale=False,

    marker_line_color='white',

    marker_line_width=0.5,

    colorbar_tickprefix = '$',

    colorbar_title = 'Sales $',

))

fig.show()
dfclone2=df

dfclone2=dfclone2.rename(columns={"PRODUCTLINE": "KATEGORI PRODUK"})

ctry=dfclone2.groupby(['COUNTRY','KATEGORI PRODUK']).agg({'QUANTITYORDERED': lambda x: x.sum() })

df5=pd.DataFrame(data = ctry)

df5=df5.reset_index()

pivot = df5.pivot_table(index='COUNTRY', columns='KATEGORI PRODUK', values='QUANTITYORDERED')

pivot.plot.bar(stacked=True, figsize=(14,10))

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.ylabel('JUMLAH PRODUK TERJUAL', fontsize=14)

plt.xlabel('NEGARA', fontsize=14)

plt.show()