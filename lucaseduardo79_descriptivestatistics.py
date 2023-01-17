import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')
df
df_sumProduct=  df.groupby(['Product line'])['Quantity'].sum().to_frame().reset_index().sort_values(by='Quantity', ascending=[False])
df_sumProduct

#Categorical variable
plt.barh(df_sumProduct['Product line'],df_sumProduct['Quantity'])
plt.xlabel('Product line', fontsize=5)
plt.ylabel('Quantity', fontsize=5)
plt.title('Sales by Product line')
plt.show()
#Categorical variable
plt.pie(df_sumProduct['Quantity'],labels =df_sumProduct['Product line'], autopct='%1.1f%%', startangle=90,explode = (0.1, 0.1, 0.1,0.1, 0.1, 0.1) )
plt.title('Sales by Product line')
plt.show()
from matplotlib.ticker import PercentFormatter
def create_Pareto_chart(by_variable, quant_variable, threshold):

    total=quant_variable.sum()
    df = pd.DataFrame({'by_var':by_variable, 'quant_var':quant_variable})
    df["cumpercentage"] = quant_variable.cumsum()/quant_variable.sum()*100
    df = df.sort_values(by='quant_var',ascending=False)
    df_above_threshold = df[df['cumpercentage'] < threshold]
    df=df_above_threshold
    df_below_threshold = df[df['cumpercentage'] >= threshold]
    sum = total - df['quant_var'].sum()
    restbarcumsum = 100 - df_above_threshold['cumpercentage'].max()
    rest = pd.Series(['OTHERS', sum, restbarcumsum],index=['by_var','quant_var', 'cumpercentage'])
    df = df.append(rest,ignore_index=True)
    df.index = df['by_var']
    df = df.sort_values(by='cumpercentage',ascending=True)

    fig, ax = plt.subplots()
    ax.bar(df.index, df["quant_var"], color="C0")
    ax2 = ax.twinx()
    ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    ax.tick_params(axis="x", colors="C0", labelrotation=70)
    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")

    plt.show()
create_Pareto_chart(df_sumProduct['Product line'], df_sumProduct['Quantity'],5000)
df_UntPrice = df['Unit price']
df_UntPrice.sort_values(ascending=True)
max_price = int(df_UntPrice.max())
min_price = int(df_UntPrice.min())
interval_price = int((max_price - min_price)/8.5)
interval_price


    
frequence = []
feqlist = []

minp = min_price
while minp < max_price:
    frequence.append('{} - {}'.format(round(minp,1),round(minp+interval_price,1)))
    feqlist.append(minp) 
    minp += interval_price 
frequence    
#Numeric variable
freq_abs = pd.qcut(df_UntPrice,len(frequence),labels=frequence)

from collections import Counter
dic_freq_abs = Counter(freq_abs)
dic_freq_abs.most_common()
plt.hist(df_UntPrice, bins = feqlist)
plt.show()
plt.hist(df_UntPrice, bins = feqlist, cumulative=True)
plt.show()