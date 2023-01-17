# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

google = pd.read_csv(r'../input/googleplaystore-acsv/googleplaystore_a.csv')
google.describe()
google.info()
google.head()
google['Installs'] = google['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else x)
google['Installs'] = google['Installs'].apply(lambda x: int(str(x).replace(',', '')) if ',' in str(x) else x)

google['Price'] = google['Price'].apply(lambda x: float(str(x).replace('$', '')) if '$' in str(x) else x)

google.drop_duplicates(subset='App', inplace=True)


google['Size'] = google['Size'].apply(lambda x: str(x).replace('M', "") if 'M' in str(x) else x)
google['Size'] = google['Size'].apply(lambda x: float(str(x).replace('k', ""))/1024 if 'k' in str(x) else x)
google['Size'] = google['Size'].apply(lambda x: str(x).replace('Varies with device', '-1') if 'Varies with device' in str(x) else x)

google['Installs'] = google['Installs'].apply(lambda x: int(str(x).replace('+', '')) if '+' in str(x) else x)
google.head()
from sklearn import preprocessing


data = google[['Rating','Reviews', 'Size', 'Installs', 'Price']]
data= data.astype('float64')


data = data.dropna()
#data.info()

data_scaled = preprocessing.scale(data)
data_scaled = pd.DataFrame(data_scaled)

data_corr = data_scaled.corr()
plt.subplots(figsize=(15, 15)) 


sns.heatmap(data_corr, annot=True, vmax=1, square=True, cmap="Blues", xticklabels = ['Rating','Reviews', 'Size','Installs', 'Price'], yticklabels = ['Rating','Reviews', 'Size', 'Installs', 'Price'])

plt.title('Correlation Of Numeric Feature')

plt.show()

google1 = google.dropna()

group1 = google1.groupby('Category')
#group1.App.count().sort_values(by='App', ascending = False)
table1 = group1.App.count().sort_values(ascending = False)

table1 = pd.DataFrame(table1)

table1.head()
plt.figure(figsize = (30,10))
sns.barplot(table1.index,table1.App)
plt.xlabel("Category")
plt.ylabel("Counts")
plt.xticks(rotation = 90)
plt.title('Bar chart of the number of different types of APP',fontsize=50)

sns.set(font_scale=3)
plt.show()
table1.index[:5]
google[['Rating','Reviews', 'Size', 'Installs', 'Price']] = google[['Rating','Reviews', 'Size', 'Installs', 'Price']].astype('float64')

google.info()

google[['App', 'Category', 'Rating', 'Installs', 'Type', 'Price', 'Genres']].sort_values(by = 'Installs', ascending = False)[:30]
google1 = google.dropna()  #Delete missing values

group3 = google1.groupby('Category')
#group1.App.count().sort_values(by='App', ascending = False)
table3 = group3.Installs.sum().sort_values(ascending = False)

table3 = pd.DataFrame(table3)

table3[:10]
import pyecharts as pe

pe.configure(
    jshost='https://cdnjs.cloudflare.com/ajax/libs/echarts/3.7.2/',
    echarts_template_dir=None,
    force_js_embed=None,
    output_image=None,
    global_theme=None
)
import pyecharts
from pyecharts import Pie
pie = Pie("Total downloads for different categories", "",title_pos='center')

pie.add("categories", 
        table3.index[:10], 
        table3.Installs[:10],
        radius=[55, 60],
        label_pos='right',
        label_text_size = 7,
        label_text_color='blue',
        is_label_show=True,
        legend_orient='vertical',
        legend_pos="left",
        legend_text_size = 7
       )

pie

google1.groupby('Category').apply(lambda t: t[t.Installs==t.Installs.max()])
group2 = google1.groupby('Rating')
#group1.App.count().sort_values(by='App', ascending = False)
table2 = group2.App.count().sort_values(ascending = False)

table2 = pd.DataFrame(table2)


plt.figure(figsize = (15,15))

plt.scatter(table2.index, table2.App,s=table2.index*100, c="g", alpha=0.5, marker=r'$\clubsuit$',
            label="Luck" )

plt.title('Scatter Plot Of Rating',)
plt.xlabel("Rating")
plt.ylabel("Count Of APP")
plt.legend(loc='upper left')
plt.show()
table4 = pd.pivot_table(google1,index=["Type",],aggfunc=[np.mean,np.sum])
table4 
google_free = google[google.Type == 'Free'].dropna()
google_paid = google[google.Type == 'Paid'].dropna()

group5 = google_free.groupby('Category')
#group1.App.count().sort_values(by='App', ascending = False)
table5 = group5.Installs.sum().sort_values(ascending = False)
table5 = pd.DataFrame(table5)[:10]

group6 = google_paid.groupby('Category')
table6 = group6.Installs.sum().sort_values(ascending = False)
table6 = pd.DataFrame(table6)[:10]
table5
from pyecharts import Funnel

attr = table5.index
value = table5.Installs
funnel = Funnel("Funnel chart of various APP downloads under free conditions", title_pos='center')
funnel.add(
    "",
    attr,
    value,
    is_label_show=True,
    label_pos="inside",
    label_text_color="#fff",
    legend_orient='vertical',
    legend_pos="left",
    legend_text_size = 7
)
funnel
result = pd.concat([table5, table6], axis=1)
result.columns = ['Installs_free', 'Installs_paid']
result
result['Installs_free'][['LIFESTYLE','PERSONALIZATION','SPORTS','WEATHER']] = pd.DataFrame(group5.Installs.sum().sort_values(ascending = False))['Installs'][['LIFESTYLE','PERSONALIZATION','SPORTS','WEATHER']]
result['Installs_paid'][['NEWS_AND_MAGAZINES','SOCIAL','TRAVEL_AND_LOCAL','VIDEO_PLAYERS']] = pd.DataFrame(group6.Installs.sum().sort_values(ascending = False))['Installs'][['NEWS_AND_MAGAZINES','SOCIAL','TRAVEL_AND_LOCAL','VIDEO_PLAYERS']]
result
from pyecharts import Bar

attr = result.index
v1 = result.Installs_free
v2 = result.Installs_paid
bar = Bar("")
bar.add("free", attr, v1, is_stack=True, xaxis_rotate=90,xaxis_label_textsize=7,yaxis_label_textsize=7)
bar.add("paid", attr, v2, is_stack=True, xaxis_rotate=90,xaxis_label_textsize=7,yaxis_label_textsize=7)

bar
google.sort_values(by = 'Price', ascending=False)[0:10]
