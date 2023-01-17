# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_colwidth', 100)
table1 = pd.read_csv('/kaggle/input/italian-cultural-heritage-sample-data-20152018/table1.csv',index_col=False)

table2 = pd.read_csv('/kaggle/input/italian-cultural-heritage-sample-data-20152018/table2.csv',index_col=False)

table4 = pd.read_csv('/kaggle/input/italian-cultural-heritage-sample-data-20152018/table4.csv',index_col=False)

table5 = pd.read_csv('/kaggle/input/italian-cultural-heritage-sample-data-20152018/table5.csv',index_col=False)
print(table1.info())

table1.head()
# considering all the visitors

# , paying and not paying

# , what is the average price?

table1['Average_price_paid']= np.around(table1['Euro'] / table1['Visitors_total'], decimals=2)

table1.to_csv('table1.csv')
# show the two charts: visitors trend by type

# , and average price by type 

sns.set(style="darkgrid")

sns.relplot(x="Year", y="Visitors_total",

            hue="Type", 

            kind="line", legend="full", data=table1)

sns.relplot(x="Year", y="Average_price_paid",

            hue="Type", 

            kind="line", legend="full", data=table1)
print(table2.info())

table2.head()
# compute the number of visitors by year- but store in separate dataframe for further uses

table2_Visitors_Year = table2[['Year','Visitors_total']].groupby('Year').sum().reset_index()

# remove potential ambiguity in column names

table2_Visitors_Year.columns = ['Year', 'Visitors_total_year']

# get from the original table only the columns needed for the chart

tablex = table2[['Year','Region','Visitors_total','Euro']].merge(table2_Visitors_Year,how='outer')

# add within the table the information about the average revenue by region by year

tablex['Average_price_paid']= np.around(tablex['Euro'] / tablex['Visitors_total'], decimals=1)

# add the weight for the bubble chart

tablex['Percentage_visitors'] = np.around((tablex['Visitors_total'] / tablex['Visitors_total_year']), decimals=2)*100
tablex.to_csv('tablex.csv')

tablex
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.set(style="darkgrid")

ax = sns.scatterplot(x="Average_price_paid", y="Region",



                     hue="Percentage_visitors", size="Percentage_visitors",



                     sizes=(40, 400), palette=cmap,



                     legend="brief", data=tablex[tablex['Year']==2018])

ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.set(style="darkgrid")

ax = sns.scatterplot(x="Percentage_visitors", y="Region",



                     hue="Average_price_paid", size="Average_price_paid",



                     sizes=(40, 400), palette=cmap,



                     legend="brief", data=tablex[tablex['Year']==2018])

ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

ax.set_title("Percentage visitors in 2018 by Region, vs Average price paid")
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.set(style="darkgrid")

ax = sns.scatterplot(x="Average_price_paid", y="Region",



                     hue="Percentage_visitors", size="Percentage_visitors",



                     sizes=(40, 400), palette=cmap,



                     legend="brief", data=tablex[tablex['Year']==2018])

ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

ax.set_title("Average price paid in 2018 by Region, vs Percentage visitors")
print(table4.info())

table4.head()
# create copy of the table, to remove the "TOTALS" rows from the Type column

table4_net = table4.copy()

table4_net = table4_net.set_index('Type').drop('TOTALS').reset_index()
# compute the number of visitors by year- but store in separate dataframe for further uses

table4_net_Visitors_Year = table4_net[['Year','Visitors_total']].groupby('Year').sum().reset_index()

# remove potential ambiguity in column names

table4_net_Visitors_Year.columns = ['Year', 'Visitors_total_year']

# get from the original table only the columns needed for the char

tabley = table4_net[['Year','Month','Type','Visitors_total','Euro']].groupby(['Year','Type']).sum().reset_index().merge(table4_net_Visitors_Year,how='outer')

# set year to string

tabley['Year'] = tabley['Year'].apply(str)

# add within the table the information about the average revenue by region by year

tabley['Average_price_paid']= np.around(tabley['Euro'] / tabley['Visitors_total'], decimals=1)

# add the weight for the bubble chart

tabley['Percentage_visitors'] = np.around((tabley['Visitors_total'] / tabley['Visitors_total_year']), decimals=2)*100
tabley.to_csv('tabley.csv')

tabley
sns.set(style="darkgrid")

ax = sns.scatterplot(x="Year", y="Type",



                     hue="Average_price_paid", size="Percentage_visitors",



                     sizes=(40, 400),



                     legend="brief", data=tabley)

ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

ax.set_title("Price impact on number of visitors 2015-2018")
# show the two charts: visitors trend by type

# , and average price by type 

sns.set(style="darkgrid")

sns.relplot(x="Year", y="Percentage_visitors",

            hue="Type", 

            kind="line", legend="full", data=tabley)

sns.relplot(x="Year", y="Average_price_paid",

            hue="Type", 

            kind="line", legend="full", data=tabley)
print(table5.info())

table5.head()