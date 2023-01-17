# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn as sns
#3 Functions

# 3.1 Print 5 Rows for any column

def print_rows(name_column):

    return df1[name_column][0:5]

# 3.2 Get Details of the Column

def describe_column(name_column):

    return df1[name_column].describe()
print(os.listdir('../input'))
nRowsRead = 1000 # specify 'None' if want to read whole file

# Superstore.csv has 9994 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/superstore-sales/superstore.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'superstore.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.shape
df1.columns
df1.isnull().sum()
df1.head()
df1.info()
df1.describe()
print_rows('Order Date')
df1['Order Date'] = pd.to_datetime(df1['Order Date'])

describe_column('Order Date')
print_rows("Ship Date")
df1['Ship Date'] = pd.to_datetime(df1['Ship Date'])

describe_column('Ship Date')
print_rows('Ship Mode')
df1['Ship Mode'].unique()
plt.figure(figsize=(16,8))

top20states = df1.groupby('State')['Row ID'].count().sort_values(ascending=False)

top20states = top20states [:20]

top20states.plot(kind='bar', color='blue')

plt.title('Top 20 States in Sales')

plt.ylabel('Count')

plt.xlabel('States')

plt.show()
plt.figure(figsize=(16,8))

top20city = df1.groupby('City')['Row ID'].count().sort_values(ascending=False)

top20city = top20city [:20]

top20city.plot(kind='bar', color='red')

plt.title('Top 20 Cities in Sales')

plt.ylabel('Count')

plt.xlabel('Cities')

plt.show()
plt.figure(figsize=(16,8))

top20pid = df1.groupby('Product ID')['Row ID'].count().sort_values(ascending=False)

top20pid = top20pid [:20]

top20pid.plot(kind='bar', color='Green')

plt.title('Top 20 Products by Product IDs in Sales')

plt.ylabel('Count')

plt.xlabel('Product IDs')

plt.show()
plt.figure(figsize=(16,8))

top20pname = df1.groupby('Product Name')['Row ID'].count().sort_values(ascending=False)

top20pname = top20pname [:20]

top20pname.plot(kind='bar', color='Orange')

plt.title('Top 20 Products in Sales')

plt.ylabel('Count')

plt.xlabel('Products')

plt.show()
x = df1.sort_values('Profit', ascending=False)

top20 = x.head(20)

top20[['Customer Name', 'Profit']] 
sns.barplot(x = "Profit", y= "Customer Name", data=top20)  # plotting of top 20 profitable customers
plt.figure(figsize=(16,8))

df1['Segment'].value_counts().plot.bar()

# sns.countplot("Segment", data = data)           #Distribution of customer Segment

plt.title('Segment Wise Sales')

plt.ylabel('Count')

plt.xlabel('Segments')

plt.show()
plt.figure(figsize=(16,8))

df1['Ship Mode'].value_counts().plot.bar()

plt.title('Ship Mode Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Ship Modes')

plt.show()
plt.figure(figsize=(16,8))

df1['Region'].value_counts().plot.bar()

plt.title('Region Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Regions')

plt.show()
plt.figure(figsize=(16,8))

df1['Category'].value_counts().plot.bar()

plt.title('Category Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Categories')

plt.show()
plt.figure(figsize=(16,8))

df1['Sub-Category'].value_counts().plot.bar()

plt.title('Sub-Category Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Sub Categories')

plt.show()
plt.figure(figsize=(14,8))

CusCountry = pd.DataFrame({'Count' : df1.groupby(["Country","State"]).size()}).reset_index().sort_values('Count',ascending = False).head(20)

sns.barplot(x = "Country", y= "Count", hue="State", data = CusCountry.sort_values('Country'))

plt.show()
plt.figure(figsize=(16,8))

sale_category = df1.groupby(["Category","Sub-Category"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)

sns.barplot(x = "Category", hue="Sub-Category", y= "Quantity", data=sale_category)

plt.show()
df1['Order Date'] = pd.to_datetime(df1['Order Date'])      

top20Cust= df1.sort_values(['Order Date'], ascending=False).head(20)

top20Cust.loc[:,['Customer Name']]
Visit=df1.groupby('Customer ID').apply(lambda x: pd.Series(dict(visit_count=x.shape[0])))

Visit.loc[(Visit.visit_count==1)]
#Relationship between sales and profit -- use scatter plot

regionwiseSalesAndProfit = df1.groupby("Region").agg({"Sales":np.sum, "Profit": np.sum})

regionwiseSalesAndProfit

fig = plt.figure(figsize=(6,4))

ax = fig.add_subplot(111)

p = sns.scatterplot(x="Sales", y="Profit", hue=regionwiseSalesAndProfit.index, data=regionwiseSalesAndProfit) # kind="scatter")

ax.set_title("Relationship between Sales and Profit by Region")

plt.tight_layout()

plt.show()
#Year-wise sales and profit

df1["Order_Year"] = pd.to_datetime(df1["Order Date"])

df1["Year"] = df1["Order_Year"].dt.year

yearwiseSalesAndProfit = df1.groupby("Year").agg({"Sales":np.sum, "Profit": np.sum})
yearwiseSalesAndProfit
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(221)

p=sns.barplot(x=yearwiseSalesAndProfit.index,y="Profit", data=yearwiseSalesAndProfit, palette="winter", ax=ax)

ax.set_title("Year-wise Profit")

ax.set_xticklabels(p.get_xticklabels(), rotation=0)

ax = fig.add_subplot(222)

p=sns.barplot(x=yearwiseSalesAndProfit.index,y="Sales", data=yearwiseSalesAndProfit, palette="spring", ax=ax)

ax.set_title("Year-wise Sales")

ax.set_xticklabels(p.get_xticklabels(), rotation=0)

plt.tight_layout()

plt.show()