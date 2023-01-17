import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import datetime as dt

import matplotlib.pyplot as plt

import squarify

from scipy.stats import norm

from sklearn.preprocessing import MinMaxScaler

#

from sklearn.cluster import KMeans

#

import plotly.offline as pyo 

import plotly.graph_objs as go

import plotly.figure_factory as ff

#

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/ecommerce-data/data.csv",encoding = 'unicode_escape')

data.head()
# Count of Countries 

data["Country"].value_counts()
# Check missing values

data.isnull().sum()
#Total Price

data['TotalPrice'] = data['UnitPrice'] * data['Quantity']

data.head()
# Total Spending of Countries

data_country = data.groupby("Country").agg({'TotalPrice': lambda x: x.sum()})
# Drop Unnecessary Countries for Visualization 

data_country.drop(["RSA","Unspecified","EIRE","European Community","Channel Islands"],axis=0,inplace=True)

data_country.head()
price = []

for i in range(len(data_country["TotalPrice"])):

    price.append(data_country["TotalPrice"][i])



country_price = pd.DataFrame(index=["AUS","AUT","BHR","BEL","BRA","CAN","CYP","CZE","DNK","FIN","FRA","DEU","GRC","HKG","ISL","ISR",

                                    "ITA","JPN","LBN","LTU","MLT","NLD","NOR","POL","PRT","SAU","SGP","ESP","SWE","CHE","USA",

                                    "ARE","GBR"],columns=["TotalPrice","country"])

country_price["country"] = data_country.index

country_price["TotalPrice"] = price

country_price.head()
worldmap = [dict(type = 'choropleth', locations = country_price['country'], locationmode = 'country names',

                 z = country_price['TotalPrice'], autocolorscale = True, reversescale = False, 

                 marker = dict(line = dict(color = 'rgb(180,180,180)', width = 0.5)), 

                 colorbar = dict(autotick = False, title = 'Total Price'))]



layout = dict(title = 'Total Price For Each Country', geo = dict(showframe = False, showcoastlines = True, 

                                                                projection = dict(type = 'Mercator')))



fig = dict(data=worldmap, layout=layout)

pyo.iplot(fig, validate=False)
data.head()
data.shape
# Change Data Type:

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])



# Adjust today:

today = dt.datetime(2012,1,1)

print(today)



# Bigger than zero and just UK

data = data[data['Quantity'] > 0]

data = data[data['TotalPrice'] > 0]

data = data[data["Country"] == "United Kingdom"]

data.shape
data.info()
# Recency and Monetary 

data_x = data.groupby('CustomerID').agg({'TotalPrice': lambda x: x.sum(),

                                        'InvoiceDate': lambda x: (today - x.max()).days})

data_x.head()
# Dataset is basis on StockCode    

data_y = data.groupby(['CustomerID','InvoiceNo']).agg({'TotalPrice': lambda x: x.sum()})

data_y.head(20)
# Find Frequency

data_z = data_y.groupby('CustomerID').agg({'TotalPrice': lambda x: len(x)})

data_z.head()
# RFM Dataframe

rfm_table= pd.merge(data_x,data_z, on='CustomerID')



# Change Column Name

rfm_table.rename(columns= {'InvoiceDate': 'Recency',

                          'TotalPrice_y': 'Frequency',

                          'TotalPrice_x': 'Monetary'}, inplace= True)

rfm_table.head()
#Frequency bulma

def FScore(x,p,d):

    if x <= d[p][0.20]:

        return 0

    elif x <= d[p][0.40]:

        return 1

    elif x <= d[p][0.60]: 

        return 2

    elif x <= d[p][0.80]:

        return 3

    else:

        return 4



quantiles = rfm_table.quantile(q=[0.20,0.40,0.60,0.80])

quantiles = quantiles.to_dict()

rfm_table['Freq_Tile'] = rfm_table['Frequency'].apply(FScore, args=('Frequency',quantiles,))



#Recency 

rfm_table = rfm_table.sort_values('Recency',ascending=True)

rfm_table['Rec_Tile'] = pd.qcut(rfm_table['Recency'],5,labels=False)



#Monetary 

rfm_table['Mone_Tile'] = pd.qcut(rfm_table['Monetary'],5,labels=False)



# instead of zero, plus 1 

rfm_table['Rec_Tile'] = rfm_table['Rec_Tile'] + 1

rfm_table['Freq_Tile'] = rfm_table['Freq_Tile'] + 1

rfm_table['Mone_Tile'] = rfm_table['Mone_Tile'] + 1



# Add to dataframe

rfm_table['RFM Score'] = rfm_table['Rec_Tile'].map(str) + rfm_table['Freq_Tile'].map(str) + rfm_table['Mone_Tile'].map(str)

rfm_table.head()
rfm_table[rfm_table['RFM Score'] == '555'].sort_values('Monetary', ascending=False).head()
#Customers who's recency value is low

rfm_table[rfm_table['Rec_Tile'] <= 2 ].sort_values('Monetary', ascending=False).head()
#Customers who's recency, frequency as well as monetary values are low 

rfm_table[rfm_table['RFM Score'] == '111'].sort_values('Recency',ascending=False).head()
#Customers with high frequency value



rfm_table[rfm_table['Freq_Tile'] >= 3 ].sort_values('Monetary', ascending=False).head()
# Calculate RFM_Score

rfm_table['RFM_Sum'] = rfm_table[['Freq_Tile','Rec_Tile','Mone_Tile']].sum(axis=1)

rfm_table.head()
# Define rfm_level function

def rfm_level(df):

    if df['RFM_Sum'] >= 9:

        return 'Can\'t Loose Them'

    elif ((df['RFM_Sum'] >= 8) and (df['RFM_Sum'] < 9)):

        return 'Champions'

    elif ((df['RFM_Sum'] >= 7) and (df['RFM_Sum'] < 8)):

        return 'Loyal'

    elif ((df['RFM_Sum'] >= 6) and (df['RFM_Sum'] < 7)):

        return 'Potential'

    elif ((df['RFM_Sum'] >= 5) and (df['RFM_Sum'] < 6)):

        return 'Promising'

    elif ((df['RFM_Sum'] >= 4) and (df['RFM_Sum'] < 5)):

        return 'Needs Attention'

    else:

        return 'Require Activation'

# Create a new variable RFM_Level

rfm_table['RFM_Level'] = rfm_table.apply(rfm_level, axis=1)

# Print the header with top 5 rows to the console

rfm_table.head()
rfm_table["RFM_Level"].value_counts()
# Calculate average values for each RFM_Level, and return a size of each segment 

rfm_level_agg = rfm_table.groupby('RFM_Level').agg({

    'Recency': 'mean',

    'Frequency': 'mean',

    'Monetary': ['mean', 'count']}).round(1)

# Print the aggregated dataset

print(rfm_level_agg)
rfm_level_agg.columns = rfm_level_agg.columns.droplevel()

rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']

#Create our plot and resize it.

fig = plt.gcf()

ax = fig.add_subplot()

fig.set_size_inches(16, 9)

squarify.plot(sizes=rfm_level_agg['Count'], 

              label=['Can\'t Loose Them',

                     'Champions',

                     'Loyal',

                     'Needs Attention',

                     'Potential', 

                     'Promising', 

                     'Require Activation'], alpha=.6 )

plt.title("RFM Segments",fontsize=18,fontweight="bold")

plt.axis('off')

plt.show()
plt.figure(figsize=(12,10))

# Plot distribution of R

plt.subplot(3, 1, 1); sns.distplot(rfm_table['Recency'],fit=norm)

# Plot distribution of F

plt.subplot(3, 1, 2); sns.distplot(rfm_table['Frequency'],fit=norm)

# Plot distribution of M

plt.subplot(3, 1, 3); sns.distplot(rfm_table['Monetary'],fit=norm)

# Show the plot

plt.show()
clustering_fm = rfm_table[['Recency',"Frequency","Monetary"]].copy()

clustering_fm.head()
min_max_scaler = MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(clustering_fm)

data_scaled2 = pd.DataFrame(x_scaled)
data_scaled2.head()
wscc = []

for i in range(1,15): 

    kmeans = KMeans(n_clusters=i, init="k-means++",random_state=0)

    kmeans.fit(data_scaled2)

    wscc.append(kmeans.inertia_)  



plt.plot(range(1,15),wscc,marker="*",c="black")

plt.title("Elbow plot for optimal number of clusters")
kmeans = KMeans(n_clusters = 4, init='k-means++', n_init =10,max_iter = 300)

kmeans.fit(data_scaled2)

pred = kmeans.predict(data_scaled2)
np.unique(kmeans.labels_)
from sklearn.metrics import silhouette_score

score = silhouette_score (data_scaled2, kmeans.labels_)

print("Score = ", score)
y_kmeans = kmeans.predict(data_scaled2)
y_kmeans[:4]
# Count of Clusters

d_frame = pd.DataFrame(clustering_fm)

d_frame['cluster'] = y_kmeans

d_frame['cluster'].value_counts()
d_frame.head()
d_frame.groupby('cluster').mean()
from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules
data_apriori = data[data['Country']=='United Kingdom']

data_apriori.head()
data_apriori["Description"].nunique()
# Which Product and Their Count 

data_apr = data_apriori.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')

data_apr.head()
def num(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1



basket_new = data_apr.applymap(num)

basket_new.head()
from mlxtend.frequent_patterns import fpgrowth

rule_fp = fpgrowth(basket_new, min_support=0.02, use_colnames=True)

rule_fp
items = apriori(basket_new, min_support=0.02, use_colnames=True)

items
rule = association_rules(items, metric="lift", min_threshold=1)

rule