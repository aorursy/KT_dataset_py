# Reseting memory

%reset -f     
# 1.0 Calling libraries



# 1.1 Warnings

import warnings

warnings.filterwarnings("ignore")



# 1.2 Data manipulation library

import pandas as pd

import numpy as np

import re



# 1.3 Plotting library

import seaborn as sns

import plotly.express as px

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



# 1.4 Modeling librray

# 1.4.1 Class to develop kmeans model

from sklearn.cluster import KMeans

# 1.4.2 Scale data

from sklearn.preprocessing import StandardScaler

# 1.4.3 Split dataset

from sklearn.model_selection import train_test_split



# 1.5 How good is clustering?

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer



# 1.6 os related

import os
# 2.0 Display multiple outputs from a jupyter cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# 3.0 Set numpy options to display wide array

np.set_printoptions(precision = 3,          # Display upto 3 decimal places

                    threshold=np.inf        # Display full array

                    )
# 4.0 Seting display options

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 100)
# 5.0 Read dataset from csv files

df_fundamentals = pd.read_csv("../input/nyse-sp500-dataset/fundamentals.csv",parse_dates = ['Period Ending'],index_col='Unnamed: 0')

df_split_adj_prices = pd.read_csv("../input/nyse-sp500-dataset/prices-split-adjusted.csv")

df_securities = pd.read_csv("../input/nyse-sp500-dataset/securities.csv")
# 6.0 Exploring dataset
df_fundamentals.head()  
df_fundamentals.shape               # (1781, 78)
df_fundamentals.dtypes 
# 7.0 cleaning/renaming column names

df_fundamentals.columns = df_fundamentals.columns.str.replace(' ', '_').str.replace("'", 'n').str.replace(',', '').str.replace('/', '_to_').str.replace('.', '').str.strip().str.lower()                         

df_fundamentals.dtypes
# 8.0 Checking missing values in columns

df_fundamentals.isna().any()
# Handling missing values

df_fundamentals.loc[df_fundamentals["cash_ratio"].isnull() == True,"cash_ratio"] = 0

df_fundamentals.loc[df_fundamentals["current_ratio"].isnull() == True,"current_ratio"] = 0

df_fundamentals.loc[df_fundamentals["quick_ratio"].isnull() == True,"quick_ratio"] = 0

df_fundamentals.loc[df_fundamentals["for_year"].isnull() == True ,"for_year"] = df_fundamentals["period_ending"].dt.year

df_fundamentals.loc[df_fundamentals["earnings_per_share"].isnull() == True,"earnings_per_share"] = 0

df_fundamentals.loc[df_fundamentals["estimated_shares_outstanding"].isnull() == True,"estimated_shares_outstanding"] = 0

df_fundamentals.loc[df_fundamentals["for_year"] == 1215,"for_year"] = 2015
# 7.0 Group data by Ticker Symbols and take a mean of all numeric variables.

df_grp_by_ticker = df_fundamentals.groupby(['ticker_symbol']).mean()

df_grp_by_ticker.head()
df_after_tax_roe = df_grp_by_ticker['after_tax_roe'].sort_values(ascending = False).head(5)

df_after_tax_roe
x = df_after_tax_roe.index

y = df_after_tax_roe.values

fig = plt.figure(figsize=(14,1.5))

fig.subplots_adjust(top=4, bottom=0.01, left=0.2, right=0.99)

plt.bar(x, y)



for index, value in enumerate(y):

    plt.text(index-.2, value + 10, (value), va = 'bottom')



plt.title('Top 5 Companies: After Tax ROE', fontsize = 24)

plt.xlabel('Ticker Symbol', fontsize = 18)

plt.ylabel('Mean After Tax ROE', fontsize = 18)    

plt.yticks([])

plt.grid(b=None)
df_Cash_Ratio = df_grp_by_ticker['cash_ratio'].nlargest(5)

#most_liquid_companies = df_Cash_Ratio.nlargest(5)

#most_liquid_companies

df_Cash_Ratio
# Cash Ratio

Ticker_Symbol = df_Cash_Ratio.index

Cash_Ratio = df_Cash_Ratio.values

fig2 = px.bar(df_Cash_Ratio,

                   x = Ticker_Symbol,

                   y = Cash_Ratio,

                   #histfunc='avg',

                   text = Cash_Ratio             

                   )

#update_traces(textposition='outside')

fig2.update_layout(

    title="Most Liquid Companies",

    xaxis_title="Ticker Symbol",

    yaxis_title="Cash Ratio",

    #font=dict(

        #family="Courier New, monospace",

        #size=18,

        #color="#7f7f7f"

    #)

)
df_EPS = df_grp_by_ticker['earnings_per_share']

most_profitable_companies = df_EPS.nlargest(5).round(decimals=2)

most_profitable_companies
# EPS

Ticker_Symbol = most_profitable_companies.index

EPS = most_profitable_companies.values

fig2 = px.bar(most_profitable_companies,

                   x = Ticker_Symbol,

                   y = EPS,

                   #histfunc='avg',

                   text = EPS             

                   )

#update_traces(textposition='outside')

fig2.update_layout(

    title="Most Profitable Companies",

    xaxis_title="Ticker Symbol",

    yaxis_title="EPS",

    #font=dict(

        #family="Courier New, monospace",

        #size=18,

        #color="#7f7f7f"

    #)

)

len(df_fundamentals)
# Dropping NaNs

df_no_null = df_fundamentals.dropna(axis = 0, how ='any')

len(df_no_null)
#df_fundamentals

sns.jointplot(df_fundamentals.profit_margin, df_fundamentals.gross_profit,        kind = 'kde')

# 'scatter', 'reg', 'resid', 'kde', or 'hex'
# Dataset group by ticker symbol and taking mean of values 

df_grp_by_ts = df_fundamentals.groupby('ticker_symbol').mean().reset_index()
# Adding a column gross margin category to dataset. 

df_grp_by_ts['gross_margin_category'] = df_grp_by_ts['gross_margin'].map(lambda x : 0 if x<=54  else 1)
# Creating a copy of dataset containing only float and int columns 

df = df_grp_by_ts.select_dtypes(include = ['float64','int64']).copy()

df.drop(columns = ['for_year'], inplace = True)
# New dataframe after droping gross margin category

y = df['gross_margin_category'].values

df.drop(columns = ['gross_margin_category'], inplace = True)
# Scaling using StandardScaler

ss = StandardScaler()

ss.fit(df)

X = ss.transform(df)
# Split dataset into train/test

X_train,X_test,_,y_test = train_test_split(X,y,test_size=0.25)
# Draw skree plot

sse =[]

for i in list(range(10)):

    n_cluster = i+1

    clf = KMeans(n_clusters = n_cluster)

    clf.fit(X_train)

    sse.append(clf.inertia_ )  #append SSE value for this no. of clusters

    

sns.lineplot(range(1, 11), sse)   
# applying KMeans algo with 2 clusters

cls = KMeans(n_clusters = 2)      # instantiate KMean object

cls.fit(X_train)                  # Get info about X_train

cls.cluster_centers_.shape        # shape of cluster centres

cls.labels_                       # Cluster labels for every observation

cls.labels_.size                

cls.inertia_                      # display value of SSE
# Predict clustering for test data

y_pred = cls.predict(X_test)

y_pred
# How good is prediction?

np.sum(y_pred==y_test)/y_test.size
# Are clusters distiguisable?

dx = pd.Series(X_test[:, 0])

dy = pd.Series(X_test[:,1])

sns.scatterplot(dx,dy, hue = y_pred)
# Silhouette score for the clusters

silhouette_score(X_train, cls.labels_)
# Yellow brick for plotting Silhouette score for each  cluster

visualizer = SilhouetteVisualizer(cls, colors='yellowbrick')

visualizer.fit(X_train)

visualizer.show()
# Combining dataset

df_sectors = pd.merge(df_fundamentals,df_securities, left_on = "ticker_symbol", right_on="Ticker symbol",how="left")
df_sectors.dtypes
df_sectors['for_year'].value_counts()
df_sectors.drop(df_sectors[(df_sectors['for_year'] == 2004.0) | 

                           (df_sectors['for_year'] == 2007.0) | 

                           (df_sectors['for_year'] == 2003.0) | 

                           (df_sectors['for_year'] == 2017.0) | 

                           (df_sectors['for_year'] == 2006.0)].index,inplace=True)

df_sectors['for_year'] = df_sectors['for_year'].astype('int')
df_sectors['for_year'].value_counts()
df = df_sectors

px.histogram(df,

                      x = 'GICS Sector',

                      y = 'after_tax_roe',

                      marginal = 'box',

                      color = 'for_year', 

                      histfunc = 'avg'

           )
df = df_sectors

px.histogram(df,

                      x = 'GICS Sector',

                      y = 'earnings_per_share',

                      marginal = 'box',

                      color = 'for_year', 

                      histfunc = 'avg'

            )