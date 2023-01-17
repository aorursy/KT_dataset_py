# Worked done by Satish Khangar : NewYork Stock Exchange Dt 14/06/2020



%matplotlib inline

%reset -f

# for numerical operations

import numpy as np

# to store and analysis data in dataframes

import pandas as pd

import seaborn as sns

# For plotting

import matplotlib.pyplot as plt

import matplotlib

import plotly.express as px

# For data processing-scale data

from sklearn.preprocessing import StandardScaler

# OS related

import os

import datetime



# Split dataset

from sklearn.model_selection import train_test_split

# Class to develop kmeans model

from sklearn.cluster import KMeans

# How good is clustering?

from sklearn.metrics import silhouette_score



import warnings

#just to suppress warning for max plots of 20

plt.rcParams.update({'figure.max_open_warning': 0}) 

warnings.filterwarnings("ignore", category=DeprecationWarning) 

warnings.filterwarnings("ignore", category=FutureWarning) 
# Display output not only of last command but all commands in a cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# Set pandas options to display results

pd.options.display.max_rows = 1000

pd.options.display.max_columns = 1000
# Go to folder containing data file

#os.chdir("C:\\Users\\satish\\Desktop\\NewYarkStockExchange")

os.chdir("../input/nyse/")

os.listdir()            # List all files in the folder
#1. Read the securities.csv



df=pd.read_csv("securities.csv")

#df

# 2. Rename columns by replacing spaces in column_names with underscore and also remove other symbols

#         from column names such as: , (comma), . (full-stop), / (backslash) etc. That is clean the column names and

#         assign these new names to your dataset.



df.columns = df.columns.str.strip().str.replace(' \ ', '_').str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace(':', '').str.replace('\'', '').str.replace('\,', '').str.replace('\.', '').str.replace('&', '_')

df.columns = df.columns.str.strip().str.replace('__', '_').str.replace('/', '_')

df.columns = df.columns.str.strip().str.replace('__', '_').str.replace('___', '_')

df.columns



# Remove NaN Rows /Columns

df=df.dropna()

df.reset_index(drop=True, inplace=True)

# Verify whether all NaN's has removed or not

df.columns[df.isnull().any()] 

# Verify (columnwise) whether all NaN's  has removed or not

df.index[df.isnull().any(axis=1)] 

df.head()

#df
# 5.0 Draw a normal bar plot  scatter

px.bar(df,

          x = "Ticker_symbol",

          y = "CIK",

   

          )

#  Plot scatter plot, CIK for GICS_Sector

df['month'] = pd.DatetimeIndex(df['Date_first_added']).month

df['year'] = pd.DatetimeIndex(df['Date_first_added']).year



#df

#df.groupby(['GICS_Sector'])['GICS_Sub_Industry'].count()

#df.groupby(['GICS_Sector','GICS_Sub_Industry'])['Security'].count()



px.scatter(df,

          x = "GICS_Sector",

          y = "CIK",

          #range_x=[0,10],

         # range_y=[0,900000000] ,

          animation_frame = "year",  # Animate/show scatter plot

                        

          animation_group = "GICS_Sector",   # Identify which circles match which ones across

           color="GICS_Sector", hover_name="GICS_Sector",size_max=25

                                    

          )

#  Plot scatter plot, CIK for GICS_Sub_Industry

px.scatter(df,

          x = "GICS_Sub_Industry",

          y = "CIK",

          #range_x=[0,10],

         # range_y=[0,900000000] ,

          animation_frame = "year",  # Animate/show scatter plot

                        

          animation_group = "GICS_Sub_Industry",   # Identify which circles match which ones across

           color="GICS_Sub_Industry", hover_name="GICS_Sub_Industry",size_max=10,title="NewYork Stock Exchange GICS Sub Industry Analysis"

           

                                    

          ) 
#1. Read in 'fundamentals.csv' file

dt=pd.read_csv("fundamentals.csv")

dt

# 2. Rename columns by replacing spaces in column_names with underscore and also remove other symbols

#   from column names such as: , (comma), . (full-stop), / (backslash) etc. That is clean the column names and

#   assign these new names to your dataset.



dt.columns = dt.columns.str.strip().str.lower().str.replace(' \ ', '_').str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace(':', '').str.replace('\'', '').str.replace('\,', '').str.replace('\.', '').str.replace('&', '_')

dt.columns = dt.columns.str.strip().str.replace('__', '_').str.replace('/', '_')

dt.columns = dt.columns.str.strip().str.replace('__', '_').str.replace('___', '_')



dt.columns
# Remove NaN Rows /Columns

dt=dt.dropna()

dt.reset_index(drop=True, inplace=True)

# Verify whether all NaN's has removed or not

dt.columns[dt.isnull().any()] 

# Verify (columnwise) whether all NaN's  has removed or not

dt.index[dt.isnull().any(axis=1)] 

#dt_filtered=dt

#dt_filtered.head()
#Take Year and quarter data for analysis

dt['year'] = pd.DatetimeIndex(dt['period_ending']).year

dt['quarter'] =  pd.DatetimeIndex(dt['period_ending']).month 



#Define the function to return quarter from month 

def month(x):

    if 0 < x <= 3:

        return "Q1"            # Quarter 1

    if 3 < x <= 6:

        return "Q2"            # Quarter 2

    if 6 < x <= 9:

        return "Q3"            # Quarter 3

    if 9 < x <= 12:

        return "Q4"            # Quarter 4



dt['quarter'] = dt['quarter'].map(lambda x : month(x))   # Which quarter clicked

dt

#Plot Density of ticker_symbol Goodwill

ax= sns.distplot(dt.goodwill)

ax.set(# xlim =(0,5.0),                     #  sns.distplot does not have **kwargs

        xlabel= "goodwill of ticker_symbol",

        ylabel = "Denity",

        title= "Density of goodwill"

      #  xticks = list(range(0,5,0.2))

        )



#Plot Density of ticker_symbol cash_ratio

ax= sns.distplot(dt.cash_ratio)

ax.set( #xlim =(0,1),                     #  sns.distplot does not have **kwargs

        xlabel= "cash_ratio of ticker_symbol",

        ylabel = "Denity",

        title= "Density of cash_ratio"

       #xticks = list(range(0,1200,100))

        )
#  Plot scatter plot, a cash_ratio of ticker_symbol



px.scatter(dt,

          x = "ticker_symbol",

          y = "cash_ratio",

          #range_x=[0,10],

         # range_y=[0,900000000] ,

          animation_frame = "year",  # Animate/show scatter plot

                        

          animation_group = "ticker_symbol",   # Identify which circles match which ones across

           color="ticker_symbol", hover_name="ticker_symbol",size_max=25

                                    

          )

# Box Plot to see relationsheep of quick_ratio and ticker symbol

sns.boxplot(x = 'ticker_symbol',      

            y = 'quick_ratio',                

            data = dt

            )

#Group data by Ticker Symbols and take a mean of all numeric variables

dc=dt.copy()

dc_grouped=dc.groupby(['ticker_symbol'], as_index=True).pipe(lambda group:group.mean()).reset_index()

dc_grouped.head()

dc_grouped.shape
#Heatmap of ticker_symbol vs Year on earnings_per_share

grouped = dc.groupby(['ticker_symbol','year'])

dg_EPS = grouped['earnings_per_share'].sum().unstack()

#dg_EPS

sns.heatmap(dg_EPS)

#Yearwise Quarterwise Gross-Profit 

grp = sns.barplot(x = 'quarter',

            y = 'gross_profit',

            hue = 'year',       

            estimator = np.mean,

            ci = 95,

            orient ='v',

            data =dc)



# Different type plots and relationship

# Distribution plot. 

# Distribution plot for earnings_per_share. 

sns.distplot(dc.earnings_per_share)                 

# Distribution plot for gross_profit.

sns.distplot(dc.gross_profit)                        # Almost symmetric  



#Distribution plot for quick_ratio.

sns.distplot(dc.quick_ratio)
# Joint Plot quick_ratio Vs  current_ratio

sns.jointplot(dc.quick_ratio, dc.current_ratio, kind = 'reg')   # Strong correlation

 
#Joint Plot goodwill Vs  profit_margin

sns.jointplot(dc.goodwill, dc.profit_margin,        kind = 'reg')   # Strong correlation
 # Relationship of 'profit_margin' to earnings_per_share

sns.catplot('profit_margin','earnings_per_share', data = dc, kind = 'box')     

#pd.plotting.andrews_curves(dc,

#                           'profit_margin',

#                           colormap = 'winter'       # Is there any pattern in the data?

#                           )
 # Relationship of 'profit_margin' to gross_profit

sns.catplot('gross_profit','profit_margin', data = dc, kind = 'box')   
# Avg profit_margin vs earnings_per_share



sns.barplot('profit_margin','earnings_per_share',   estimator = np.mean, data = dc) 
# Avg gross_profit vs profit_margin

sns.barplot('gross_profit','profit_margin', estimator = np.mean, data = dc)
#  total_liabilities vs total_equity



sns.barplot('total_liabilities','total_equity',   estimator = np.mean, data = dc) 
# 6 .Normalise the data using sklearn's StandardScaler()



ss = StandardScaler()     # Create an instance of class

#dc

#Copy 'ticker_symbol' column to another variable and then drop it

y = dc['ticker_symbol'].values

#drop the column which will not used 

dc.drop(columns = ['ticker_symbol','period_ending','quarter'], inplace = True)



ss.fit(dc)                # Train object on the data

X = ss.transform(dc)      # Transform data

X[:5, :]                  # See first 5 rows



# Split dataset into train/test

X_train, X_test, _, y_test = train_test_split( X,               # np array without target

                                               y,               # Target

                                               test_size = 0.25 # test_size proportion

                                               )

# Examine the results

X_train.shape              

X_test.shape  
# 7) Perform clustering and check Silhoutte score.



clf = KMeans(n_clusters = 2)

# Train the object over data

clf.fit(X_train)



# what are our clusters?

clf.cluster_centers_

clf.cluster_centers_.shape         # (2, 78)

clf.labels_                        # Cluster labels for every observation

clf.labels_.size                   # 974

clf.inertia_                       # Sum of squared distance to respective centriods, SSE = 56988.21216328321



silhouette_score(X_train, clf.labels_)    # 0.7733123436865416
#  Make prediction over our test data and check accuracy



y_pred = clf.predict(X_test)

y_pred

# How good is prediction

np.sum(y_pred == y_test)/y_test.size



#     Are clusters distiguisable?

#     We plot 1st and 2nd columns of X

#     Each point is coloured as per the

#     cluster to which it is assigned (y_pred)

dx = pd.Series(X_test[:, 0])

dy = pd.Series(X_test[:,1])

sns.scatterplot(dx,dy, hue = y_pred)



# draw a Scree Plot and decide how many clusters are required?

# Scree plot:

sse = []

for i,j in enumerate(range(10)):

    # How many clusters?

    n_clusters = i+1

    # Create an instance of class

    clf = KMeans(n_clusters = n_clusters)

    # Train the kmeans object over data

    clf.fit(X_train)

    # Store the value of inertia in sse

    sse.append(clf.inertia_ )



# 7.2 Plot the line now

sns.lineplot(range(1, 11), sse)



#Maximum distance is 2 points, so only 2 Clusters are required.
#8) Perform TSNE visualization (of the dataset) and color points with the clusters discovered above.





from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,perplexity=10.0)

X_tsne = tsne.fit_transform(X)

sns.scatterplot(X_tsne[:,0], X_tsne[:,1], legend='full')

plt.title('TSNE-visualization')

#plt.legend(loc=2)

#plt.show()
