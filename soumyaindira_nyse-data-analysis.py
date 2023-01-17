# 1.1 Call libraries

import warnings
warnings.filterwarnings("ignore")
import re
%reset -f
# 1.2 For data manipulations
import numpy as np
import pandas as pd
import seaborn as sns
# 1.3 For plotting
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot
import plotly.express as px
# 1.4 For data processing
from sklearn.preprocessing import StandardScaler
# 1.5 OS related
import os
# 1.3 Modeling librray
# 1.3.1 Scale data
from sklearn.preprocessing import StandardScaler
# 1.3.2 Split dataset
from sklearn.model_selection import train_test_split
# 1.3.3 Class to develop kmeans model
from sklearn.cluster import KMeans
# 1.4 Plotting library
import seaborn as sns
# 1.5 How good is clustering?
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
# 1.6 Set numpy options to display wide array
np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )

#Load the data frame
stock_df1=pd.read_csv("../input/nyse/fundamentals.csv")
stock_df1.head()
stock_df1.drop('Unnamed: 0',axis=1,inplace=True)
#Replace space in column names with underscore
stock_df1.columns=stock_df1.columns.str.replace(' ','_')
#Replace special characters
stock_df1.columns=stock_df1.columns.str.replace(',','')
stock_df1.columns=stock_df1.columns.str.replace('.','')
stock_df1.columns=stock_df1.columns.str.replace('/','')
stock_df1.columns=stock_df1.columns.str.replace('-','_')
#Remove the NaN with zero
stock_df1.dropna(inplace=True)
stock_df1.head()
stock_df1.For_Year = stock_df1.For_Year.astype('int64')
stock_df1.drop(stock_df1[stock_df1.Ticker_Symbol == 'nan'].index , inplace=True)
#correlation between Total_Current_Assets and Total_Current_Liabilities
ax=sns.scatterplot(x='Total_Current_Assets',y='Total_Current_Liabilities',data=stock_df1)
#Relationship between accounts payable and accounts receivable
sns.jointplot(stock_df1.Accounts_Payable,stock_df1.Accounts_Receivable, kind = 'reg')
plt.plot(stock_df1['Current_Ratio'])
stock_df11 = stock_df1[["Ticker_Symbol", "For_Year", "Earnings_Per_Share"]]
stock_df11.For_Year = stock_df11.For_Year.astype('int64')
stock_df11.drop(stock_df11[stock_df11.For_Year == 1215].index , inplace=True)
symbols = stock_df11["Ticker_Symbol"].unique().tolist()
stock_df11.For_Year.unique()
for u in symbols[:10]:
    dates = stock_df11[(stock_df11["Ticker_Symbol"] == u)]["For_Year"]
    values = stock_df11[(stock_df11["Ticker_Symbol"] == u)]["Earnings_Per_Share"]
    plt.plot(dates.tolist(), values.tolist())
plt.legend(symbols, loc='upper left')
#Group data by mean of ticker symbol
stock_df2 = stock_df1.groupby('Ticker_Symbol').mean().reset_index()
stock_df2.head()
#Using Heat map to analyse ticker symbol wise earnings per share
#drop rows where year 1510

stock_df2.Earnings_Per_Share = stock_df2.Earnings_Per_Share.astype('float')
stock_df2.drop(stock_df2[stock_df2.For_Year == 1510].index , inplace=True)
ticker1=stock_df2[['Ticker_Symbol','For_Year','Earnings_Per_Share']]
#ax=sns.heatmap(ticker1)
#symbol=((np.asarray(ticker1['Ticker_Symbol'])))
#EPS=((np.asarray(ticker1['Earnings_Per_Share'])))
ticker1.For_Year=ticker1.For_Year.astype('int')


#result=ticker1.pivot(index='Ticker_Symbol',columns='For_Year',values='Earnings_Per_Share')
#result=result.fillna(0)
#sns.heatmap(result,fmt="",cmap='CMRmap_r')
ticker1.drop(ticker1[ticker1.For_Year == 1813].index , inplace=True)
stock_df2.For_Year=stock_df2.For_Year.astype('int')
stock_df2.drop(stock_df2[stock_df2.For_Year == 1813].index , inplace=True)
px.density_heatmap(data_frame =ticker1,
                   x = 'For_Year',
                   y = 'Ticker_Symbol',
                   z = 'Earnings_Per_Share', # histfunc() of this is intensity of colour
                   histfunc = 'sum' # Diverging color scale
                   )
sns.distplot(stock_df2.After_Tax_ROE) 
stock_df2.For_Year=stock_df2.For_Year.astype('int')
stock_df2.drop(stock_df2[stock_df2.For_Year == 1813].index , inplace=True)
px.density_heatmap(data_frame =stock_df2,
                   x = 'For_Year',
                   y = 'Ticker_Symbol',
                   z = 'Estimated_Shares_Outstanding',  # histfunc() of this is intensity of colour
                   histfunc = 'sum' # Diverging color scale
                   )
#stock_df2.For_Year.unique()
#sns.barplot('Ticker_Symbol', 'After_Tax_ROE',   estimator = np.mean, data = stock_df2)

sns.jointplot(stock_df2.After_Tax_ROE, stock_df2.Pre_Tax_ROE, kind = 'reg') 
px.histogram(data_frame =stock_df2,
                   x = 'Ticker_Symbol',
                   y = 'Total_Revenue',
                   histfunc = "sum",
                    template="plotly_dark"
                   )
# Relationship between Capital expenditures and capital surplus

px.density_contour(
                   data_frame =stock_df2,
                   x = 'Total_Liabilities',
                   y = 'Total_Liabilities_&_Equity'
                   )
#Clustering

stock_df2.info()
clust_df=stock_df2[['Accounts_Payable','Accounts_Receivable','Capital_Expenditures','Cash_Ratio','Current_Ratio','Investments','Liabilities',
                    'Total_Assets','Total_Equity','Total_Liabilities','Total_Liabilities_&_Equity','Total_Revenue','Treasury_Stock','Earnings_Per_Share','Estimated_Shares_Outstanding']]
clust_df.head()
#Create a new variable
clust_df.loc[(clust_df['Earnings_Per_Share'] > 0),'share_profit'] = 1
clust_df.loc[(clust_df['Earnings_Per_Share'] <= 0),'share_profit'] = 0

x = clust_df['share_profit'].values

clust_df.drop(columns = ['share_profit'], inplace = True)


ss = StandardScaler()                 # Create an instance of class
ss.fit(clust_df)                # Train object on the data
X = ss.transform(clust_df)      # Transform data
X[:5, :]                              # See first 5 rows
X_train, X_test, _, y_test = train_test_split( X,               # np array without target
                                               x,               # Target
                                               test_size = 0.25 # test_size proportion
                                               )
# 4.1 Examine the results

X_train.shape    


X_test.shape  
clf = KMeans(n_clusters = 2)
# 5.2 Train the object over data
clf.fit(X_train)

# 5.3 So what are our clusters?
clf.cluster_centers_
clf.cluster_centers_.shape         # (2, 7)
clf.labels_                        # Cluster labels for every observation
clf.labels_.size                   # 375
clf.inertia_                       # Sum of squared distance to respective centriods, SSE

# 5.4 For importance and interpretaion of silhoutte score, see:
# See Stackoverflow:  https://stats.stackexchange.com/q/10540
silhouette_score(X_train,clf.labels_)    # 0.20532663345078295
# 6 Make prediction over our test data and check accuracy
y_pred = clf.predict(X_test)
y_pred

# 6.1 How good is prediction
np.sum(y_pred == y_test)/y_test.size
# 7.0 Are clusters distiguisable?
#     We plot 1st and 2nd columns of X
#     Each point is coloured as per the
#     cluster to which it is assigned (y_pred)
dx = pd.Series(X_test[:, 0])
dy = pd.Series(X_test[:,1])
sns.scatterplot(dx,dy, hue = y_pred)
# 7.1 Scree plot:
sse = []
for i,j in enumerate(range(10)):
    # 7.1.1 How many clusters?
    n_clusters = i+1
    # 7.1.2 Create an instance of class
    clf1 = KMeans(n_clusters = n_clusters)
    # 7.1.3 Train the kmeans object over data
    clf1.fit(X_train)
    # 7.1.4 Store the value of inertia in sse
    sse.append(clf1.inertia_ )
# 7.2 Plot the line now
sns.lineplot(range(1, 11), sse)
#Silhoutte plot

visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()              # Finalize and render the figure

#Thank You