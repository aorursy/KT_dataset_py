%reset -f  

import warnings
warnings.filterwarnings("ignore")

# 1.1 Data manipulation library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
# 1.2 OS related package
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
import re
# 1.6 Set numpy options to display wide array
np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )
# Display multiple outputs from a jupyter cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# DateFrame object is created while reading file available at particular location given below

df=pd.read_csv("../input/nyse/fundamentals.csv",parse_dates = ['Period Ending'])
# Remove NaN or Null values from DataFrame

df.dropna(inplace=True)

# Displaying first 10 rows of DataFrame

df.head()
# Remove special characters from DataFrame

cols = {col : re.sub('[^A-Za-z0-9]+','_',col) for col in df.columns.values}

df.rename(columns = cols,inplace=True)

#df.columns=df.columns.str.replace(r"[^a-zA-Z\d\_]+",'_')

df.info()
# Group data by Ticker Symbols and take a mean of all numeric variables.

gr1=df.groupby('Ticker_Symbol')

gr1.agg([np.mean]).head()
# This graph showing Ticker Symbol wise gross profit

px.histogram(data_frame =df,
                    x='Ticker_Symbol',
                    y='Gross_Profit',
                    histfunc="sum",
                    template="plotly_dark"
            )
# Graph showing the relationship of Net Income and Estimated shares outstanding

px.histogram(data_frame =df,
                    y='Net_Income',
                    x='Estimated_Shares_Outstanding',
                    histfunc="sum",
                    template="plotly_dark"
            )
# Relationship between Capital expenditures and capital surplus

px.density_contour(
                   data_frame =df,
                   x = 'Capital_Expenditures',
                   y = 'Capital_Surplus',
                   template="plotly_dark"
                   )
# Relationship between net income and estimated shares outstanding using heatmap

px.density_heatmap(
                   data_frame =df,
                   x = 'Net_Income',
                   y = 'Estimated_Shares_Outstanding',
                   template="plotly_dark",
                   nbinsx = 10,             
                   nbinsy = 20
                   )
 # New column is created by extracting only day from date

df['Year']= df['Period_Ending'].dt.year

df.Year.unique()

fig=px.scatter(df,
          x = "Gross_Margin",
          y = "Profit_Margin",
          size = "Goodwill",
          range_x=[0,85],
          range_y=[0,120] ,
          animation_frame = "Year",   
          animation_group = "Ticker_Symbol",   
          color = "Ticker_Symbol"              
          )

# 5.3 The following code slows down animation
#  Ref: https://community.plotly.com/t/how-to-slow-down-animation-in-plotly-express/31309/6
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
fig.show()
sns.distplot(df.Gross_Profit,color='b')
sns.jointplot(df.Gross_Margin, df.Profit_Margin, kind = 'reg',color='b')
# Take the selected data from dataframe 

dfselecteddata = df[['Accounts_Payable','Accounts_Receivable','Gross_Profit',
               'Cost_of_Revenue','Gross_Margin','Gross_Profit','Net_Income','Profit_Margin','Total_Assets',
               'Total_Liabilities','Estimated_Shares_Outstanding']]

# New field named net_profit_loss where value is 2 if there is profit is more than or equal to 100 Cr,
# 1,if profit is less than 100 Cr and value is 0 if there is loss


dfselecteddata['Net_Profit_Loss'] =  dfselecteddata['Net_Income'].map(lambda x: 0 if x <=0  
                                     else (1 if x/10000000<100  else 2))


# Copy 'net_profit_loss' column to another variable and then drop it

y = dfselecteddata['Net_Profit_Loss'].values

dfselecteddata.drop(columns = ['Net_Profit_Loss'], inplace = True)

# Scale data using StandardScaler
    
ss = StandardScaler()                 # Create an instance of class
ss.fit(dfselecteddata)                # Train object on the data
X = ss.transform(dfselecteddata)      # Transform data
X[:5, :]                              # See first 5 rows
X_train, X_test, y_train, y_test = train_test_split( X,               # np array without target
                                               y,               # Target
                                               test_size = 0.25 # test_size proportion
                                               )
# 4.1 Examine the results

X_train.shape    

X_test.shape  
# Create an instance of modeling class
# We will have three clusters

clf = KMeans(n_clusters = 3)

# Train the object over data

clf.fit(X_train)

# So what are our clusters?

clf.cluster_centers_
clf.cluster_centers_.shape         
clf.labels_                        # Cluster labels for every observation
clf.labels_.size                   
clf.inertia_                       # Sum of squared distance to respective centriods, SSE
# For importance and interpretaion of silhoutte score, see:

silhouette_score(X_train, clf.labels_)    
# Make prediction over our test data and check accuracy

y_pred = clf.predict(X_test)
y_pred                 
np.sum(y_pred == y_test)/y_test.size
dx = pd.Series(X_test[:, 0])
dy = pd.Series(X_test[:,1])
sns.scatterplot(dx,dy, hue = y_pred)
# Scree plot:
sse = []
for i,j in enumerate(range(10)):
    
    # How many clusters?
    n_clusters = i+1
    
    # Create an instance of class
    clf1 = KMeans(n_clusters = n_clusters)
    
    # Train the kmeans object over data
    clf1.fit(X_train)
    
    # Store the value of inertia in sse
    sse.append(clf1.inertia_ )

# Plot the line now
sns.lineplot(range(1, 11), sse)
# Silhoutte plot
visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()   
# TSNE visualization and color points with the clusters discovered above

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,perplexity=100)
X_tsne = tsne.fit_transform(X_train)
sns.scatterplot(X_tsne[:,0], X_tsne[:,1], legend='full',hue=y_train)
plt.title('TSNE-visualization')