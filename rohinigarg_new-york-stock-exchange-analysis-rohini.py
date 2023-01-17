

%reset -f 

import numpy as np

import pandas as pd

# 1.2 For plotting

import matplotlib.pyplot as plt

#import matplotlib

#import matplotlib as mpl     # For creating colormaps

import seaborn as sns

import plotly.graph_objects as go 

import plotly.express as px



# 1.3 For data processing

from sklearn.preprocessing import StandardScaler

# 1.4 OS related

import os



#Split dataset

from sklearn.model_selection import train_test_split



#Class to develop kmeans model

from sklearn.cluster import KMeans



#How good is clustering?

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer

%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

plt.style.use('dark_background')

#pd.options.display.float_format = '{:.2f}'.format
#pin to directory

os.chdir("/kaggle/input/nyse")

os.listdir()  
df = pd.read_csv('fundamentals.csv', parse_dates=['Period Ending'])

df.drop('Unnamed: 0', axis=1, inplace=True)

#drop NA Columns

df.dropna(inplace=True)

dct = dict()

for col in df.columns.values:

    ret = ''

    t = '_'.join([word.capitalize() for word in col.split(' ')])

    for c in t:

        if (c >= '0' and c <= '9') or (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c in ['_']:

            ret += c

    dct[col] = ret

df.rename(columns=dct, inplace=True)

df.columns

df.head()
#drop rows where year 1215

df.For_Year = df.For_Year.astype('int64')

df.For_Year.unique()

df.drop(df[df.For_Year == 1215].index , inplace=True)

df.For_Year.unique()
fig = px.density_contour(

                         data_frame =df,

                         x = 'Gross_Profit',

                         y = 'Total_Assets',

                        )

fig.update_traces(

                  contours_coloring="fill",

                  contours_showlabels = True

                 )

#Observation : Max count is where Gross profit about 2.5B and Total assets 10B
plt.figure(figsize=(15,5))

sns.distplot(df.Total_Liabilities)

#observation lies inbetween .0 to .5
px.histogram(data_frame =df,

             x = 'Earnings_Per_Share',

             facet_col = 'For_Year',

             )

#Observations : All Year's earning lies inbetween inbetween 1-1.99 & 2-2.99

#Maximum Trcikers earned in 2013 Year
dfYear = df.groupby(['For_Year']).agg('mean').reset_index()

dfYear



px.density_heatmap(

                   data_frame =df,

                   x = 'For_Year',

                   y = 'Ticker_Symbol',

                   z = 'Treasury_Stock',  # histfunc() of this is intensity of colour

                   histfunc = 'sum' # Diverging color scale

    

                   )

#Highest Treasury stock is of XEL for all years > 200B


dfTricker_Symbol = df.groupby(['Ticker_Symbol']).agg('mean').reset_index()

dfTricker_Symbol
sns.jointplot(dfTricker_Symbol.Cash_Ratio, dfTricker_Symbol.Quick_Ratio, kind='scatter')
sns.jointplot(dfTricker_Symbol.Total_Assets, dfTricker_Symbol.Total_Liabilities, kind = 'reg') 

dfTricker_Symbol


fig = go.Figure()

fig=px.histogram(data_frame=dfTricker_Symbol,x='Gross_Profit',title='Gross Profit Analysis ')

fig.update_layout(

    

    

    yaxis_title="Count",

    title = {

             'y':0.9,

             'x':0.5,

            'xanchor': 'center'

            ,'yanchor': 'top'

        }

)





#Observation: Mostly Avg earning of Tricker lies inbetween 0-4.99B



dfTricker_Symbol.loc[(dfTricker_Symbol.Gross_Profit/1000000000 >= 2.5),'Good_gross_profit']=1

dfTricker_Symbol.loc[(dfTricker_Symbol.Gross_Profit/1000000000 < 2.5),'Good_gross_profit']=0

dfTricker_Symbol.Good_gross_profit= dfTricker_Symbol.Good_gross_profit.astype('int64')

dfTricker_Symbol.Good_gross_profit.value_counts()
# we will del  Gross_Profit because we are going to perdict clustering on this column

#Earnings_Per_Share,'Operating_Margin','Cash_Ratio'

#,'Quick_Ratio','Current_Ratio','Gross_Margin','Pretax_Margin','Pretax_Roe','Profit_Margin' because not able to find units,

#very less as compared to other data

dfTricker_Symbol.drop(

        columns = ['Ticker_Symbol', 'For_Year','Earnings_Per_Share','Gross_Profit','Operating_Margin','Cash_Ratio'

                   ,'Quick_Ratio','Current_Ratio','Gross_Margin','Pretax_Margin','Pretax_Roe','Profit_Margin'],    # List of columns to drop

        inplace = True                                   # Modify dataset here only

        )



#relationship between Good_gross_profit and Assets

sns.catplot('Good_gross_profit','Total_Assets', data = dfTricker_Symbol, kind = 'box')  

#observation : Data is more distributed where gross profit is good
sns.barplot('Good_gross_profit', 'Income_Tax',   estimator = np.mean, data = dfTricker_Symbol)

#Observation:High profit high income tax

pd.plotting.andrews_curves(dfTricker_Symbol,

                           'Good_gross_profit',

                           colormap = 'winter'       # Is there any pattern in the data?

                           )


#Copy 'Good_gross_profit' column to another variable and then drop it

#     We will not use it in clustering

y = dfTricker_Symbol['Good_gross_profit'].values

dfTricker_Symbol.drop(columns = ['Good_gross_profit'], inplace = True)
ss = StandardScaler()     # Create an instance of class

ss.fit(dfTricker_Symbol)                # Train object on the data

X = ss.transform(dfTricker_Symbol)      # Transform data

#Split dataset into train/test

X_train, X_test, _, y_test = train_test_split( X,               # np array without target

                                               y,               # Target

                                               test_size = 0.3 # test_size proportion

                                               )

X_train.shape              

X_test.shape  







#set no of clusters

clf = KMeans(n_clusters = 2,precompute_distances=True)

#Train the object over data

clf.fit(X_train)
clf.cluster_centers_

print("Shape of cluster is:",clf.cluster_centers_.shape)

print("\nLables of cluster are",clf.labels_ )                       # Cluster labels for every observation

print("\nLables size:",clf.labels_.size)                

clf.inertia_  



print("silhouette_score:",silhouette_score(X_train, clf.labels_))
#Make prediction over our test data and check accuracy

y_pred = clf.predict(X_test)

y_pred

#How good is prediction

np.sum(y_pred == y_test)/y_test.size

dx = pd.Series(X_test[:, 0])

dy = pd.Series(X_test[:,1])



fig, ax = plt.subplots(1, 1, figsize=(17,20))

sns.scatterplot(dx,dy, hue = y_pred, ax=ax)
sse = []

for i in range(10):

    #  How many clusters?

    n_clusters = i+1

    # Create an instance of class

    clf1 = KMeans(n_clusters = n_clusters)

    #  Train the kmeans object over data

    clf1.fit(X_train)

    # Store the value of inertia in sse

    sse.append(clf1.inertia_ )



#  Plot the line now

sns.lineplot(range(1, 11), sse)



#observations: Drastic change in curve for 1 and 2. After 2 it is normal so we will take no of clusters=2




visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')

visualizer.fit(X_train)        # Fit the data to the visualizer

visualizer.show()              # Finalize and render the figure



# Intercluster distance:

from yellowbrick.cluster import InterclusterDistance

visualizer = InterclusterDistance(clf)

visualizer.fit(X_train)        # Fit the data to the visualizer

visualizer.show()              # Finalize and render the figure
