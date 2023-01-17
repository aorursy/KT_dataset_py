"""
Last amended: 20th June, 2020

Objectives:

          1. Understanding Clustering
          2. Working with multiple datasets    


"""
# 1.0 Call libraries
#%reset -f                       # Reset memory
# 1.1 Data manipulation library
import pandas as pd
import numpy as np
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

# 1.3 For plotting
import matplotlib.pyplot as plt
import matplotlib
# Install as: conda install -c plotly plotly 
import plotly.express as px


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
# 1.1 Display multiple outputs from a jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )
#1) Read in 'fundamentals.csv' file
df = pd.read_csv("../input/nyse/fundamentals.csv")
#aset & display dtypes
#ad = pd.read_csv("../input/uncover/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv")
# 2) Rename columns by replacing spaces in column_names with underscore and also remove other symbols
# from column names such as: , (comma), . (full-stop), / (backslash) etc. That is clean the column names and
# assign these new names to your dataset.

cv = df.columns.values # Copy column names to another DF
cv # Display new DF 
cv[0] = 'Serial No' #Rename Column Header - 1 as this is blank
df.columns.values[0] = 'Serial No' # Rename Column Header - 1 as this is blank
cv[0]
#2.1 Clean Column Names by replacing/removing special characters
j = 0
for i in cv:
    cv[j] = re.sub(' ', '_', cv[j]) # Replace space with _
    cv[j] = re.sub('\'','', cv[j])  # Replace apostrophe with blank
    cv[j] = re.sub('[*|\(\)\{\}]','', cv[j]) # Replace special characters
    cv[j] = re.sub('/','_', cv[j])    # Replace / with _
    cv[j] = re.sub('&','_', cv[j])    # Replace & with _
    cv[j] = re.sub('-','_', cv[j])    # Replace - with _    
    cv[j] = re.sub('\.','', cv[j])    # Replace . with _
    cv[j] = re.sub('[,]','', cv[j])   # Replace , with blank         
    cv[j] = re.sub('__.','_', cv[j])  # Replace multiple _ with single _          
    j = j + 1

# Show cleaned column names
cv
# Make a disctionary of Old & new column names
y = dict(zip(df.columns.values, cv))
y

# Replace the column names
df.rename(
         y,
         inplace = True,
         axis = 1             # Note the axis keyword. By default it is axis = 0
         )

# Show the new column names
df.columns.values
# 3) Group data by Ticker Symbols and take a mean of all numeric variables.
# Create the group on Ticker_Symbol and show the group properties
type(df)
gr1 = df.groupby('Ticker_Symbol').agg(['mean'])
gr1
gr1.shape

gr1.drop(columns = ['Serial_No'], inplace = True)
gr1
df["EPS_bins"] = pd.cut(
                       df['Earnings_Per_Share'],
                       bins = 3,           # Else devise your bins: [0,20,60,110]
                       labels= ["l", "m", "h"]
                      )
df
df.EPS_bins.unique()
df["GP_bins"] = pd.cut(
                       df['Gross_Profit'],
                       bins = 3,           # Else devise your bins: [0,20,60,110]
                       labels= ["l", "m", "h"]
                      )
df
df.EPS_bins.unique()
# 4) Using ungrouped and grouped data, perform visual analysis of data using seaborn and plotly-express

sns.distplot(df.Gross_Profit)
sns.distplot(df.Earnings_Per_Share)                  # Distribution plot. Almost symmetric

sns.distplot(df.Gross_Profit)  
sns.distplot(df.Total_Revenue)                # Almost symmetric

sns.boxplot(x = 'GP_bins',       # Discrete
            y = 'Total_Revenue',                 # Continuous
            data = df,
            notch = True
            )

sns.boxplot(x = 'GP_bins',       # Discrete
            y = 'Earnings_Per_Share',                 # Continuous
            data = df,
            notch = True
            )

sns.boxplot(x = 'EPS_bins',       # Discrete
            y = 'Gross_Profit',                 # Continuous
            data = df,
            notch = True
            )

sns.distplot(gr1.Gross_Profit)

sns.distplot(gr1.Earnings_Per_Share)
#sns.barplot('Earnings_Per_Share','Gross_Profit',  estimator = np.mean, data = df) # Avg GRE score of admit vs non-admit


sns.jointplot(df.Earnings_Per_Share,
              df.Gross_Profit,
              kind = "kde"
              )
sns.jointplot(df.Earnings_Per_Share,
              df.Gross_Profit,
              kind = "hex"
              )
sns.jointplot(df.Earnings_Per_Share,
              df.Gross_Profit,
              kind = "reg"
              )
#Plotly Express
fig = px.density_contour(
                   data_frame =df,
                   x = 'Gross_Profit',
                   y = 'Earnings_Per_Share',
                   )

fig.show() 
fig = px.histogram(data_frame =df,
                    x ='EPS_bins',
                   nbins =20,
                   template="plotly_dark", # Available themes: ["plotly", "plotly_white", "plotly_dark",
                                           #     "ggplot2", "seaborn", "simple_white", "none"]
                                           # https://plotly.com/python/templates/
                   width = 10,    # in inches in interval [10, inf]
                   height = 10    # in interval [10,inf]
            )
fig.show() 
fig = px.density_heatmap(
                   data_frame = df,
                   x = 'Gross_Profit',
                   y = 'Earnings_Per_Share',
                   )
fig.show()
px.bar(data_frame=df,
                x='EPS_bins',
                y = 'Gross_Profit',
                template='plotly_white'

      )
#     5) Dataset has a number of NaNs. Either remove rows that have NaNs or remove columns that have NaNs.
#         So that dataset has no nulls.
'NaN' in df
df2 = df.dropna(axis=1, how="any")
df2.shape

# List of Numeric Columns
num_columns = df.select_dtypes(include = ['float64']).copy()

num_columns.head()
type(num_columns)
# Create a DF of Six columns for the clustering
nc = num_columns.iloc[:, 0:6]
nc
# 6) Normalise the data using sklearn's StandardScaler()
ss = StandardScaler()     # Create an instance of class
ss.fit(nc)                # Train object on the data
nc.shape
X = ss.transform(nc)      # Transform data
X[:5, :]                  # See first 5 rows
X.shape
y = nc
X_train, X_test, _, y_test = train_test_split( X,               # np array without target
                                               y,               # Target
                                               test_size = 0.25 # test_size proportion
                                               )
X_train.shape              # (1335, 6)
X_test.shape               # (446, 6)
'NaN' in num_columns
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
silhouette_score(X_train, clf.labels_)    # 0.20532663345078295

# 6 Make prediction over our test data and check accuracy
y_pred = clf.predict(X_test)
y_pred

# 6.1 How good is prediction
y_test.size
y_pred.size

#np.sum(y_pred == y_test)/y_test.size

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
visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()              # Finalize and render the figure

# Intercluster distance: Does not work
from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(clf)
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()              # Finalize and render the figure

#     8) Perform TSNE visualization (of the dataset) and color points with the clusters discovered above.
# t-SNE visualization
tsne = TSNE(n_components=2);
Xtsne = tsne.fit_transform(X);
df_tick = df.Ticker_Symbol.unique()


plt.figure();
for i in range(len(df_tick)):
    idx = np.where(y == i)[0];
    plt.scatter(Xtsne[idx,0], Xtsne[idx,1], alpha=0.6, label=df_tick[i]);
    
plt.title('t-SNE visualization');
#plt.legend();
plt.show();
# perform sector-wise analysis by merging two pandas DataFrames: fundamentals.csv and securities.csv.
#1) Read in 'fundamentals.csv' file
#df_sec = pd.read_csv("datasets_854_1575_securities.csv")
df_sec = pd.read_csv("../input/nyse/securities.csv")


df_sec
df_sec

cv_sec = df_sec.columns.values # Copy column names to another DF
cv_sec # Display new DF 

#2.1 Clean Column Names by replacing/removing special characters
j = 0
for i in cv_sec:
    cv_sec[j] = re.sub(' ', '_', cv_sec[j]) # Replace space with _
    cv_sec[j] = re.sub('\'','', cv_sec[j])  # Replace apostrophe with blank
    cv_sec[j] = re.sub('[*|\(\)\{\}]','', cv_sec[j]) # Replace special characters
    cv_sec[j] = re.sub('/','_', cv_sec[j])    # Replace / with _
    cv_sec[j] = re.sub('&','_', cv_sec[j])    # Replace & with _
    cv_sec[j] = re.sub('-','_', cv_sec[j])    # Replace - with _    
    cv_sec[j] = re.sub('\.','', cv_sec[j])    # Replace . with _
    cv_sec[j] = re.sub('[,]','', cv_sec[j])   # Replace , with blank         
    cv_sec[j] = re.sub('__.','_', cv_sec[j])  # Replace multiple _ with single _          
    j = j + 1
    
# Show cleaned column names
cv_sec 


# Make a disctionary of Old & new column names
y_sec = dict(zip(df_sec.columns.values, cv_sec))
y_sec
type(y_sec)



y_sec['Ticker_symbol'] = 'Ticker_Symbol'
y_sec
# Replace the column names
df_sec.rename(
         y_sec,
         inplace = True,
         axis = 1             # Note the axis keyword. By default it is axis = 0
         )
# Show the new column names
df_sec.columns.values
df_fnd_sec = pd.merge(df, df_sec, on='Ticker_Symbol', how='inner')
df_fnd_sec
sns.catplot(x = 'GICS_Sector',
            y = 'Gross_Profit',
            kind = 'bar',
            data = df_fnd_sec)

sns.catplot(x = 'GICS_Sector',
            y = 'Earnings_Per_Share',
            kind = 'box',
            data = df_fnd_sec)

sns.relplot(x = 'Earnings_Per_Share', y = 'GICS_Sector', kind = 'scatter', data = df_fnd_sec)

