import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="white")

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/CreditCardUsage.csv')
df.head()
df.shape
df.info()
df.describe().T
print("Number of unique id's in CUST_ID column : ",df.CUST_ID.nunique())

print("Number of rows in dataframe : ",df.shape[0])

print('This is to check if we have a single row for each unique ID. We can drop customer id since we do not get any information from it.')
df.drop(columns='CUST_ID',inplace=True)
df.columns
f=plt.figure(figsize=(20,20))

for i, col in enumerate(df.columns):

    ax=f.add_subplot(6,3,i+1)

    sns.distplot(df[col].ffill(),kde=False)

    ax.set_title(col+" Distribution",color='Blue')

    plt.ylabel('Distribution')

f.tight_layout()
corr = df.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(200, 50, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap,)
import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

trace1 = go.Box(y = df['BALANCE'])

trace2 = go.Box(y=df['BALANCE_FREQUENCY'])

trace3=go.Box(y=df['PURCHASES'])

trace4=go.Box(y=df['ONEOFF_PURCHASES'])

trace5=go.Box(y=df['INSTALLMENTS_PURCHASES'])

trace6=go.Box(y=df['CASH_ADVANCE'])

trace7=go.Box(y=df['PURCHASES_FREQUENCY'])

trace8=go.Box(y=df['ONEOFF_PURCHASES_FREQUENCY'])

trace9=go.Box(y=df['PURCHASES_INSTALLMENTS_FREQUENCY'])

trace10=go.Box(y=df['CASH_ADVANCE_FREQUENCY'])

trace11=go.Box(y=df['CASH_ADVANCE_TRX'])

trace12=go.Box(y=df['PURCHASES_TRX'])

trace13=go.Box(y=df['CREDIT_LIMIT'])

trace14=go.Box(y=df['PAYMENTS'])

trace15=go.Box(y=df['MINIMUM_PAYMENTS'])

trace16=go.Box(y=df['PRC_FULL_PAYMENT'])

trace17=go.Box(y=df['TENURE'])



fig = tools.make_subplots(rows=3, cols=6, subplot_titles=('BALANCE', 'Balance_freq', 'PURCHASES', 'oneoff_purchases',

       'Installment_purchases', 'Cash_advance', 'Purchases_freq',

       'Oneoff_purchases_freq', 'Purchases_Installments_freq',

       'Cash_advance_freq', 'Cash_advance_trx', 'Purchases_trx',

       'Credit_Limit', 'Payments', 'Min_Payments', 'PRC_FULL_PAYMENT',

       'TENURE'))

fig['layout'].update(height=800, width=1420, title='Box Plots to visualize data distribution in each column')



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 1, 4)

fig.append_trace(trace5, 1, 5)

fig.append_trace(trace6, 1, 6)

fig.append_trace(trace7, 2, 1)

fig.append_trace(trace8, 2, 2)

fig.append_trace(trace9, 2, 3)

fig.append_trace(trace10, 2, 4)

fig.append_trace(trace11, 2, 5)

fig.append_trace(trace12, 2, 6)

fig.append_trace(trace13, 3, 1)

fig.append_trace(trace14, 3, 2)

fig.append_trace(trace15, 3, 3)

fig.append_trace(trace16, 3, 4)

fig.append_trace(trace17, 3, 5)

plt.tight_layout()

# data = [fig]

py.iplot(fig)
print('\n***************************************************    CHECK FOR NULL VALUES   ********************************************************* \n \n',df.isna().sum())
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(),inplace=True)
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(),inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Scaled_df = scaler.fit_transform(df)
df_scaled = pd.DataFrame(Scaled_df,columns=df.columns)

df_scaled.head()
fig, ax=plt.subplots(1,2,figsize=(15,5))

sns.distplot(df['BALANCE'], ax=ax[0],color='#D341CD')

ax[0].set_title("Original Data")

sns.distplot(df_scaled['BALANCE'], ax=ax[1],color='#D341CD')

ax[1].set_title("Scaled data")

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6,random_state=0)

kmeans.fit(Scaled_df)
kmeans.labels_
Sum_of_squared_distances = []

K = range(1,21)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(Scaled_df)

    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
from sklearn.metrics import silhouette_score, silhouette_samples



for n_clusters in range(2,21):

    km = KMeans (n_clusters=n_clusters)

    preds = km.fit_predict(Scaled_df)

    centers = km.cluster_centers_



    score = silhouette_score(Scaled_df, preds, metric='euclidean')

    print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
from yellowbrick.cluster import SilhouetteVisualizer



# Instantiate the clustering model and visualizer

km = KMeans (n_clusters=3)

visualizer = SilhouetteVisualizer(km)



visualizer.fit(Scaled_df) # Fit the training data to the visualizer

visualizer.poof() # Draw/show/poof the data
from yellowbrick.cluster import KElbowVisualizer

# Instantiate the clustering model and visualizer

km = KMeans (n_clusters=3)

visualizer = KElbowVisualizer(

    km, k=(2,21),metric ='silhouette', timings=False

)



visualizer.fit(Scaled_df) # Fit the training data to the visualizer

visualizer.poof() # Draw/show/poof the data
km = KMeans(n_clusters=3)
km.fit(Scaled_df)
cluster_label = km.labels_
df['KMEANS_LABELS'] = cluster_label
df.head()
df.columns
f=plt.figure(figsize=(20,20))

scatter_cols =['BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',

       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',

       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',

       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',

       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',

       'TENURE']

for i, col in enumerate(scatter_cols):

    ax=f.add_subplot(4,4,i+1)

    sns.scatterplot(x=df['BALANCE'],y=df[col],hue=df['KMEANS_LABELS'],palette='Set1')

    ax.set_title(col+" Scatter plot with clusters",color='blue')

    plt.ylabel(col)

f.tight_layout()
sample_df = pd.DataFrame([df['BALANCE'],df['PURCHASES']])

sample_df = sample_df.T

sample_df.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Sample_Scaled_df = scaler.fit_transform(sample_df)
km_sample = KMeans(n_clusters=4)

km_sample.fit(Sample_Scaled_df)
labels_sample = km_sample.labels_
sample_df['label'] = labels_sample
sns.set_palette('Set2')

sns.scatterplot(sample_df['BALANCE'],sample_df['PURCHASES'],hue=sample_df['label'],palette='Set1')