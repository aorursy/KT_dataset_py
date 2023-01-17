#!pip install -q datascience                   # Package that is required by pandas profiling

#!pip install -q pandas-profiling              # Toolbox for Generating Statistics Report
#!pip install -q --upgrade pandas-profiling

#!pip install -q --upgrade yellowbrick
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/).

# Writing will be preserved as output when you create a version using "Save & Run All".

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session.



# For Panel Data Analysis

import pandas as pd

from pandas_profiling import ProfileReport

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', None)

pd.set_option('display.max_rows', None)

pd.set_option('mode.chained_assignment', None)



# For Numerical Python

import numpy as np



# For Random seed values

from random import randint



# For Scientifc Python

from scipy import stats



# For datetime

import datetime

from datetime import datetime as dt



# For Data Visualization

import plotly.graph_objects as go

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# For Preprocessing

from sklearn.preprocessing import StandardScaler



# For Data Modeling

from sklearn.cluster import KMeans



# To Disable Warnings

import warnings

warnings.filterwarnings(action = "ignore")
LINK = '/kaggle/input/online-retail-data-v3/RetailDataIII.csv'



def load_reatil_data(link = LINK):

    return pd.read_csv(filepath_or_buffer = link)
data = load_reatil_data()

print('Data Shape:', data.shape)

data.head()
print('Column Length of Described Features:', len(data.describe().columns))

data.describe()
data.info()
#pre_profile = ProfileReport(df = data)

#pre_profile.to_file(output_file = 'Pre Profiling Report.html')

#print('Success!')
null_frame = pd.DataFrame(index = data.columns.values)

null_frame['Null Frequency'] = data.isnull().sum().values

percent = data.isnull().sum().values/data.shape[0]

null_frame['Missing %age'] = np.round(percent, decimals = 4) * 100

null_frame.transpose()
before_shape = data.shape

print('Data Shape [Before]:', before_shape)



data.dropna(axis = 0, subset = ['Product', 'CustomerID'], inplace = True)



after_shape = data.shape

print('Data Shape [After]:', after_shape)



drop_nums = before_shape[0] - after_shape[0]

drop_ratio = np.round(drop_nums / before_shape[0], decimals = 2) * 100

print('Drop Ratio:', drop_ratio, '%')
null_frame = pd.DataFrame(index = data.columns.values)

null_frame['Null Frequency'] = data.isnull().sum().values

percent = data.isnull().sum().values/data.shape[0]

null_frame['Missing %age'] = np.round(percent, decimals = 4) * 100

null_frame.transpose()
zero_frame = pd.DataFrame(index = data.columns.values)

zero_frame['Null Frequency'] = data[data == 0].count().values

percent = data[data == 0].count().values / data.shape[0]

zero_frame['Missing %age'] = np.round(percent, decimals = 4) * 100

zero_frame.transpose()
print('Contains Redundant Records?:', data.duplicated().any())

print('Duplicate Count:', data.duplicated().sum())
before_shape = data.shape

print('Data Shape [Before]:', before_shape)



data.drop_duplicates(inplace = True)



after_shape = data.shape

print('Data Shape [After]:', after_shape)



drop_nums = before_shape[0] - after_shape[0]

drop_percent = np.round(drop_nums / before_shape[0], decimals = 2) * 100



print('Drop Ratio:', drop_percent, '%')
print('Contains Redundant Records?:', data.duplicated().any())

print('Duplicate Count:', data.duplicated().sum())
def duplicate_cols(dataframe):

    ls1 = []

    ls2 = []



    columns = dataframe.columns.values

    for i in range(0, len(columns)):

        for j in range(i+1, len(columns)):

            if (np.where(dataframe[columns[i]] == dataframe[columns[j]], True, False).all() == True):

                ls1.append(columns[i])

                ls2.append(columns[j])



    if ((len(ls1) == 0) & (len(ls2) == 0)):

        return None

    else:

        duplicate_frame = pd.DataFrame()

        duplicate_frame['Feature 1'] = ls1

        duplicate_frame['Feature 2'] = ls2

        return duplicate_frame
print(duplicate_cols(data))
type_frame = pd.DataFrame(data = data.dtypes, columns = ['Type'])

type_frame.transpose()
data['Bill'] = data['Bill'].astype(np.int64)

data['BillDate'] = pd.to_datetime(data['BillDate'])

data['CustomerID'] = data['CustomerID'].astype(np.int64)
type_frame = pd.DataFrame(data = data.dtypes, columns = ['Type'])

type_frame.transpose()
# Converting negative values to positive (typo handling)

data['Quota'] = np.abs(data['Quota'])



# Creating a new feature

data['TotalSpend'] = data['Quota'] * data['Amount']
# Grouping data based on bill date

dx = data.groupby(by = 'BillDate', as_index = False).agg('sum')



# Adding Trend Factor

dx['EMA_Quota'] = dx.iloc[:, 2].ewm(span = 40,adjust = False).mean()

dx['EMA_TotalSpend'] = dx.iloc[:, 5].ewm(span = 40,adjust = False).mean()
figure = plt.figure(figsize = [15, 7])

sns.lineplot(x = 'BillDate', y = 'Quota', data = dx, color = '#D96552')

sns.lineplot(x = 'BillDate', y = 'EMA_Quota', data = dx, color = '#32B165')



plt.xlabel('Bill Date', size = 14)

plt.ylabel('Quantity Ordered per Day', size = 14)

plt.legend(labels = ['Quantity Ordered per Day', 'Trend'], loc = 'upper right', frameon = False)

plt.title('Quantity Ordered vs Invoice Date', size = 16)

plt.grid(b = True, axis = 'y')

plt.show()
figure = plt.figure(figsize = [15, 7])

sns.lineplot(x = 'BillDate', y = 'TotalSpend', data = dx, color = '#32B165')

sns.lineplot(x = 'BillDate', y = 'EMA_TotalSpend', data = dx, color = '#D96552')



plt.xlabel('Bill Date', size = 14)

plt.ylabel('Total Sales per Day', size = 14)

plt.legend(labels = ['Sales per Day', 'Trend'], loc = 'upper right', frameon = False)

plt.title('Total Sales vs Invoice Date', size = 16)

plt.grid(b = True, axis = 'y')

plt.show()
limit = 20



dx = data.groupby(by = 'Product', as_index = False).agg('sum').sort_values(by ='Quota', ascending = False)



figure = plt.figure(figsize = [15, 7])

ax = sns.barplot(x = 'Quota', y ='Product', data = dx[0 : limit])

total = len(data)

for p in ax.patches:

        percentage = '{:.2f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width()

        y = p.get_y() + p.get_height() / 2

        ax.annotate(percentage, (x, y))

plt.xlabel(xlabel = 'Quota', size = 14)

plt.ylabel(ylabel = 'Product', size = 14)

plt.title(label = 'Highest Sold Product Items (Top 10)', size = 16)

plt.grid(b = True, axis = 'x')

plt.show()
dx = data.groupby(by = 'Product', as_index = False).agg('sum').sort_values(by ='Quota', ascending = True)



print('Total Count of Product Items:', dx[dx['Quota'] == 1]['Quota'].count())



figure = plt.figure(figsize = [15, 7])

plt.subplot(1, 2, 1)

sns.barplot(x = 'Quota', y ='Product', data = dx[0 : 28], palette = sns.color_palette('RdBu_r', 28))

plt.xticks(ticks = range(0, 6))

plt.xlabel(xlabel = 'Quota', size = 14)

plt.ylabel(ylabel = 'Product', size = 14)

plt.title(label = 'Lowest Sold Product Items (First 28)', size = 16)



plt.subplot(1, 2, 2)

sns.barplot(x = 'Quota', y ='Product', data = dx[28 : 56], palette = sns.color_palette('RdBu_r', 28))

plt.xticks(ticks = range(0, 6))

plt.xlabel(xlabel = 'Quota', size = 14)

plt.ylabel(ylabel = 'Product', size = 14)

plt.title(label = 'Lowest Sold Product Items (Last 28)', size = 16)

plt.grid(b = True, axis = 'x')

plt.tight_layout(pad = 3.0)

plt.show()
dx = data.groupby(by = 'Country', as_index = False).agg('sum').sort_values(by ='TotalSpend', ascending = False)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [15, 8])

ax1 = sns.barplot(x = 'TotalSpend', y = 'Country', data = dx[0:20], ci = None, ax = ax1, palette = sns.color_palette('GnBu_d', 20))

ax1.set_xlabel(xlabel = 'Total Spend', size = 14)

ax1.set_ylabel(ylabel = 'Country', size = 14)

ax1.set_title(label = 'Total Spend per Country', size = 14)

ax1.grid(b = True, axis = 'x')



ax2 = sns.barplot(x = 'TotalSpend', y = 'Country', data = dx[21:], ci = None, ax = ax2, palette = sns.color_palette('GnBu_d', 21))

ax2.set_xlabel(xlabel = 'Total Spend', size = 14)

ax2.set_ylabel(ylabel = '')

ax2.set_title(label = 'Total Spend per Country', size = 14)

ax2.grid(b = True, axis = 'x')

plt.tight_layout(pad=3.0)

plt.show()
data['LogQuota'] = np.log1p(data['Quota'])

data['LogTotalSpend'] = np.log1p(data['TotalSpend'])
# Have some patience, may take some time :)

inertia_vals = []

K_vals = [x for x in range(1, 20)]



for i in K_vals:

    k_model = KMeans(n_clusters = i, max_iter = 500, random_state = 42, n_jobs = -1)

    k_model.fit(data[['LogQuota', 'LogTotalSpend']])

    inertia_vals.append(k_model.inertia_)

    print('Iteration:[', i, ']:completed')
# Visualzing the Inertia vs K Values

fig = go.Figure()



fig.add_trace(go.Scatter(x = K_vals, y = inertia_vals, mode = 'lines+markers'))

fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 1, dtick = 1),

                  title_text = 'Within Cluster Sum of Squared Distances VS K Values',

                  title_x = 0.5,

                  xaxis_title = 'K values',

                  yaxis_title = 'Cluster Sum of Squared Distances')

fig.show()
kmeans = KMeans(n_clusters = 5, max_iter = 500, random_state = 42, n_jobs = -1)

kmeans.fit(X = data[['LogQuota', 'LogTotalSpend']])

data['Labels'] = kmeans.labels_

centers = kmeans.cluster_centers_
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [20, 7])

ax1.scatter(x = data['LogQuota'], y = data['LogTotalSpend'], marker='.', s = 30, lw = 0, alpha = 0.7, c = data['Labels'], edgecolor = 'k')

ax1.scatter(x = centers[:, 0], y = centers[:, 1], marker = 'o', c = "white", alpha = 1, s = 200, edgecolor = 'k')

for i, c in enumerate(centers):

    ax1.scatter(x = c[0], y = c[1], marker = '$%d$' % i, alpha = 1, s = 50, edgecolor = 'k')

ax1.set_xlabel(xlabel = 'Feature Space 1 (LogQuota)', size = 14)

ax1.set_ylabel(ylabel = 'Feature Space 2 (LogTotalSpend)', size = 14)

ax1.set_title(label = 'Visualization of Clustered Data', size = 16)



flatui = ["#03051A", "#662758", "#CB1D50", "#F58860", "#FAEBDD"]

sns.countplot(x = 'Labels', data = data, palette = sns.color_palette(flatui), ax = ax2)

total = data.shape[0]

for p in ax2.patches:

    percentage = '{:.2f}%'.format(100*p.get_height()/ total)

    x = p.get_x() + p.get_width() / 3

    y = p.get_y() + p.get_height()

    ax2.annotate(percentage, (x, y))



ax2.set_xlabel('Clusters', size = 14)

ax2.set_ylabel('Frequency', size = 14)

ax2.set_title(label = 'Frequency distribution of Customers', size = 16)

plt.suptitle('Cluster Visualization', size = 18)

plt.show()
data.head()
print('Last Date:', data['BillDate'].max())



NOW = datetime.datetime(2019,12,10)



rfm_table = data.groupby(by = 'CustomerID', as_index = False).agg({'BillDate': lambda x: (NOW - x.max()).days, 

                                                                    'Bill': lambda x: len(x), 

                                                                    'TotalSpend': lambda x: x.sum()})

rfm_table.rename(columns = {'BillDate':'Recency', 'Bill':'Frequency', 'TotalSpend':'Monetary'}, inplace = True)

rfm_table.head()
quantiles = rfm_table.quantile(q = [0.25, 0.5, 0.75])

quantiles = quantiles.to_dict()

quantiles_frame = pd.DataFrame(data = quantiles)

quantiles_frame
def calRecency(x, y, z):

    if x <= z[y][0.25]:

        return 4

    elif x <= z[y][0.50]:

        return 3

    elif x <= z[y][0.75]:

        return 2

    else:

        return 1



def calFrequencyMonetary(x, y, z):

    if x <= z[y][0.25]:

        return 1

    elif x <= z[y][0.50]:

        return 2

    elif x <= z[y][0.75]:

        return 3

    else:

        return 4
rfm_table['R'] = rfm_table['Recency'].apply(calRecency, args = ('Recency', quantiles))

rfm_table['F'] = rfm_table['Frequency'].apply(calFrequencyMonetary, args = ('Frequency', quantiles))

rfm_table['M'] = rfm_table['Monetary'].apply(calFrequencyMonetary, args = ('Monetary', quantiles))
rfm_table.head()
rfm_table['RFM Segment'] = rfm_table['R'].map(str) + rfm_table['F'].map(str) + rfm_table['M'].map(str)

rfm_table['RFM Score'] = rfm_table[['R', 'F', 'M']].sum(axis = 1)

rfm_table.head()
score_labels = ['Green', 'Bronze', 'Silver', 'Gold']

score_groups = pd.qcut(x = rfm_table['RFM Score'], q = 4, labels = score_labels)

rfm_table['RFM Level'] = score_groups.values

rfm_table_2 = rfm_table.copy()

rfm_table.head()
rfm_table[(rfm_table['R'] == 4) & (rfm_table['F'] == 4) & (rfm_table['M'] == 4)].head()
rfm_table[(rfm_table['F'] == 4)].head()
rfm_table[(rfm_table['M'] == 4)].head()
rfm_table[(rfm_table['R'] == 2) & (rfm_table['F'] == 4) & (rfm_table['M'] == 4)].head()
rfm_table[(rfm_table['R'] == 1) & (rfm_table['F'] == 4) & (rfm_table['M'] == 4)].head()
rfm_table[(rfm_table['R'] == 1) & (rfm_table['F'] == 1) & (rfm_table['M'] == 1)].head()
# Creating subplots

fig, axes = plt.subplots(nrows = 1, ncols = 3, sharex = False, figsize=(15, 6))



feat = ['Recency', 'Frequency', 'Monetary']



# Generating random colors based on number of columns

colors = []

for i in range(len(feat)):

    colors.append('#%06X' % randint(0, 0xFFFFFF))  



for ax, col, color in zip(axes.flat, feat, colors):

    sns.distplot(a = rfm_table[col], bins = 30, ax = ax, color = color)

    ax.set_title(col)

    plt.setp(axes, xlabel = '')

    ax.grid(False)

plt.tight_layout()

plt.show()
scaler = StandardScaler()



rfm_table['Recency'] = scaler.fit_transform(np.log(rfm_table[['Recency']]))

rfm_table['Frequency'] = scaler.fit_transform(np.log(rfm_table[['Frequency']]))

rfm_table['Monetary'] = scaler.fit_transform(np.log(rfm_table[['Monetary']]))
rfm_table.head()
# Creating subplots

fig, axes = plt.subplots(nrows = 1, ncols = 3, sharex = False, figsize=(15, 6))



feat = ['Recency', 'Frequency', 'Monetary']



# Generating random colors based on number of columns

colors = []

for i in range(len(feat)):

    colors.append('#%06X' % randint(0, 0xFFFFFF))  



for ax, col, color in zip(axes.flat, feat, colors):

    sns.distplot(a = rfm_table[col], bins = 30, ax = ax, color = color)

    ax.set_title(col)

    plt.setp(axes, xlabel = '')

    ax.grid(False)

plt.tight_layout()

plt.show()
# Have some patience, may take some time :)

inertia_vals = []

K_vals = [x for x in range(1, 20)]



for i in K_vals:

    k_model = KMeans(n_clusters = i, max_iter = 500, random_state = 42, n_jobs = -1)

    k_model.fit(rfm_table[['Recency', 'Frequency', 'Monetary']])

    inertia_vals.append(k_model.inertia_)

    print('Iteration:[', i, ']:completed')
# Visualzing the Inertia vs K Values

fig = go.Figure()



fig.add_trace(go.Scatter(x = K_vals, y = inertia_vals, mode = 'lines+markers'))

fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 1, dtick = 1),

                  title_text = 'Within Cluster Sum of Squared Distances VS K Values',

                  title_x = 0.5,

                  xaxis_title = 'K values',

                  yaxis_title = 'Cluster Sum of Squared Distances')

fig.show()
kmeans = KMeans(n_clusters = 4, max_iter = 1000, random_state = 42, n_jobs = -1)

kmeans.fit(X = rfm_table[['Recency', 'Frequency', 'Monetary']])

rfm_table['Cluster'] = kmeans.labels_

centers = kmeans.cluster_centers_
rfm_table.head()
rfm_striped = rfm_table[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'RFM Level']]

rfm_melted = pd.melt(frame = rfm_striped, id_vars= ['CustomerID', 'RFM Level', 'Cluster'], var_name = 'Metrics', value_name = 'Value')

rfm_melted.head()
colors = palette = sns.color_palette("hls", 4)

figure = plt.figure(figsize = [15, 7])

plt.subplot(1, 2, 1)

sns.lineplot(x = 'Metrics', y = 'Value', hue = 'RFM Level', data = rfm_melted, palette = colors)

plt.title('Snake Plot of RFM', size = 16)

plt.legend(loc = 'upper right')



plt.subplot(1, 2, 2)

sns.lineplot(x = 'Metrics', y = 'Value', hue = 'Cluster', data = rfm_melted, palette = colors)

plt.title('Snake Plot of Clusters', size = 16)

plt.legend(loc = 'upper right')



plt.tight_layout(pad = 3.0)

plt.show()
# Total Mean value in R, F & M 

total_avg = rfm_table.iloc[:, 1:4].mean()



# Estimate RFM Level average

rfm_level_avg = rfm_table.groupby('RFM Level').mean().iloc[:, 1:4]

rfm_prop = rfm_level_avg/total_avg - 1



# Estimating cluster average

cluster_avg = rfm_table.groupby('Cluster').mean().iloc[:, 1:4]

cluster_prop = cluster_avg/total_avg - 1
figure = plt.figure(figsize = [15, 7])

plt.subplot(1, 2, 1)

sns.heatmap(data = rfm_prop, cmap= 'plasma' , annot = True)

plt.yticks(rotation = 0)

plt.title('Heatmap of RFM quantile', size = 14)



plt.subplot(1, 2, 2)

sns.heatmap(data = cluster_prop, cmap= 'viridis', annot = True)

plt.yticks(rotation = 0)

plt.title('Heatmap of K-Means', size = 14)

plt.suptitle(t = 'Correlation Analysis', y = 1.02, size = 16)

plt.tight_layout(pad = 3.0)

plt.show()