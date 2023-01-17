import pandas as pd                 # data manipulation

import numpy as np                  # linear algebra

import matplotlib.pyplot as plt     # basic plotting 

import seaborn as sns               # plotting libriary

import datetime as dt                  # basic date and time types                    

import warnings    

import math

import time

import re

import os



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



warnings.filterwarnings('ignore')

plt.style.use(style = 'tableau-colorblind10')

plt.rcParams['figure.figsize'] = (15, 10)

os.getcwd()

os.listdir(os.getcwd())
file_path = '/kaggle/input/ecommerce-data/data.csv'



dtypes = {'InvoiceNo' : str, 'StockCode' : str, 'Description' : str, 'Quantity' : int,

          'UnitPrice' : float, 'CustomerID' : str, 'Country' : str}
online = pd.read_csv(file_path, encoding = 'latin1', dtype = dtypes)

online.head()
online.info()
online[online.Description.isna()]
online.dropna(inplace = True)

online.info()
online['InvoiceDate'] = pd.to_datetime(online['InvoiceDate'])
online.dtypes
online.head()
# Check the date variations in the DataFrame

print("Min Date: {} \t Max Date: {}".format(online['InvoiceDate'].min(), online['InvoiceDate'].max()))
# Check how many customers the DataFrame have and how many purchases they made

len(online['CustomerID'].value_counts())

# So there are 4372 unique customers that have made the purchases during the period of 1 year.
# Check quantity of positions 

print("The number of sell positions are {}.\n".format(len(online['Description'].value_counts())))



# So cutomers spread over 3896 position. I am interested what product have the most popularity.

print("The most popular merchandise in the E-commerce shop: \n\n{}.".format(online['Description'].value_counts()[:10]))
online.describe()
online = online[online['Quantity'] > 0]

online = online[online['UnitPrice'] > 0.05]

online.describe()
online['Country'].value_counts()
online = online[online['Country'] == 'United Kingdom']
online.head()
online.head()
# Define a function that will parse the dates from datetime object

def extract_days(x):

    """

    The function extract_dates receives a datetype object from a specified column and return

    splitted values into year, month and day.

    

    Usage:

    extract_dates(df['datetime object column'])

    or

    df['datetime_object_column'].apply(extract_dates)

    

    Returns:

    Series object:

    2020-02-01

    2020-02-02

    ..........

    2020-02-29

    

    Exceptions:

    This function is applied by the .apply() method if dataframe datetime column is passed.

    """

    return dt.datetime(x.year, x.month, x.day)
# Create InvoiceDay column that contains invoice dates issue by applying extract_dates function

online['InvoiceDay'] = online['InvoiceDate'].apply(extract_days)



# Groupby InvoiceDay column

grouping = online.groupby('CustomerID')['InvoiceDay'] 



# Assign a minimum InvoiceDay value to the dataset

online['CohortDay'] = grouping.transform('min')



# View the top 5 rows

print(online.head())
# Let do the another column that contains month cohort and invoice month for cohorts

# I will not describe what the function does as you can read similar to this above

def extract_month_int(x):

    return dt.datetime(x.year, x.month, 1)



# Create a column InvoiceMonth

online['InvoiceMonth'] = online['InvoiceDate'].apply(extract_month_int)



# Group

grouping = online.groupby('CustomerID')['InvoiceMonth']



# Assign new column

online['CohortMonth'] = grouping.transform('min')



# Show several rows of transformed data

online.head()
def extract_dates_int(df, column):

    """

    The function extract_dates_int help to divide datetime column into several columns based

    on splitting date into year, month and day.

    

    Usage:

    extract_dates_int(dataFrame, dataFrame['DateTime_Column'])

    

    Returns:

    tuple object that contains unique years, months and days.

    

    ((416792    2011

      482904    2011)

      ..............

      482904    11

      263743     7

      ..............

      482904    11

      263743     7

    

    Type:

    function

    

    """

    # extract years from datetime column

    year = df[column].dt.year

    

    # extract months from datetime column

    month = df[column].dt.month

    

    # extract days from datetime column

    day = df[column].dt.day

    

    return year, month, day
# Get the integers for date parts from 'InvoiceDay' column and 'CohortDay' column



# InvoiceDay column manipulation

invoice_year, invoice_month, _ = extract_dates_int(online, 'InvoiceMonth')



# CohortDay column manipulation

cohort_year, cohort_month, _ = extract_dates_int(online, 'CohortMonth')
# calculation of the difference in years

years_difference = invoice_year - cohort_year



# calculation of the difference in months

months_difference = invoice_month - cohort_month
# Extract the difference in days from all the previous extracted values above and create 

# new column called CohortIndex



# ~365 days in one year, ~30 days in one month and plus 1 day to differ from zero value

online['CohortIndex'] = years_difference * 12 + months_difference + 1

online.head()
grouping = online.groupby(['CohortMonth', 'CohortIndex'])



# Count the number of unique values per CustomerID

cohort_data = grouping['CustomerID'].apply(pd.Series.nunique).reset_index()



# Creating cohort pivot table 

cohort_counts = cohort_data.pivot(index = 'CohortMonth', columns = 'CohortIndex', values = 'CustomerID')



# Review the results

cohort_counts
# Select the first column and store value in cohort_sizes

cohort_sizes = cohort_counts.iloc[:, 0]



# Calculate Retention table by dividing the cohort count by cohort sizes along the rows

retention = cohort_counts.divide(cohort_sizes, axis = 0)



# Review the retention table

retention.round(3) * 100
grouping_avg_quantity = online.groupby(['CohortMonth', 'CohortIndex'])



# Extract Quantity column from grouping and calculate its mean value

cohort_data_avg_quantity = grouping_avg_quantity['Quantity'].mean().reset_index()



# average quantity table similar to retention but showing the change in quantity of products purchased

average_quantity = cohort_data_avg_quantity.pivot(index = 'CohortMonth', columns = 'CohortIndex', values = 'Quantity')

average_quantity.round(1).fillna('')
# Build a figure

plt.figure(figsize = (10, 8))

plt.title('Retentoin rate for customers')



# Initialize a heatmap grapgh 

sns.heatmap(data = retention, annot = True, fmt = '.0%', vmin = 0.01, vmax = 0.5, cmap = 'BuGn')



# show the retention graph

plt.show()
# Build a figure

plt.figure(figsize = (10, 8))

plt.title('average_quantity for customers')



# Initialize a heatmap grapgh 

sns.heatmap(data = average_quantity, annot = True, vmin = 0.01, vmax = 0.5, cmap = 'BuGn')



# show the retention graph

plt.show()
# Creating TotalSum Column in order to define a total amount spent by customers during the period

online['TotalSum'] = online['Quantity'] * online['UnitPrice']

online.head()
print('Min_date {} \nMax_date {}'.format(min(online.InvoiceDate), max(online.InvoiceDate)))
snapshot_date = max(online.InvoiceDate) + dt.timedelta(days = 1)
# Calculate Recency, Frequency and Monetary Values for each customer in the dataFrame

rfm_data = online.groupby(['CustomerID']).agg({

                                                'InvoiceDate' : lambda x: (snapshot_date - x.max()).days,

                                                'InvoiceNo' : 'count',

                                                'TotalSum' : 'sum'

                                                })



# Rename the created data columns in order to interpritate the obtained results

rfm_data.rename(columns = {

                            'InvoiceDate' : 'Recency',

                            'InvoiceNo' : 'Frequency',

                            'TotalSum' : 'MonetaryValue'

                            }, inplace = True)



# Check the obtained results

rfm_data.head()
# Labels for Recency, Frequenct and Monetary values metrics

r_labels = range(4, 0, -1)

f_labels = range(1, 5)

m_labels = range(1, 5)



# Recency metric quartiles

r_quartiles = pd.qcut(rfm_data['Recency'], 4, labels = r_labels)

rfm_data = rfm_data.assign(R = r_quartiles.values)



# Frequency metric quartiles

f_quartiles = pd.qcut(rfm_data['Frequency'], 4, labels = r_labels)

rfm_data = rfm_data.assign(F = f_quartiles.values)



# Monetary Value metric quartiles

m_quartiles = pd.qcut(rfm_data['MonetaryValue'], 4, labels = m_labels)

rfm_data = rfm_data.assign(M = m_quartiles.values)
rfm_data.head()
# Define function concat_rfm that will concatenate integer to string value

def concat_rfm(x):

    """

    Function which return a concatenated string from integer values.

    """

    return str(x['R']) + str(x['F']) + str(x['M'])



# Calculate the RFM segment 

rfm_data['RFM_Segment'] = rfm_data.apply(concat_rfm, axis = 1)



# Calculate the RFM score which is the sum of RFM values

rfm_data['RFM_Score'] = rfm_data[['R', 'F', 'M']].sum(axis = 1)
rfm_data.head()
# Explore the RFM score in the rfm_data

rfm_data.RFM_Score.value_counts().sort_index()
# Explore the RFM segment in the rfm_data

rfm_data.RFM_Segment.value_counts().sort_values(ascending = False)[:10]
# Function that assigns a humanlike label to each of the RFM segment based on the RFM scores

def auto_rfm_level(df):

    """

    Function that auto assigns humanlike segment to each RFM Segments.

    """

    if df['RFM_Score'] >= 8:

        return 'Top'

    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 8)):

        return 'Middle'

    else:

        return 'Low'







# In order to save the RFM labels create a new column where to store it

rfm_data['RFM_Level'] = rfm_data.apply(auto_rfm_level, axis = 1)



# Explore the obtained results

rfm_data.head()
# Explore the obtained RFM levels

print("Absolute values \n{} \n\nRelative values \n{}".format(rfm_data.RFM_Level.value_counts(),rfm_data.RFM_Level.value_counts(normalize = True)))
# Let's dig deeper into the RFM Score

rfm_data.groupby('RFM_Score').agg({

                                    'Recency' : 'mean',

                                    'Frequency' : 'mean',

                                    'MonetaryValue' : ['mean', 'count']

}).round(1)
# Let's dig deeper into the RFM Score

rfm_data.groupby('RFM_Level').agg({

                                    'Recency' : 'mean',

                                    'Frequency' : 'mean',

                                    'MonetaryValue' : ['mean', 'count']

}).round(1)
rfm_data.head()
# So to identify what type of data we have let's call describe method to show basic stats info

rfm_data.describe()
# The average values of the variables in the rfm_data dataset

print(np.mean(rfm_data))
# The standard deviation of the variables in the dataset

print(np.std(rfm_data))
# Visual Exploration of skeweness in the data

plt.subplot(3, 1, 1);

sns.distplot(rfm_data['Recency']);



plt.subplot(3, 1, 2);

sns.distplot(rfm_data['Frequency']);



plt.subplot(3, 1, 3);

sns.distplot(rfm_data['MonetaryValue'])



plt.show();
# Log Transform the Recency metric

rfm_data['Recency_log'] = np.log(rfm_data['Recency'])



# Log Transform the Frequency metric

rfm_data['Frequency_log'] = np.log(rfm_data['Frequency'])



# Log Transform the MonetaryValue metric

rfm_data['MonetaryValue_log'] = np.log(rfm_data['MonetaryValue'])
# Visual Exploration of log transformed data

plt.subplot(3, 1, 1);

sns.distplot(rfm_data['Recency_log']);



plt.subplot(3, 1, 2);

sns.distplot(rfm_data['Frequency_log']);



plt.subplot(3, 1, 3);

sns.distplot(rfm_data['MonetaryValue_log'])



plt.show();
rfm_data.head()
# Choose clean values to perform again step by step transoframtion using StandardScaler

raw_rfm = rfm_data.iloc[:,:3]

raw_rfm.head()
# Log Transformation

log_transformed_rfm = np.log(raw_rfm)



# Initializing a standard scaler and fitting it

scaler = StandardScaler()

scaler.fit(log_transformed_rfm)



# Scale and center data

rfm_normalized = scaler.transform(log_transformed_rfm)



# Create the final dataframe to work with a Clustering problem

rfm_normalized = pd.DataFrame(data = rfm_normalized, index = raw_rfm.index, columns = raw_rfm.columns)
# Visualise the obtained results

plt.subplot(3, 1, 1)

sns.distplot(rfm_normalized['Recency'])



plt.subplot(3, 1, 2)

sns.distplot(rfm_normalized['Frequency'])



plt.subplot(3, 1, 3)

sns.distplot(rfm_normalized['MonetaryValue'])



plt.show()
# Mean

print('Mean value of the data: \n\n{}'.format(rfm_normalized.mean(axis = 0).round(2)))



# Standard Deviation

print('\nStandard Deviation value of the data: \n\n{}'.format(rfm_normalized.std(axis = 0).round(2)))
# Initialisation of KMeans algorithm with number of cluster 4 (why I choose this I explain further)

kmeans = KMeans(n_clusters = 4, random_state = 42)



# Fit k-means cluster algorithm on the normalized data (rfm_normalized)

kmeans.fit(rfm_normalized)



# Extract the obtained cluster labels

cluster_labels = kmeans.labels_
# Create a DataFrame by adding a new cluster label column

rfm_cluster_k4 = rfm_data.assign(Cluster = cluster_labels)



# Group by cluster label

grouped_clusters_rfm = rfm_cluster_k4.groupby(['Cluster'])



grouped_clusters_rfm.agg({

    'Recency': 'mean',

    'Frequency': 'mean',

    'MonetaryValue': ['mean', 'count']

  }).round(1)

# Fit KMeans and calculate SSE for each k

sse = {}



for k in range(1, 25):

  

    # Initialize KMeans with k clusters

    kmeans = KMeans(n_clusters = k, random_state = 1)

    

    # Fit KMeans on the normalized dataset

    kmeans.fit(rfm_normalized)

    

    # Assign sum of squared distances to k element of dictionary

    sse[k] = kmeans.inertia_



# Add the plot title "The Elbow Method"

plt.title('The Elbow Method')



# Add X-axis label "k"

plt.xlabel('k')



# Add Y-axis label "SSE"

plt.ylabel('SSE')



# Plot SSE values for each key in the dictionary

sns.pointplot(x = list(sse.keys()), y = list(sse.values()))

plt.show();
# Calculate average RFM values for each cluster

cluster_avg = rfm_cluster_k4.groupby(['Cluster']).mean() 



# Calculate average RFM values for the total customer population

population_avg = raw_rfm.mean()



# Calculate relative importance of cluster's attribute value compared to population

relative_imp = cluster_avg / population_avg - 1



# Print relative importance scores rounded to 2 decimals

print(relative_imp.iloc[:, [0, 2, 5]].round(2))
# Initialize a plot with a figure size of 8 by 2 inches 

plt.figure(figsize = (8, 4))



# Add the plot title

plt.title('Relative importance of attributes')



# Plot the heatmap

sns.heatmap(data = relative_imp.iloc[:, [0, 2, 5]], annot = True, fmt='.2f', cmap='RdYlGn')

plt.show()