# lets import the required libraries and packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd/read_csv)

# Input data files are available in the "../input/" directory.



import io # Core tools for working with streams

import itertools # Functions creating iterators for efficient looping

import os # Miscellaneous operating system interfaces



# visualization

import matplotlib.pyplot as plt

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import seaborn as sns 



%matplotlib inline

from datetime import datetime as dt

from matplotlib.pyplot import xticks

from PIL import Image

RANDOM_STATE = 42

%matplotlib inline



# import matplotlib.ticker as mtick

# pd.set_option('display.max_columns', None)

# from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"



# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



py.init_notebook_mode(connected=True)



#ML Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier



# Metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import precision_score, recall_score, accuracy_score

from sklearn.metrics import precision_recall_curve

from sklearn import metrics

from sklearn.datasets import fetch_openml

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Scaling

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm



#Balancing

from imblearn import under_sampling 

from imblearn import over_sampling

from imblearn.over_sampling import SMOTE
# Reading Dataset

data = pd.read_csv("../input/telecom/telecom_churn_data.csv")

# first few rows

data.head()
# lets check the dimensions of the dataset

data.shape
# Missing data check



def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False).round(2)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print("Rows \t\t\t: ", data.shape[0])

print("Columns \t\t: ", data.shape[1])

print("\nFeatures \t\t: \n", data.columns.tolist())

print("\nMissing values  \t: ", data.isnull().sum().values.sum())

print("\nUnique values \t\t: \n", data.nunique())
# lets check the missing data

missing_data(data)
# find out the missing values

percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False).round(2)

percent_1 = percent[percent > 0]

percent_1
# number of coulmns with missing values

percent_1.shape
# more than 70-75% missing values

percent_70 = percent[percent > 70]
# number of coulmns with more than 70-75% missing values

percent_70.shape
# Checking if all mobile numbers are unique thereby confirming all user entries are unique

data['mobile_number'].nunique()
data.set_index('mobile_number', inplace = True)

data.head()
# statistics

data.describe().round(2)
# converting all blank cells to NA(null values)

data = data.replace(" ", np.nan)

data.head()
data[percent[percent == 73.66].index].describe()
data[percent[percent == 74.08].index].describe()
data[percent[percent == 74.43].index].describe()
data[percent[percent == 74.85].index].describe()
months = {"June": 6, "July": 7, "August": 8, "September": 9}

month_list = {}



# append all months columns in month_list

for i in months:

    month_list[i] = list(percent_70[percent_70.index.str.contains(str(months[i]))].index)



print(month_list, '\n')



for i in month_list:

    print('\n', i)

    print(100*data[month_list[i]].isnull().all(axis = 1).value_counts()/data.shape[0])
# Imputing blanks with zero for columns having > 70% missing values

for i in month_list:

    data[month_list[i]] = data[month_list[i]].fillna(value = 0)
percent_1.value_counts()
# quick check on each percentage values (3.94)

percent_1[percent_1 == 3.94]
# quick check on each percentage values (3.86)

percent_1[percent_1 == 3.86]
# quick check on each percentage values (5.38)

percent_1[percent_1 == 5.38]
# quick check on each percentage values (7.75)

percent_1[percent_1 == 7.75]
percent_3 = percent_1[percent_1 >= 3.8]

    

months = {"June": 6, "July": 7, "August": 8, "September": 9}

month_list = {}



# append all months columns in month_list

for i in months:

    month_list[i] = list(percent_3[percent_3.index.str.contains(str(months[i]))].index)



for i in month_list:

    print('\n', i)

    print(100*data[month_list[i]].isnull().all(axis = 1).value_counts()/data.shape[0])
for i in month_list:

    data[month_list[i]] = data[month_list[i]].fillna(value = 0)
# Final check on remaining missing values

percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False).round(2)

percent_1 = percent[percent > 0]

print(percent_1.shape)

percent_1
# Check the date columns which are of data type object and convert to date/time format

date_columns = data.columns[data.columns.str.contains('date')]

date_columns
# Converting all dates to YYYY-MM-DD

data[date_columns] = data[date_columns].apply(pd.to_datetime)

data[date_columns].head()
# last_date_of_month_9

data['last_date_of_month_9'] = data['last_date_of_month_9'].fillna(data['last_date_of_month_9'].unique()[0])

# last_date_of_month_8

data['last_date_of_month_8'] = data['last_date_of_month_8'].fillna(data['last_date_of_month_8'].unique()[0])

# last_date_of_month_7

data['last_date_of_month_7'] = data['last_date_of_month_7'].fillna(data['last_date_of_month_7'].unique()[0])
# date_of_last_rech_8

data['date_of_last_rech_8'] = data['date_of_last_rech_8'].fillna(value = 0)

# date_of_last_rech_7

data['date_of_last_rech_7'] = data['date_of_last_rech_7'].fillna(value = 0)

# date_of_last_rech_6

data['date_of_last_rech_6'] = data['date_of_last_rech_6'].fillna(value = 0)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False).round(2)

percent_1 = percent[percent > 0]

percent_1
# Recharge columns

recharge_data = data.columns[data.columns.str.contains('rech')]

recharge_data
# finding recharge amount for data

data['total_rech_amt_data_6'] = data.total_rech_data_6 * data.av_rech_amt_data_6

data['total_rech_amt_data_7'] = data.total_rech_data_7 * data.av_rech_amt_data_7

data['total_rech_amt_data_8'] = data.total_rech_data_8 * data.av_rech_amt_data_8
# creating new feature to filter high value customers

# Average recharge amount for good period (month 6 & 7)

data['average_amt_good'] = (data['total_rech_amt_6']+ data['total_rech_amt_7']+ data['total_rech_amt_data_6'] 

                            + data['total_rech_amt_data_7'])/4
recharge_amt_data = data[['total_rech_amt_6','total_rech_amt_7','total_rech_amt_data_6','total_rech_amt_data_7','average_amt_good']]

recharge_amt_data.head()
plt.figure(figsize = (12,8))

data['average_amt_good'].plot(kind='box')

plt.title('Avg. Recharge Amount during Good Phase')

plt.show()
data['average_amt_good'].describe(percentiles=[.25,.5,.70,.75,.80,.90,.95,.96,.97,.98,.99]).round(2)
# create a filter for values greater than 70th percentile of total average recharge amount for good phase 

A = data['average_amt_good'].quantile(0.7)

high_value_df = data[(data['average_amt_good'] >= A)]

high_value_df.shape
plt.figure(figsize = (12,8))

high_value_df['average_amt_good'].plot(kind='box')

plt.title('Avg. Recharge Amount during Good Phase')

plt.show()
high_value_df['average_amt_good'].describe(percentiles=[.70,.80,.90,.95,.99,.997,.999]).round(2)
B = data['average_amt_good'].quantile(0.99)

high_value = high_value_df[(high_value_df['average_amt_good'] <= B)]

high_value.shape
high_val_final = high_value.copy()
# Deleting the remaining blank / NA values

high_val_final = high_val_final.dropna(how='any',axis=0)
missing_data(high_val_final)
high_val_final[['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9']].head()
#Creating column for churn and classifying churned customers, Initially set all the values as 0

high_val_final['churn'] = 0

high_val_final['churn'].head()
# set all which having is_churned True condition as 1

churned = (high_val_final.total_ic_mou_9 == 0) & (high_val_final.total_og_mou_9 == 0) & (high_val_final.vol_2g_mb_9 == 0) & (high_val_final.vol_3g_mb_9 == 0)

high_val_final.loc[churned,'churn'] = 1
# Churn Rate %

round(100*high_val_final.churn.sum()/len(churned),2)
plt.figure(figsize=(8, 6)) 

plt.title("Churn Data")

sns.countplot(x = "churn", data = high_val_final)

xticks(rotation = 90)
# Deleting all columns of month 9 as these values have to be predicted

month_9 = high_val_final.columns[high_val_final.columns.str.contains('_9')]

month_9
high_val_final.drop(month_9, axis=1, inplace = True)

high_val_final.shape
# AON(Age on network) - number of days the customer is using the operator T network

plt.figure(figsize=(8,6))

plt.title("Age on Network Vs Churn Data")

sns.barplot(x = 'churn', y = 'aon', data = high_val_final)

plt.ylabel('Avg. Age of Network')

plt.xlabel('Churn')

print("Churned = ", high_val_final.loc[high_val_final.churn == 1, 'aon'].mean())

print("Not Churned = ", high_val_final.loc[high_val_final.churn == 0, 'aon'].mean())
sns.boxplot(y = 'aon', x = 'churn', data = high_val_final)

plt.show()
# Total Recharge Amount

# June

plt.figure(figsize=(8,6))

plt.title("Total Recharge Amount (June)")

sns.barplot(x = 'churn', y = 'total_rech_amt_6', data = high_val_final)

plt.ylabel('Avg. Recharge Amount')

plt.xlabel('Churn')

print("Churned = ", high_val_final.loc[high_val_final.churn == 1, 'total_rech_amt_6'].mean())

print("Not Churned = ", high_val_final.loc[high_val_final.churn == 0, 'total_rech_amt_6'].mean())
# July

plt.figure(figsize=(8,6))

plt.title("Total Recharge Amount (July)")

sns.barplot(x = 'churn', y = 'total_rech_amt_7', data = high_val_final)

plt.ylabel('Avg. Recharge Amount')

plt.xlabel('Churn')

print("Churned = ", high_val_final.loc[high_val_final.churn == 1, 'total_rech_amt_7'].mean())

print("Not Churned = ", high_val_final.loc[high_val_final.churn == 0, 'total_rech_amt_7'].mean())
plt.figure(figsize=(8,6))

plt.title("Total Recharge Amount (August)")

sns.barplot(x = 'churn', y = 'total_rech_amt_8', data = high_val_final)

plt.ylabel('Avg. Recharge Amount')

plt.xlabel('Churn')

print("Churned = ", high_val_final.loc[high_val_final.churn == 1, 'total_rech_amt_8'].mean())

print("Not Churned = ", high_val_final.loc[high_val_final.churn == 0, 'total_rech_amt_8'].mean())
# Creating a new feature for total recharge amount during good phase

high_val_final["total_rech_amt_goodphase"] = (high_val_final.total_rech_amt_6 + high_val_final.total_rech_amt_7)/2

# Dropping the original columns for good phase

high_val_final.drop(['total_rech_amt_6','total_rech_amt_7'], inplace=True, axis=1)
# Average Recharge Amount - Data

plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Data Recharge Amount (June)")

sns.barplot(x = 'churn', y = 'av_rech_amt_data_6', data = high_val_final)

plt.ylabel('Avg. Data Recharge Amount')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Data Recharge Amount (July)")

sns.barplot(x = 'churn', y = 'av_rech_amt_data_7', data = high_val_final)

plt.ylabel('Avg. Data Recharge Amount')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Data Recharge Amount (August)")

sns.barplot(x = 'churn', y = 'av_rech_amt_data_8', data = high_val_final)

plt.ylabel('Avg. Data Recharge Amount')

plt.xlabel('Churn')

plt.show()
# Fidning the total data recharge amount

high_val_final["total_rech_amt_data_6"] = high_val_final.total_rech_data_6*high_val_final.av_rech_amt_data_6

high_val_final["total_rech_amt_data_7"] = high_val_final.total_rech_data_7*high_val_final.av_rech_amt_data_7

high_val_final["total_rech_amt_data_8"] = high_val_final.total_rech_data_8*high_val_final.av_rech_amt_data_8
# Creating a new feature for total data recharge amount during good phase

high_val_final["total_rech_amt_data_goodphase"] = (high_val_final.total_rech_amt_data_6 + high_val_final.total_rech_amt_data_7)/2



# Creating a new feature for total data + calls recharge amount during good phase

high_val_final["overall_rech_amt_goodphase"] = high_val_final.total_rech_amt_goodphase + high_val_final.total_rech_amt_data_goodphase
# Dropping the original columns for good phase



high_val_final.drop(['total_rech_data_6','total_rech_data_7','total_rech_data_8','av_rech_amt_data_6',

                    'av_rech_amt_data_7','av_rech_amt_data_8'], axis=1, inplace=True)
# ARPU - Average Revenue Per User



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Average Revenue Per User (June)")

sns.barplot(x = 'churn', y = 'arpu_6', data = high_val_final)

plt.ylabel('Avg. Data Recharge Amount')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Average Revenue Per User (July)")

sns.barplot(x = 'churn', y = 'arpu_7', data = high_val_final)

plt.ylabel('Avg. Revenue Per User')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Average Revenue Per User (August)")

sns.barplot(x = 'churn', y = 'arpu_8', data = high_val_final)

plt.ylabel('Avg. Revenue Per User')

plt.xlabel('Churn')

plt.show()
# Creating a new variable for Average Revenue Per User in good phase

high_val_final["arpu_goodphase"] = (high_val_final.arpu_6 + high_val_final.arpu_7)/2



# Dropping the original columns for good phase                    

high_val_final.drop(['arpu_6','arpu_7'], axis=1, inplace=True)
# Maximum Recharge Amount



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Maximum Recharge Amount (June)")

sns.barplot(x = 'churn', y = 'max_rech_amt_6', data = high_val_final)

plt.ylabel('Avg. Maximum Recharge Amountt')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Maximum Recharge Amount (July)")

sns.barplot(x = 'churn', y = 'max_rech_amt_7', data = high_val_final)

plt.ylabel('Avg. Maximum Recharge Amount')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Maximum Recharge Amount (August)")

sns.barplot(x = 'churn', y = 'max_rech_amt_8', data = high_val_final)

plt.ylabel('Avg. Maximum Recharge Amount')

plt.xlabel('Churn')

# Creating a new variable for avg. max recharge amount in good phase

high_val_final["max_rech_amt_calls_goodphase"] = (high_val_final.max_rech_amt_6 + high_val_final.max_rech_amt_7)/2



# Dropping the original columns for good phase                    

high_val_final.drop(['max_rech_amt_6','max_rech_amt_7'], axis=1, inplace=True)
# last_day_rch_amt (rch = recharge, amt = amount)



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Last day recharge amount (June)")

sns.barplot(x = 'churn', y = 'last_day_rch_amt_6', data = high_val_final)

plt.ylabel('Avg. Last day recharge amount')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Last day recharge amount (July)")

sns.barplot(x = 'churn', y = 'last_day_rch_amt_7', data = high_val_final)

plt.ylabel('Avg. Last day recharge amountt')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Last day recharge amount (August)")

sns.barplot(x = 'churn', y = 'last_day_rch_amt_8', data = high_val_final)

plt.ylabel('Avg. Last day recharge amount')

plt.xlabel('Churn')

plt.show()
# total_rech_num

plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Total Number of Recharges (June)")

sns.barplot(x = 'churn', y = 'total_rech_num_6', data = high_val_final)

plt.ylabel('Avg. Total Number of Recharges')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Total Number of Recharges (July)")

sns.barplot(x = 'churn', y = 'total_rech_num_7', data = high_val_final)

plt.ylabel('Avg. Total Number of Recharges')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Total Number of Recharges (August)")

sns.barplot(x = 'churn', y = 'total_rech_num_8', data = high_val_final)

plt.ylabel('Avg. Total Number of Recharges')

plt.xlabel('Churn')

plt.show()
# max_rech_data



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Maximum data recharges (June)")

sns.barplot(x = 'churn', y = 'max_rech_data_6', data = high_val_final)

plt.ylabel('Avg. Maximum data recharges')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Maximum data recharges (July)")

sns.barplot(x = 'churn', y = 'max_rech_data_7', data = high_val_final)

plt.ylabel('Avg. Maximum data recharges')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Maximum data recharges (August)")

sns.barplot(x = 'churn', y = 'max_rech_data_8', data = high_val_final)

plt.ylabel('Avg. Maximum data recharges')

plt.xlabel('Churn')

plt.show()
# vol_2g_mb



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("2G volume (June)")

sns.barplot(x = 'churn', y = 'vol_2g_mb_6', data = high_val_final)

plt.ylabel('Avg. 2G volume')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("2G volume (July)")

sns.barplot(x = 'churn', y = 'vol_2g_mb_7', data = high_val_final)

plt.ylabel('Avg. 2G volume')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("2G volume (August)")

sns.barplot(x = 'churn', y = 'vol_2g_mb_8', data = high_val_final)

plt.ylabel('Avg. 2G volume')

plt.xlabel('Churn')

plt.show()
# monthly_2g



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Monthly 2G (June)")

sns.barplot(x = 'churn', y = 'monthly_2g_6', data = high_val_final)

plt.ylabel('Monthly 2G')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Monthly 2G (July)")

sns.barplot(x = 'churn', y = 'monthly_2g_7', data = high_val_final)

plt.ylabel('Monthly 2G')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Monthly 2G (August)")

sns.barplot(x = 'churn', y = 'monthly_2g_8', data = high_val_final)

plt.ylabel('Monthly 2G')

plt.xlabel('Churn')

plt.show()
# vol_3g_mb



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("3G volume (June)")

sns.barplot(x = 'churn', y = 'vol_3g_mb_6', data = high_val_final)

plt.ylabel('Avg. 3G volume')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("3G volume (July)")

sns.barplot(x = 'churn', y = 'vol_3g_mb_7', data = high_val_final)

plt.ylabel('Avg. 3G volume')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("3G volume (August)")

sns.barplot(x = 'churn', y = 'vol_3g_mb_8', data = high_val_final)

plt.ylabel('Avg. 3G volume')

plt.xlabel('Churn')

plt.show()
# monthly_3g



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Monthly 3G (June)")

sns.barplot(x = 'churn', y = 'monthly_3g_6', data = high_val_final)

plt.ylabel('Monthly 3G')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Monthly 3G (July)")

sns.barplot(x = 'churn', y = 'monthly_3g_7', data = high_val_final)

plt.ylabel('Monthly 3G')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Monthly 3G (August)")

sns.barplot(x = 'churn', y = 'monthly_3g_8', data = high_val_final)

plt.ylabel('Monthly 3G')

plt.xlabel('Churn')

plt.show()
# night_pck_user



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Night Pack (June)")

sns.barplot(x = 'churn', y = 'night_pck_user_6', data = high_val_final)

plt.ylabel('Night Pack')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Night Pack(July)")

sns.barplot(x = 'churn', y = 'night_pck_user_7', data = high_val_final)

plt.ylabel('Night Pack')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Night Pack (August)")

sns.barplot(x = 'churn', y = 'night_pck_user_8', data = high_val_final)

plt.ylabel('Night Pack')

plt.xlabel('Churn')

plt.show()
# Creating a new variable for monthly 2G and 3G

high_val_final["monthly_2g_goodphase"] = (high_val_final.monthly_2g_6 + high_val_final.monthly_2g_7)/2

high_val_final["monthly_3g_goodphase"] = (high_val_final.monthly_3g_6 + high_val_final.monthly_3g_7)/2



# Dropping the original columns for good phase

high_val_final.drop(['monthly_2g_6','monthly_2g_7','monthly_3g_6','monthly_3g_7'], axis=1, inplace=True)
# Creating a new variable for 2G and 3G Volumes

high_val_final["vol_2g_mb_goodphase"] = (high_val_final.vol_2g_mb_6 + high_val_final.vol_2g_mb_7)/2

high_val_final["vol_3g_mb_goodphase"] = (high_val_final.vol_3g_mb_6 + high_val_final.vol_3g_mb_7)/2



# Dropping the original columns for good phase

high_val_final.drop(['vol_2g_mb_6','vol_2g_mb_7','vol_3g_mb_6','vol_3g_mb_7'], axis=1, inplace=True)
# sachet_2g



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Sachet 2G (June)")

sns.barplot(x = 'churn', y = 'sachet_2g_6', data = high_val_final)

plt.ylabel('Sachet 2G')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Sachet 2G (July)")

sns.barplot(x = 'churn', y = 'sachet_2g_7', data = high_val_final)

plt.ylabel('Sachet 2G')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Sachet 2G (August)")

sns.barplot(x = 'churn', y = 'sachet_2g_8', data = high_val_final)

plt.ylabel('Sachet 2G')

plt.xlabel('Churn')

plt.show()
# sachet_3g



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Sachet 3G (June)")

sns.barplot(x = 'churn', y = 'sachet_3g_6', data = high_val_final)

plt.ylabel('Sachet 3G')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Sachet 3G (July)")

sns.barplot(x = 'churn', y = 'sachet_3g_7', data = high_val_final)

plt.ylabel('Sachet 3G')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Sachet 3G (August)")

sns.barplot(x = 'churn', y = 'sachet_3g_8', data = high_val_final)

plt.ylabel('Sachet 3G')

plt.xlabel('Churn')

plt.show()
# Creating a new variable for sachet 2G and 3G

high_val_final["sachet_2g_goodphase"] = (high_val_final.sachet_2g_6 + high_val_final.sachet_2g_7)/2

high_val_final["sachet_3g_goodphase"] = (high_val_final.sachet_3g_6 + high_val_final.sachet_3g_7)/2



# Dropping the original columns for good phase

high_val_final.drop(['sachet_2g_6','sachet_2g_7','sachet_3g_6','sachet_3g_7'], axis=1, inplace=True)
# Volume based cost - when no specific scheme is not purchased and paid as per usage

# changing the Volume based cost column names same as others

high_val_final.rename(columns= {'jun_vbc_3g': 'vbc_3g_6', 'jul_vbc_3g':'vbc_3g_7', 'aug_vbc_3g':'vbc_3g_8'}, inplace=True)

high_val_final.drop('sep_vbc_3g', axis = 1, inplace=True)
plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Volume based cost (June)")

sns.barplot(x = 'churn', y = 'vbc_3g_6', data = high_val_final)

plt.ylabel('Volume based cost')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Volume based cost (July)")

sns.barplot(x = 'churn', y = 'vbc_3g_7', data = high_val_final)

plt.ylabel('Volume based cost')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Volume based cost (August)")

sns.barplot(x = 'churn', y = 'vbc_3g_8', data = high_val_final)

plt.ylabel('Volume based cost')

plt.xlabel('Churn')

plt.show()
# Creating a new variable for VBC

high_val_final["vbc_3g_goodphase"] = (high_val_final.vbc_3g_6 + high_val_final.vbc_3g_7)/2



# Dropping the original columns for good phase

high_val_final.drop(['vbc_3g_6','vbc_3g_7'], axis=1, inplace=True)
# All kind of calls within the same operator network - onnet



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("On Network (June)")

sns.barplot(x = 'churn', y = 'onnet_mou_6', data = high_val_final)

plt.ylabel('On Network')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("On Network (July)")

sns.barplot(x = 'churn', y = 'onnet_mou_7', data = high_val_final)

plt.ylabel('On Network')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("On Network (August)")

sns.barplot(x = 'churn', y = 'onnet_mou_8', data = high_val_final)

plt.ylabel('On Network')

plt.xlabel('Churn')

plt.show()
# All kind of calls outside the operator T network - offnet



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Off Network (June)")

sns.barplot(x = 'churn', y = 'offnet_mou_6', data = high_val_final)

plt.ylabel('Off Network')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Off Network (July)")

sns.barplot(x = 'churn', y = 'offnet_mou_7', data = high_val_final)

plt.ylabel('Off Network')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Off Network (August)")

sns.barplot(x = 'churn', y = 'offnet_mou_8', data = high_val_final)

plt.ylabel('Off Network')

plt.xlabel('Churn')

plt.show()
# Roaming Minutes of usage (voice calls) - Incoming calls



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Roaming - Incoming MOU (June)")

sns.barplot(x = 'churn', y = 'roam_ic_mou_6', data = high_val_final)

plt.ylabel('Roaming - Incoming MOU')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Incoming MOU (July)")

sns.barplot(x = 'churn', y = 'roam_ic_mou_7', data = high_val_final)

plt.ylabel('Incoming MOU')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Roaming - Incoming MOU (August)")

sns.barplot(x = 'churn', y = 'roam_ic_mou_8', data = high_val_final)

plt.ylabel('Roaming - Incoming MOU')

plt.xlabel('Churn')

plt.show()
# # Roaming Minutes of usage (voice calls) - Outgoing calls



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Roaming - Outgoing MOU (June)")

sns.barplot(x = 'churn', y = 'roam_og_mou_6', data = high_val_final)

plt.ylabel('Roaming - Outgoing MOU')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Outgoing MOU (July)")

sns.barplot(x = 'churn', y = 'roam_og_mou_7', data = high_val_final)

plt.ylabel('Roaming - Outgoing MOU')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Roaming - Outgoing MOU (August)")

sns.barplot(x = 'churn', y = 'roam_og_mou_8', data = high_val_final)

plt.ylabel('Roaming - Outgoing MOU')

plt.xlabel('Churn')

plt.show()
# Overall Incoming and Outgoing MOU

mou = high_val_final.columns[high_val_final.columns.str.contains('_mou')]

mou
# Quick check on incoming usage for june - Local

print("Incoming")

high_val_final.loc_ic_mou_6.sum()

high_val_final.loc_ic_t2t_mou_6.sum()+ high_val_final.loc_ic_t2m_mou_6.sum()+high_val_final.loc_ic_t2f_mou_6.sum()

# Quick check on outgoing usage for june - Local

print("Outgoing")

high_val_final.loc_og_mou_6.sum()

high_val_final.loc_og_t2t_mou_6.sum()+ high_val_final.loc_og_t2m_mou_6.sum()+high_val_final.loc_og_t2f_mou_6.sum()
# Quick check on incoming usage for june - STD

print("Incoming")

high_val_final.std_ic_mou_6.sum()

high_val_final.std_ic_t2t_mou_6.sum()+ high_val_final.std_ic_t2m_mou_6.sum()+high_val_final.std_ic_t2f_mou_6.sum()

# Quick check on outgoing usage for june - STD

print("Outgoing")

high_val_final.std_og_mou_6.sum()

high_val_final.std_og_t2t_mou_6.sum()+ high_val_final.std_og_t2m_mou_6.sum()+high_val_final.std_og_t2f_mou_6.sum()
# Quick check on incoming usage for june - ISD

print("Incoming")

high_val_final.total_ic_mou_6.sum()

high_val_final.loc_ic_mou_6.sum()+high_val_final.std_ic_mou_6.sum()+high_val_final.isd_ic_mou_6.sum()+high_val_final.spl_ic_mou_6.sum()+high_val_final.ic_others_6.sum()

# # Quick check on outgoing usage for june - ISD

print("Outgoing")

high_val_final.total_og_mou_6.sum()

high_val_final.loc_og_mou_6.sum()+ high_val_final.std_og_mou_6.sum()+high_val_final.isd_og_mou_6.sum()+high_val_final.spl_og_mou_6.sum()+high_val_final.og_others_6.sum()
# Creating new features for incoming and outgoing MOU for good phase

# Local incoming

high_val_final["loc_ic_t2f_mou_goodphase"] = (high_val_final.loc_ic_t2f_mou_6 + high_val_final.loc_ic_t2f_mou_7)/2

high_val_final["loc_ic_t2m_mou_goodphase"] = (high_val_final.loc_ic_t2m_mou_6 + high_val_final.loc_ic_t2m_mou_7)/2

high_val_final["loc_ic_t2t_mou_goodphase"] = (high_val_final.loc_ic_t2t_mou_6 + high_val_final.loc_ic_t2t_mou_7)/2
# STD incoming

high_val_final["std_ic_t2f_mou_goodphase"] = (high_val_final.std_ic_t2f_mou_6 + high_val_final.std_ic_t2f_mou_7)/2

high_val_final["std_ic_t2m_mou_goodphase"] = (high_val_final.std_ic_t2m_mou_6 + high_val_final.std_ic_t2m_mou_7)/2

high_val_final["std_ic_t2o_mou_goodphase"] = (high_val_final.std_ic_t2o_mou_6 + high_val_final.std_ic_t2o_mou_7)/2

high_val_final["std_ic_t2t_mou_goodphase"] = (high_val_final.std_ic_t2t_mou_6 + high_val_final.std_ic_t2t_mou_7)/2



# Other incoming

high_val_final["isd_ic_mou_goodphase"] = (high_val_final.isd_ic_mou_6 + high_val_final.isd_ic_mou_7)/2

high_val_final["spl_ic_mou_goodphase"] = (high_val_final.spl_ic_mou_6 + high_val_final.spl_ic_mou_7)/2

high_val_final["ic_others_goodphase"] = (high_val_final.ic_others_6 + high_val_final.ic_others_7)/2

high_val_final["roam_ic_mou_goodphase"] = (high_val_final.roam_ic_mou_6 + high_val_final.roam_ic_mou_7)/2
# Local outgoing

high_val_final["loc_og_t2f_mou_goodphase"] = (high_val_final.loc_og_t2f_mou_6 + high_val_final.loc_og_t2f_mou_7)/2

high_val_final["loc_og_t2m_mou_goodphase"] = (high_val_final.loc_og_t2m_mou_6 + high_val_final.loc_og_t2m_mou_7)/2

high_val_final["loc_og_t2t_mou_goodphase"] = (high_val_final.loc_og_t2t_mou_6 + high_val_final.loc_og_t2t_mou_7)/2
# STD Outgoing

high_val_final["std_og_t2f_mou_goodphase"] = (high_val_final.std_og_t2f_mou_6 + high_val_final.std_og_t2f_mou_7)/2

high_val_final["std_og_t2m_mou_goodphase"] = (high_val_final.std_og_t2m_mou_6 + high_val_final.std_og_t2m_mou_7)/2

high_val_final["std_og_t2t_mou_goodphase"] = (high_val_final.std_og_t2t_mou_6 + high_val_final.std_og_t2t_mou_7)/2

# Other Outgoing

high_val_final["isd_og_mou_goodphase"] = (high_val_final.isd_og_mou_6 + high_val_final.isd_og_mou_7)/2

high_val_final["spl_og_mou_goodphase"] = (high_val_final.spl_og_mou_6 + high_val_final.spl_og_mou_7)/2

high_val_final["og_others_goodphase"] = (high_val_final.og_others_6 + high_val_final.og_others_7)/2

high_val_final["roam_og_mou_goodphase"] = (high_val_final.roam_og_mou_6 + high_val_final.roam_og_mou_7)/2
# Dropping redundant columns

# Incoming

high_val_final.drop(['loc_ic_t2f_mou_6','loc_ic_t2m_mou_6','loc_ic_t2t_mou_6','std_ic_t2f_mou_6',

                           'std_ic_t2m_mou_6','std_ic_t2o_mou_6','std_ic_t2t_mou_6','isd_ic_mou_6', 'spl_ic_mou_6',

                           'ic_others_6', 'roam_ic_mou_6','loc_ic_t2f_mou_7','loc_ic_t2m_mou_7','loc_ic_t2t_mou_7',

                           'std_ic_t2f_mou_7','std_ic_t2m_mou_7', 'std_ic_t2o_mou_7','std_ic_t2t_mou_7', 

                           'isd_ic_mou_7', 'spl_ic_mou_7', 'ic_others_7','roam_ic_mou_7'], inplace=True, axis=1)
# Outgoing

high_val_final.drop(['loc_og_t2f_mou_6','loc_og_t2m_mou_6', 'loc_og_t2t_mou_6',

                           'std_og_t2f_mou_6','std_og_t2m_mou_6', 'std_og_t2t_mou_6', 'isd_og_mou_6',

                           'spl_og_mou_6', 'og_others_6','roam_og_mou_6', 'loc_og_t2f_mou_7','loc_og_t2m_mou_7',

                           'loc_og_t2t_mou_7','std_og_t2f_mou_7','std_og_t2m_mou_7','std_og_t2t_mou_7',

                           'isd_og_mou_7', 'spl_og_mou_7', 'og_others_7', 'roam_og_mou_7'], inplace=True, axis=1)
# RPU - Revenue Per User 2G



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Revenue Per User 2G (June)")

sns.barplot(x = 'churn', y = 'arpu_2g_6', data = high_val_final)

plt.ylabel('Revenue Per User 2G')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Revenue Per User 2G (July)")

sns.barplot(x = 'churn', y = 'arpu_2g_7', data = high_val_final)

plt.ylabel('Revenue Per User 2G')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Revenue Per User 2G (August)")

sns.barplot(x = 'churn', y = 'arpu_2g_8', data = high_val_final)

plt.ylabel('Revenue Per User 2G')

plt.xlabel('Churn')

plt.show()
# RPU - Revenue Per User 3G



plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

# June

plt.title("Revenue Per User 3G (June)")

sns.barplot(x = 'churn', y = 'arpu_3g_6', data = high_val_final)

plt.ylabel('Revenue Per User 3G')

plt.xlabel('Churn')



plt.subplot(1,3,2)

# July

plt.title("Revenue Per User 3G (July)")

sns.barplot(x = 'churn', y = 'arpu_3g_7', data = high_val_final)

plt.ylabel('Revenue Per User 3G')

plt.xlabel('Churn')



plt.subplot(1,3,3)

# August

plt.title("Revenue Per User 3G (August)")

sns.barplot(x = 'churn', y = 'arpu_3g_8', data = high_val_final)

plt.ylabel('Revenue Per User 3G')

plt.xlabel('Churn')

plt.show()
high_val_final.shape
high_val_final.to_csv("high_value_final.csv")
# Correlation Matrix / Heat map

plt.figure(figsize=(20,12))

cor = high_val_final.corr()

sns.heatmap(cor, annot=True, cmap="YlGnBu")

plt.tight_layout()

plt.show()
# Dropping date columns & circle id as it will nit be needed for modelling and also we have other new features



high_val_final.drop(['last_date_of_month_6','last_date_of_month_7', 'last_date_of_month_8','date_of_last_rech_6',

                    'date_of_last_rech_7','date_of_last_rech_8','date_of_last_rech_data_6','date_of_last_rech_data_7',

                    'date_of_last_rech_data_8'], axis=1, inplace=True)

high_val_final.drop('circle_id', axis = 1, inplace = True)

high_val_final.shape
# Correlation Matrix / Heat map

plt.figure(figsize=(20,12))

cor = high_val_final.corr()

sns.heatmap(cor, annot=True, cmap="YlGnBu")

plt.tight_layout()

plt.show()
telecom_model = high_val_final.copy()
# Response variable

y = telecom_model['churn']

# Feature variable

x = telecom_model.drop('churn', axis=1)
# Split the data between train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = RANDOM_STATE)
# This is performed only on train data set as balancing is needed whie training the model

#check churn ratio before balacing using smote

print('Churn % before balancing')

(100 * y_train.value_counts()/ y_train.shape[0]).round(2)
#Churn Distribution

pie_chart = telecom_model['churn'].value_counts()*100.0 /len(telecom_model)

ax = pie_chart.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(8,6), fontsize = 14 )                                                                           

ax.set_ylabel('Churn',fontsize = 12)

ax.set_title('Churn Distribution', fontsize = 12)

plt.show()
# Balancing

smote = SMOTE()

x_train, y_train = smote.fit_sample(x_train, y_train)
#check churn ratio after balacing using smote

print('Churn % after balancing')

print((y_train != 0).sum()/(y_train == 0).sum())
# Scale variables

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
#Perform PCA

pca = PCA(svd_solver = 'randomized', random_state = RANDOM_STATE)

pca.fit(x_train)
pca.components_
# Checking the first 20 principal components

pca.explained_variance_ratio_[0:20]
#Making the screeplot



fig = plt.figure(figsize = (8,6))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.title('Cumulative Proportion of Variance Explained by PC')

plt.show()

pca_explain_variance_x = {'PC-10':round(pca.explained_variance_ratio_[0:10].sum(),2), 

                               'PC-20':round(pca.explained_variance_ratio_[0:20].sum(),2),

                               'PC-30':round(pca.explained_variance_ratio_[0:30].sum(),2), 

                               'PC-40':round(pca.explained_variance_ratio_[0:40].sum(),2),

                               'PC-50':round(pca.explained_variance_ratio_[0:50].sum(),2),

                               'PC-60':round(pca.explained_variance_ratio_[0:60].sum(),2),

                               'PC-70':round(pca.explained_variance_ratio_[0:70].sum(),2),

                               'PC-80':round(pca.explained_variance_ratio_[0:80].sum(),2),

                               'PC-90':round(pca.explained_variance_ratio_[0:90].sum(),2)}

pca_explain_variance_x
#Visualizing the principal components

col = list(x.columns)

pca_df = pd.DataFrame({'PC1':pca.components_[0], 'PC2':pca.components_[1],'Feature':col})

pca_df.head(10)
fig = plt.figure(figsize = (8,8))

plt.scatter(pca_df.PC1, pca_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.show()

# Fitting the PCA

pca_final = IncrementalPCA(n_components = 50)
df_train_pca = pca_final.fit_transform(x_train)
# Checking multicollinearity - if there is correlation between PC's

corrmat = np.round(np.corrcoef(df_train_pca.transpose()),2)
# Plot the correlation matrix

plt.figure(figsize = (20,20))

sns.heatmap(corrmat, annot= True, cmap="YlGnBu")

# Applying the PC's on test data

df_test_pca = pca_final.transform(x_test)
# Logistic Regression

LR_with_PCA = LogisticRegression(random_state=RANDOM_STATE)

C=list(np.power(10.0, np.arange(-4, 4)))

n_folds = 5

gs = GridSearchCV(LR_with_PCA,

    param_grid=[{'C': C, 'penalty': ['l1', 'l2']}],

    cv= n_folds,                 

    scoring='accuracy',

    n_jobs= 1,

    verbose=1,

    refit=True,

    return_train_score = True)
# Fittng the Model

LR_with_PCA_model = gs.fit(df_train_pca, y_train)
# cv results

cv_results = pd.DataFrame(LR_with_PCA_model.cv_results_)

cv_results.columns
# Getting the important columns out

cv_results_important = cv_results[['param_C', 'param_penalty',

                                   'mean_train_score','std_train_score',

                                   'mean_test_score','std_test_score',

                                   'rank_test_score']]

cv_results_important.sort_values('rank_test_score').head()
cv_results_l1 = cv_results[cv_results_important['param_penalty'] == 'l1']

cv_results_l2 = cv_results[cv_results_important['param_penalty'] == 'l2']
cv_results_l1['param_C'] = cv_results_l1['param_C'].astype('int')



plt.figure(figsize=(8,8))

l1 = cv_results_l1

plt.plot(l1["param_C"], l1["mean_test_score"])

plt.plot(l1["param_C"], l1["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("L1 regularization")

plt.ylim([0.84, 0.85])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
plt.figure(figsize=(8,8))

l2 = cv_results_l2



plt.plot(l2["param_C"], l2["mean_test_score"])

plt.plot(l2["param_C"], l2["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("L2 regularization")

plt.ylim([0.81, 0.85])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
## Optimal scores

opt_score = LR_with_PCA_model.best_score_

opt_score.round(4)
# Optimal params

opt_hyperparams = LR_with_PCA_model.best_params_

opt_hyperparams
# Fitting the model with best hyper parameters

# specifying optimal hyperparameters



# Optimal Model

LR = LogisticRegression(penalty = 'l2', C = 1.0, random_state=RANDOM_STATE)

LR_PCA_final = LR.fit(df_train_pca, y_train)
# Train and Test Accuracy

predict_prob_train = LR_PCA_final.predict_proba(df_train_pca)[:,1]

predict_prob_test = LR_PCA_final.predict_proba(df_test_pca)[:,1]
print("Train Data Accurancy = ",round(100*metrics.roc_auc_score(y_train, predict_prob_train),4))

print("Test Data Accurancy = ",round(100*metrics.roc_auc_score(y_test, predict_prob_test),4))
# Creating data frame with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Churn':y_train, 'Churn_prob':predict_prob_train})
# Plotting ROC Curve

def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None



# Calling the function

draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_prob)
# Creating columns with different probability cutoffs

numbers = [float(x)/10 for x in range (10)]

for i in numbers: 

    y_train_pred_final[i] = y_train_pred_final.Churn_prob.map(lambda x: 1 if x>i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensitivity','specificity'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Curve

cutoff_df.plot.line(x='prob',y=['accuracy','sensitivity','specificity'])

plt.show()
y_train_pred_final['Final_Predict'] = y_train_pred_final.Churn_prob.map(lambda x: 1 if x>=0.55 else 0)
# Final DataFrame

finfr = list(y_train_pred_final.columns)

y_train_pred_final.drop(finfr[2:-1],axis = 1, inplace = True)

y_train_pred_final.head()
#### Metrics

Precision_sklearn = round(100*precision_score(y_train_pred_final.Churn, y_train_pred_final.Final_Predict),4)

Recall_sklearn = round(100*recall_score(y_train_pred_final.Churn, y_train_pred_final.Final_Predict),4)

Accuracy_sklearn = accuracy_score(y_train_pred_final.Churn,y_train_pred_final.Final_Predict)



print('Accuracy_sklearn',Precision_sklearn)

print('Precision_sklearn',Precision_sklearn)

print('Recall_sklearn',Recall_sklearn)
conf_mat_2 = metrics.confusion_matrix(y_train_pred_final.Churn,y_train_pred_final.Final_Predict)

conf_mat_2

TP = conf_mat_2[1,1] # true positive 

TN = conf_mat_2[0,0] # true negatives

FP = conf_mat_2[0,1] # false positives

FN = conf_mat_2[1,0] # false negatives
# Check the accuracy of the model

Accuracy = round(100*(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.Final_Predict)),1)



# Measure the sensitivity

Sensitivity = round(100*TP/float(TP+FN))



# Measure specificity

Specificity = round(100*TN/float(TN+FP),1)



# Precision

Precision = round(100*TP/float(TP+FP))



# Recall

Recall = round(100*TP/float(TP+FN),1)
Metrics_train = {'Accuracy': Accuracy, 

     'Precision': Precision,

     'Recall': Recall}



Metrics_train
# Precision recall curve



p,r, thresholds = precision_recall_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_prob)

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
# Model performance on test data

df_test_pca[0:1]
# Test Result Dataframe

# Predict the y test values

y_test_pred = LR_PCA_final.predict(df_test_pca)



# Convert y_pred to a data frame which is an array

y_pred_1 = pd.DataFrame(y_test_pred)



# View the dataframe

y_pred_1.head()
# Convert y_test to a Dataframe

y_test_df = pd.DataFrame(y_test)

y_test_df.reset_index(inplace = True)
# Append y_pred_1 and y_test_df

y_pred_final = pd.concat([y_test_df,y_pred_1], axis=1)

y_pred_final.head()
# Rename the column

y_pred_final = y_pred_final.rename(columns={0 :'Churn_Predict'})

y_pred_final.head()
y_pred_final['Final_pred'] = y_pred_final.Churn_Predict.map(lambda x: 1 if x >= 0.55 else 0 )

y_pred_final.head()
# Confusion Matrix - Test

conf_mat_3 = metrics.confusion_matrix(y_pred_final.churn, y_pred_final.Churn_Predict)

conf_mat_3
# Confusion Matrix - Test

TP = conf_mat_3[1,1] # true positive 

TN = conf_mat_3[0,0] # true negatives

FP = conf_mat_3[0,1] # false positives

FN = conf_mat_3[1,0] # false negatives



# Check the accuracy of the model

Accuracy = round(100*(metrics.accuracy_score(y_pred_final.churn, y_pred_final.Final_pred)),1)



# Measure the sensitivity

Sensitivity = round(100*TP/float(TP+FN))



# Measure specificity

Specificity = round(100*TN/float(TN+FP),1)





# Precision

Precision = round(100*TP/float(TP+FP))



# Recall

Recall = round(100*TP/float(TP+FN),1)
Metrics_test = {'Accuracy': Accuracy, 

     'Precision': Precision,

     'Recall': Recall}

print('Test Metrics:', Metrics_test)

print('Train Metrics', Metrics_train)
# Creating the parameter grid

param_grid = {

    'max_depth': [1,7,15],

    'min_samples_leaf': range(15, 50, 5),

    'min_samples_split': range(20, 40, 5),

    'n_estimators': [15], 

    'max_features': [10, 20, 5]

}



rf = RandomForestClassifier(random_state=RANDOM_STATE)

# run the grid search

grid_search = GridSearchCV(rf, param_grid=param_grid, 

                          cv= 3, n_jobs=-1, 

                          verbose=1,

                          return_train_score = True)
# Fitting the model to our data

rf_model = grid_search.fit(df_train_pca, y_train)
# cv results

cv_results_rf = pd.DataFrame(rf_model.cv_results_)

cv_results_rf.columns
# Getting the important columns out

cv_results_important_rf = cv_results_rf[['param_max_depth', 'param_max_features','param_min_samples_leaf',

                                   'param_min_samples_split','param_n_estimators',

                                   'mean_train_score','std_train_score',

                                   'mean_test_score','std_test_score',

                                   'rank_test_score']]

cv_results_important_rf.sort_values('rank_test_score').head(5)
cv_results_important_rf.to_csv("cv_results_rf.csv")
# plotting accuracies with min_samples_split

plt.figure(figsize=(10,6))



plt.plot(cv_results_important_rf["param_min_samples_split"], 

         cv_results_important_rf["mean_train_score"], 

         label="training accuracy")

plt.plot(cv_results_important_rf["param_min_samples_split"], 

         cv_results_important_rf["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.legend()
plt.figure(figsize=(10,6))

plt.plot(cv_results_important_rf["param_min_samples_leaf"], 

         cv_results_important_rf["mean_train_score"], 

         label="training accuracy")

plt.plot(cv_results_important_rf["param_min_samples_leaf"], 

         cv_results_important_rf["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()
plt.figure(figsize=(10,6))

plt.plot(cv_results_important_rf["param_max_features"], 

         cv_results_important_rf["mean_train_score"], 

         label="training accuracy")

plt.plot(cv_results_important_rf["param_max_features"], 

         cv_results_important_rf["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_features")

plt.ylabel("Accuracy")

plt.legend()

plt.figure(figsize=(10,6))

plt.plot(cv_results_important_rf["param_max_depth"], 

         cv_results_important_rf["mean_train_score"], 

         label="training accuracy")

plt.plot(cv_results_important_rf["param_max_depth"], 

         cv_results_important_rf["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()
plt.figure(figsize=(10,6))

plt.plot(cv_results_important_rf["param_n_estimators"], 

         cv_results_important_rf["mean_train_score"], 

         label="training accuracy")

plt.plot(cv_results_important_rf["param_n_estimators"], 

         cv_results_important_rf["mean_test_score"], 

         label="test accuracy")

plt.xlabel("n_estimators")

plt.ylabel("Accuracy")

plt.legend()
# optimal accuracy score and hyperparameters

opt_score = rf_model.best_score_

opt_hyperparams = rf_model.best_params_



print("Optimal Test Score {0} corresponding to hyperparameters {1}".format(opt_score, opt_hyperparams))
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=15,

                             min_samples_leaf=15, 

                             min_samples_split=20,

                             max_features=20,

                             n_estimators=15)
# fitting the data

rfc.fit(df_train_pca,y_train)
# Predictng the results on x_train

predict_train = rfc.predict(df_train_pca)



# Predicting the results on x_test

predict_test = rfc.predict(df_test_pca)
# Checking results for train data

print(classification_report(y_train, predict_train))
# Checking results for test data

print(classification_report(y_test, predict_test))
# Print confusion matrix

# Train Data

conf_mat_train = confusion_matrix(y_train,predict_train)

# Test Data

conf_mat_test = confusion_matrix(y_test,predict_test)
# Train Data

TP = conf_mat_train[1,1] # true positive 

TN = conf_mat_train[0,0] # true negatives

FP = conf_mat_train[0,1] # false positives

FN = conf_mat_train[1,0] # false negatives



# Check the accuracy of the model

Accuracy = round(100*(metrics.accuracy_score(y_train, predict_train)),1)



# Measure the sensitivity

Sensitivity = round(100*TP/float(TP+FN))



# Measure specificity

Specificity = round(100*TN/float(TN+FP),1)





# Precision

Precision = round(100*TP/float(TP+FP))



# Recall

Recall = round(100*TP/float(TP+FN),1)
Metrics_train = {'Accuracy': Accuracy, 

     'Precision': Precision,

     'Recall': Recall}



Metrics_train
# Train Data

TP = conf_mat_test[1,1] # true positive 

TN = conf_mat_test[0,0] # true negatives

FP = conf_mat_test[0,1] # false positives

FN = conf_mat_test[1,0] # false negatives



# Check the accuracy of the model

Accuracy = round(100*(metrics.accuracy_score(y_test, predict_test)),1)



# Measure the sensitivity

Sensitivity = round(100*TP/float(TP+FN))



# Measure specificity

Specificity = round(100*TN/float(TN+FP),1)





# Precision

Precision = round(100*TP/float(TP+FP))



# Recall

Recall = round(100*TP/float(TP+FN),1)
Metrics_test = {'Accuracy': Accuracy, 

     'Precision': Precision,

     'Recall': Recall}
print(Metrics_train)

print(Metrics_test)
# Response variable

y_2 = telecom_model['churn']

# Feature variable

x_2 = telecom_model.drop('churn', axis=1)
# Split the data between train and test

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2, y_2, train_size = 0.7, test_size = 0.3, random_state = RANDOM_STATE)
# Balancing - smote

list_column_names = list(x_2.columns)

smote_2 = SMOTE()

x_train_2, y_train_2 = smote_2.fit_sample(x_train_2, y_train_2)
print('Churn % after smote')

print((y_train_2 != 0).sum()/(y_train_2 == 0).sum())
scaler = StandardScaler()

x_train_2 = scaler.fit_transform(x_train_2)

x_test_2 = scaler.transform(x_test_2)

df_train_2 = pd.DataFrame(x_train_2, columns = list_column_names)
logreg = LogisticRegression()



rfe = RFE(logreg, 20)

rfe = rfe.fit(df_train_2,y_train_2)

print(rfe.support_)

print(rfe.ranking_)
# Variables from RFE

col = df_train_2.columns[rfe.support_]
#1 Assessing the model with stats model

import statsmodels.api as sm

x_train_sm = sm.add_constant(df_train_2[col])

logm1 = sm.GLM(y_train_2, x_train_sm, family=sm.families.Binomial())

res = logm1.fit()

res.summary()
x_train_sm.columns
# Check VIF values

vif = pd.DataFrame()

x = x_train_sm

vif['Features'] = x.columns

vif['VIF'] = [variance_inflation_factor(x_train_sm.values, i) for i in range(x_train_sm.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by='VIF', ascending=False)

vif
# Dropping Columns 

#x_train_sm.drop(['offnet_mou_6'], axis = 1, inplace = True)
# 2

x_train_sm.columns
#2 Assessing the model with stats model

import statsmodels.api as sm

logm2 = sm.GLM(y_train_2, x_train_sm, family=sm.families.Binomial())

res2 = logm2.fit()

res2.summary()
# Check VIF values

vif = pd.DataFrame()

x = x_train_sm

vif['Features'] = x.columns

vif['VIF'] = [variance_inflation_factor(x_train_sm.values, i) for i in range(x_train_sm.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by='VIF', ascending=False)

vif
## Drop Columns 

x_train_sm.drop(['loc_ic_mou_8'], axis = 1, inplace = True)
# 3

x_train_sm.columns
# Assess the model with stats model

import statsmodels.api as sm

logm3 = sm.GLM(y_train_2, x_train_sm, family=sm.families.Binomial())

res3 = logm3.fit()

res3.summary()
# Check VIF values

vif = pd.DataFrame()

x = x_train_sm

vif['Features'] = x.columns

vif['VIF'] = [variance_inflation_factor(x_train_sm.values, i) for i in range(x_train_sm.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by='VIF', ascending=False)

vif
## Drop Columns 

#x_train_sm.drop(['onnet_mou_6'], axis = 1, inplace = True)
# 4

x_train_sm.columns
# Assess the model with stats model

import statsmodels.api as sm

logm4 = sm.GLM(y_train_2, x_train_sm, family=sm.families.Binomial())

res4 = logm4.fit()

res4.summary()
# Check VIF values

vif = pd.DataFrame()

x = x_train_sm

vif['Features'] = x.columns

vif['VIF'] = [variance_inflation_factor(x_train_sm.values, i) for i in range(x_train_sm.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by='VIF', ascending=False)

vif
contribution = 100*res4.params.sort_values(ascending = False)/sum(abs(res4.params.values))

contribution