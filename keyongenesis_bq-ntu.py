import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import folium

from folium.plugins import HeatMap



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Setting up BigQuery library

PROJECT_ID = 'bigquery-ml-geotab'

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID, location="US")

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

from google.cloud import automl_v1beta1 as automl

automl_client = automl.AutoMlClient()

from google.cloud.bigquery import magics

from kaggle.gcp import KaggleKernelCredentials

magics.context.credentials = KaggleKernelCredentials()

magics.context.project = PROJECT_ID



# load biquery commands

%load_ext google.cloud.bigquery
# Read data

df_train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

df_test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')
df_train.info()
df_test.info()
df_train.isnull().sum()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html

obj_df = df_train.select_dtypes(include=['object'])

# Return whether any element is True, potentially over an axis.

obj_df[obj_df.isnull().any(axis=1)].count()
# Checking for distribution of data BY UNIQUE INTERSECTION ID

fig = df_train.groupby(['City'])['IntersectionId'].nunique().sort_index().plot.bar()

fig.set_title('# of Intersections per city in train Set', fontsize=15)

fig.set_ylabel('# of Intersections', fontsize=15);

fig.set_xlabel('City', fontsize=17);
# let's see the distribution of traffic by month and date

plt.figure(figsize=(15,12))



plt.subplot(211)

g = sns.countplot(x="Hour", data=df_train, hue='City', dodge=True)

g.set_title("Distribution by hour and city", fontsize=20)

g.set_ylabel("Count",fontsize= 17)

g.set_xlabel("Hours of Day", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)



g.set_ylim(0, max(sizes) * 1.15)



plt.subplot(212)

g1 = sns.countplot(x="Month", data=df_train, hue='City', dodge=True)

g1.set_title("Hour Count Distribution by Month and City", fontsize=20)

g1.set_ylabel("Count",fontsize= 17)

g1.set_xlabel("Months", fontsize=17)

sizes=[]

for p in g1.patches:

    height = p.get_height()

    sizes.append(height)



g1.set_ylim(0, max(sizes) * 1.15)



plt.subplots_adjust(hspace = 0.3)



plt.show()