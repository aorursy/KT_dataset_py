# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Set your own project id here

PROJECT_ID = 'your-google-cloud-project'

from google.cloud import automl_v1beta1 as automl

automl_client = automl.AutoMlClient()

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df
df.head()
df.info()
df1 = df.loc[:,df.notnull().all()]

df1.info()
df2 = df.dropna(axis=0)

df2.info()
df2.head()
#cols = [0,1,2,3,5,14,15]

#df3 = df2.drop(df2.columns[cols],axis=1)

#df3.info()



df3 = df2[["neighbourhood","price"]].groupby("neighbourhood").mean().reset_index()

print(len(df3))

df3.head()
df4 = df3.set_index('neighbourhood')
from matplotlib import pyplot as plt

from scipy.cluster import hierarchy



plt.figure(figsize=(40, 40)) #graph size

plt.title("Dendrograms of New York City Airbnb", fontsize= 35)



d = hierarchy.linkage(df4, method='ward')

hierarchy.dendrogram(d, leaf_rotation=90, leaf_font_size=12, labels=df4.index)



plt.show()