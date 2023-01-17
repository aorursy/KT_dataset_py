# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import dataset
gps_ds = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv") #edit this
gps_ds.head(5)
#gps_ds.info()

# dropping one row from the dataset which is of poor quality before datatype casting
gps_ds = gps_ds[gps_ds.App != "Life Made WI-Fi Touchscreen Photo Frame"]

# Column type casting

gps_ds["Reviews"] = pd.to_numeric(gps_ds["Reviews"])
gps_ds["Rating"] = pd.to_numeric(gps_ds["Rating"])

# remove the currency code and then cast to numeric
gps_ds["Price"] = gps_ds["Price"].map(lambda x:x.lstrip('$'))
gps_ds["Price"] = pd.to_numeric(gps_ds["Price"])

gps_ds["Last Updated"] = pd.to_datetime(gps_ds["Last Updated"])

# remove the symbols + and , and then cast to numeric
gps_ds["Installs"] = gps_ds["Installs"].map(lambda x:x.rstrip('+'))
gps_ds["Installs"] = gps_ds["Installs"].str.replace(',','')
gps_ds["Installs"] = pd.to_numeric(gps_ds["Installs"])
#################################### end of casting ####################################

#gps_ds.info()
# Step 2.1: Let's check the NaN stats(count) across the dataframe first
print(gps_ds.isnull().sum())

# Step 2.2: We can infer that, there is about 13.59% records with missing values for the 'Rating' attribute. We can impute this with 0s because:
# It is safe to assume that apps whose reviews are filled with NaN could mean that there has been no reviews yet. Since the other records have count of reviews in this dataset, let us replace NaN with 0 safely and avoid losing these records otherwise.

gps_ds["Reviews"] = gps_ds["Reviews"].fillna(0)
print("1. Most popular categories in terms of no. of apps:\n")
print(gps_ds.groupby(["Category","Type"])["Category"].count().sort_values(ascending=False).head(10))
## Let us visualise the distribution of apps as a stacked bar chart were blue and orange colors denote Free and Paid apps respectively.
apps_by_category = gps_ds.groupby(["Category","Type"])["App"].count()
apps_by_category = apps_by_category.unstack()
print('\n')
apps_by_category.plot(kind='bar',stacked=True,figsize=(10,5))
plt.title("Distribution of free and paid apps by category")
plt.xlabel("Category")
plt.ylabel("No. of Apps")
plt.show()
# Let us convert the no. of Installs into categorical data for a better summary
buckets = pd.cut(gps_ds.Installs,bins=[0,1000,100000,100000000,10000000000],labels=['0-1000','1001-100000','100001-100000000','100000001-10000000000'])
gps_ds.insert(5,"downloads_range",buckets)

# splitting the main df into two:
apps_by_download_free = gps_ds.loc[gps_ds['Type'] == "Free"]
apps_by_download_paid = gps_ds.loc[gps_ds['Type'] == "Paid"]
# Create a subplot of dimension 1 row X 2 columns
installs_free = apps_by_download_free.groupby(["downloads_range","Type"])["Type"].count()
installs_paid = apps_by_download_paid.groupby(["downloads_range","Type"])["Type"].count()

fig = plt.figure()

# cell 1x1 of a 1x2 grid, for Free Apps category and cell 1x2 for Paid Apps:

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

installs_free.plot(kind='bar',x='downloads_range', ax=ax1, legend=False,title="Free Apps",figsize=(10,6))
installs_paid.plot(kind='bar',x='downloads_range', ax=ax2, legend=False,title="Paid Apps",figsize=(10,6))

ax1.set(xlabel='total downloads bucket', ylabel='No. of Installs')
ax2.set(xlabel='total downloads bucket', ylabel='No. of Installs')

plt.show()
fig2 = plt.show()

#df.groupby('country')['unemployment'].mean().sort_values().plot(kind='barh', ax=ax2)
apps_by_download_paid = apps_by_download_paid.drop_duplicates(keep='first')
apps_by_download_paid.head(5)

df1 = apps_by_download_paid.groupby(["Category"])["Rating"].median().sort_values(ascending=False).to_frame()
df2 = apps_by_download_free.groupby(["Category"])["Rating"].median().sort_values(ascending=False).to_frame()
rating_comparison = df1.join(df2,on='Category',lsuffix='_paid_apps',rsuffix='_free_apps').sort_values('Category')

fig3 = rating_comparison.plot(kind='bar',y=['Rating_paid_apps','Rating_free_apps'],title='Performance Free vs. Paid Apps',figsize=(20,10))