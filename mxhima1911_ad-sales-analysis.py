# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/clicks-conversion-tracking/KAG_conversion_data.csv")
df.head()
df.info()
print("Shape:",df.shape)
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
g = sns.FacetGrid(df, col="Gender", hue="Age")
g.map(plt.scatter, "Clicks", "Total_Conversion", alpha=.4)
g.add_legend();
g = sns.FacetGrid(df, col="age", hue="gender")
g.map(plt.scatter, "Clicks", "Total_Conversion", alpha=.4)
g.add_legend();
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "Impressions", "Clicks", alpha=.4)
g.add_legend();
g = sns.FacetGrid(df, col="age", hue="gender")
g.map(plt.scatter, "Impressions", "Clicks", alpha=.4)
g.add_legend();
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "Total_Conversion", "Approved_Conversion", alpha=.4)
g.add_legend();
g = sns.FacetGrid(df, col="age", hue="gender")
g.map(plt.scatter, "Total_Conversion", "Approved_Conversion", alpha=.4)
g.add_legend();
sns.scatterplot(x = 'Impressions' ,y='Clicks', hue='age', data=df)
g = sns.FacetGrid(df, col="age", hue="gender")
g.map(plt.scatter, "interest", "Approved_Conversion", alpha=.4)
g.add_legend();
plt.figure(figsize=(20,8))
sns.swarmplot(x = 'interest' ,y='Spent', data=df, alpha = .6)
plt.figure(figsize=(8,15))
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "fb_campaign_id", "Clicks", alpha=.4)
g.add_legend()
plt.figure(figsize=(8,15))
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "xyz_campaign_id", "Clicks", alpha=.4)
g.add_legend();
plt.figure(figsize=(8,4))
print('Facebook')
sns.scatterplot(x = 'fb_campaign_id' ,y='Approved_Conversion', hue='gender', data=df)

print('Others')
plt.figure(figsize=(8,4))
sns.scatterplot(x = 'xyz_campaign_id' ,y='Approved_Conversion', hue='gender', data=df)
plt.figure(figsize=(8,4))
sns.scatterplot(x = 'fb_campaign_id' ,y='Impressions', data=df)
plt.figure(figsize=(8,4))
sns.scatterplot(x = 'xyz_campaign_id' ,y='Impressions', data=df)
plt.figure(figsize=(8,4))
sns.scatterplot(x = 'fb_campaign_id' ,y='Spent', data=df)
plt.figure(figsize=(8,4))
sns.scatterplot(x = 'xyz_campaign_id' ,y='Spent', data=df)
plt.figure(figsize=(8,6))
sns.scatterplot(x = 'fb_campaign_id' ,y='Approved_Conversion', data=df)
plt.figure(figsize=(8,6))
sns.scatterplot(x = 'xyz_campaign_id' ,y='Approved_Conversion', data=df)
df['xyz_campaign_id'].unique()
