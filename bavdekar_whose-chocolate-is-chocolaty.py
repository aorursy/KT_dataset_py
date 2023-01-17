# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
choco_df = pd.read_csv("../input/flavors_of_cacao.csv")
choco_df.head()
choco_df.columns=['Maker', 'Bean Origin', 'REF', 'Review Date','Cocoa percent', 'Company Location', 'Rating', 'Bean Type',' Bean Country']
choco_df.head()
choco_df['Cocoa percent'] = choco_df['Cocoa percent'].str.replace('%','')
choco_df['Cocoa percent'] = pd.to_numeric(choco_df['Cocoa percent'])
choco_df['Company Location'].unique()
choco_df['Maker'].unique()
plt.subplots(figsize=(12,9))
sns.distplot(choco_df['Cocoa percent'],kde=False,color='red')
choco_country_rating = choco_df[['Company Location','Rating']]
choco_country_rating.head()
choco_country_rating_mean_std = choco_country_rating.groupby('Company Location',sort=False).mean()
choco_country_rating_mean_std=choco_country_rating_mean_std.reset_index()

choco_country_rating_mean_std.columns = [['Company Location','Rating Mean']]
choco_country_rating_mean_std.head()
plt.subplots(figsize=(12,9))
sns.barplot(x='Company Location',y='Rating Mean',data=choco_country_rating_mean_std.nlargest(20,'Rating Mean'))
plt.xticks(rotation=90)
plt.tight_layout
choco_df[(choco_df['Company Location']=='Chile')]['Company Location'].value_counts()
choco_df[(choco_df['Company Location']=='Netherlands')]['Company Location'].value_counts()
choco_df[(choco_df['Company Location']=='Amsterdam')]['Company Location'].value_counts()
choco_co_loc10 = choco_df['Company Location'].value_counts()>10
choco_co_loc10.head()
choco_df_10loc = choco_df.merge(choco_co_loc10.to_frame(),left_on='Company Location',right_index=True)
choco_df_10loc.head()
choco_highloc_rating = choco_df_10loc[choco_df_10loc['Company Location_y']==True].groupby('Company Location').mean().reset_index()
choco_highloc_rating.head()
plt.subplots(figsize=(12,9))
sns.barplot(x='Company Location',y='Rating',data=choco_highloc_rating.nlargest(20,'Rating'))
plt.xticks(rotation=90)
plt.tight_layout
plt.subplots(figsize=(16,12))
sns.swarmplot(x='Cocoa percent',y='Rating',data=choco_df)
plt.xticks(rotation=90)
plt.tight_layout
choco_df_high_rates = choco_df[choco_df['Rating']>=3.0]
choco_df_high_rates.head()
plt.subplots(figsize=(16,12))
choco_df_high_rates['Maker'].value_counts().head(20).plot.barh()
plt.xlabel('No. of bars')
plt.ylabel('Maker')
plt.tight_layout
plt.subplots(figsize=(16,12))
choco_df[(choco_df['Maker']=='Soma')]['Rating'].plot.hist()
plt.subplots(figsize=(16,12))
choco_df[choco_df['Maker']=='Soma'][' Bean Country'].value_counts().head(20).plot.barh()
plt.ylabel('Origin of bean for Soma chocolates')
plt.xlabel('No. of beans sourced from each country')
plt.tight_layout
plt.subplots(figsize=(16,12))
choco_df_high_rates[choco_df_high_rates['Maker']=='Soma']['Company Location'].value_counts().head(20).plot.barh()
plt.ylabel('Company Location')
plt.subplots(figsize=(16,12))
choco_df_high_rates['Company Location'].value_counts().head(20).plot.barh()
plt.xlabel('No. of bars')
plt.ylabel('Company Location')
plt.subplots(figsize=(16,12))
choco_df_high_rates[(choco_df_high_rates['Company Location']=='U.S.A.')]['Rating'].plot.hist()
plt.subplots(figsize=(16,12))
choco_df_high_rates[' Bean Country'].value_counts().head(20).plot.barh()
plt.xlabel('No. of Bars')
plt.ylabel('Origin of Bean')
plt.tight_layout
plt.subplots(figsize=(16,12))
choco_df_high_rates[(choco_df_high_rates[' Bean Country']=='Venezuela') & (choco_df_high_rates['Company Location']=='U.S.A.')]['Maker'].value_counts().head(20).plot.barh()
plt.xlabel('No. of Companies')
plt.ylabel('Company name')