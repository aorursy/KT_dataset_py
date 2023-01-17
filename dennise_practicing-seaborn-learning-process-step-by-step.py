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
data=pd.read_csv("../input/data.csv",encoding = 'utf8')
data.info()
# First line makes the float numbers readable (not e^something)

pd.options.display.float_format = '{:,.2f}'.format

data.describe()
data.head()
data["Region"].nunique()
data["Date"].nunique()
maxi_views=data.groupby("Region").describe().Streams["max"].sort_values(ascending=False)

maxi_views.head(10)
data[(data["Streams"].isin(maxi_views.head(10)))&(data["Position"]==1)].sort_values("Streams", ascending=False)
type(data["Date"][0])
data["Date"][20000]
data["Date"].min()
data["Date"].max()
# For proper plotting time needs to be in a non-string format

from datetime import datetime



data["Date2"]=data["Date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
type(data["Date2"][10])
data["Date2"][10]
global_total_streams=data[(data["Region"]=="global")].groupby("Date2").sum().drop("Position",axis=1)

global_total_streams.head()
plt.plot(global_total_streams)
total_views_regions=data.groupby("Region").sum().drop("Position",axis=1).drop("global",axis=0)

total_views_regions.head()
regions=data["Region"].unique()

regions
ax=sns.distplot(total_views_regions)

ax.set(xlabel="Total Streams per Region")
total_views_regions_wo_US=data.groupby("Region").sum().drop("Position",axis=1).drop(["global","us"],axis=0)
ax=sns.distplot(total_views_regions_wo_US, bins=20)

ax.set(xlabel="Total Streams",ylabel="% of regions within respective cluster of streams")
streams_per_position=data.groupby("Position").sum()

streams_per_position.head()
ax=sns.distplot(streams_per_position)

ax.set(xlabel="Total Streams",ylabel="% of positions within respective cluster of streams")
#Group by titles per region to have one list of titles per region

tracks_per_region=data.groupby(["Region","Track Name"]).sum()

tracks_per_region.head()
tracks_per_region=tracks_per_region["Streams"]

tracks_per_region.head()
region_matrix=pd.DataFrame(index=regions,columns=regions)
for region1 in region_matrix.index:

    for region2 in region_matrix.index:

        count=0

        for element in tracks_per_region[region1].index:

            if element in tracks_per_region[region2].index:

                count+=1

        region_matrix[region1][region2]=count

region_matrix
region_matrix = region_matrix.astype(int)
sns.heatmap(region_matrix)
sns.clustermap(region_matrix)
core_markets=total_views_regions.sort_values("Streams",ascending=False)[0:14].index
core_markets
region_matrix_core=region_matrix[list(core_markets)]
region_matrix_core=region_matrix_core.loc[list(core_markets)]
sns.heatmap(region_matrix_core)
#Normalizing the Dataset to abstract from size

region_matrix_core_norm=(region_matrix_core-region_matrix_core.mean())/region_matrix_core.std()
sns.heatmap(region_matrix_core_norm)
sns.clustermap(region_matrix_core_norm)