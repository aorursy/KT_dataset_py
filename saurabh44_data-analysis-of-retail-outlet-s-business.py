# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import plotly.express as px

from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA



# #Reading all data files....

dfs = pd.read_excel('/kaggle/input/irisdata/Test Analyse de donnes v2.xlsx', sheet_name="Data")

dfs['PR'] = (dfs['Client Value']/dfs['Panel Value'])*100

dfs['Dis/PV'] = dfs['Distance Meter']/dfs['Panel Value']

dfs.rename(columns={"Distance Meter": "Distance_Meter"}, inplace=True)
def cat(x):

    if x<=1000 and x>=0:

        return "0-1000"

    if x<=2000 and x>1000:

        return "1001-2000"

    if x<=3000 and x>2000:

        return "2001-3000"  

    if x<=4000 and x>3000:

        return "3001-4000"  

    if x<=5000 and x>4000:

        return "4001-5000"  

    if x<=6000 and x>5000:

        return "5001-6000"  

    if x<=7000 and x>6000:

        return "6001-7000"  

    if x<=8000 and x>7000:

        return "7001-8000"  

    if x<=9000 and x>8000:

        return "8001-9000"

    if x<=10000 and x>9000:

        return "9001-10000"



dfs['dist_group'] = dfs['Distance_Meter'].apply(lambda x: cat(x))

dfs.head()
dist_client_dfs = dfs.groupby("dist_group",as_index=False)["Client Value"].sum()

fig = px.bar(dist_client_dfs, x="dist_group", y="Client Value", orientation='v',text="Client Value")

fig.show()
# plt.scatter(dfs['Distance Meter'],dfs['Client Value'])

fig = px.scatter(dfs, x="Distance_Meter", y="Client Value")

fig.show()
dist_panel_dfs = dfs.groupby("dist_group",as_index=False)["Panel Value"].sum()



fig = px.bar(dist_panel_dfs, x="dist_group", y="Panel Value", orientation='v',text="Panel Value")

fig.show()
fig = px.scatter(dfs, x="Distance_Meter", y="Panel Value")

fig.show()
df1 = dfs.drop(['Iris Name','Iris Code','PR','Dis/PV','dist_group'],axis=1)

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(df1)

kmeans_model = KMeans(n_clusters=3, max_iter=100)

X = kmeans_model.fit(X_scaled)

labels=kmeans_model.labels_.tolist()

l = kmeans_model.fit_predict(X_scaled)

dfs['Cluster_Segment'] = l

dfs.head()
pca = PCA(n_components=2).fit(X_scaled)

datapoint = pca.transform(X_scaled)

get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure

label1 = ["#FFFF00", "#008000","#0000FF"]

color = [label1[i] for i in labels]

plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_

centroidpoint = pca.transform(centroids)

plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150,c='#000000')

plt.show()
# Grouping Penetration rate on the basis of Cluster_Segment

avg_PR_df = dfs.groupby("Cluster_Segment",as_index=False)["PR"].mean()

avg_PR_df.head()
# Plotting Penetration rate v/s Cluster_Segment 

fig = px.bar(avg_PR_df, x="Cluster_Segment", y="PR", orientation='v',text="PR")

fig.show()
# Grouping Client Value on the basis of Cluster_Segment

client_share_df = dfs.groupby("Cluster_Segment",as_index=False)["Client Value"].sum()

client_share_df.head()
# Plotting percentage share of CLient Value across Cluster_Segment

fig = px.pie(client_share_df, values='Client Value', names='Cluster_Segment', title='Client share across segments')

fig.show()
dfs['Dis/PV'].hist(bins=100)
Disposable_dfs = pd.read_excel('/kaggle/input/irisrevenue/BASE_TD_FILO_DISP_IRIS_2014.xls',header=5)

Declared_dfs = pd.read_excel('/kaggle/input/irisrevenue/BASE_TD_FILO_DEC_IRIS_2014.xls',header=5)
merged_df = Disposable_dfs.append(Declared_dfs, ignore_index=True, sort=False)

merged_income_df=merged_df[["IRIS","LIBIRIS","DISP_MED14","DEC_MED14"]]

merged_income_df['DISP_MED14'].fillna(0,inplace=True)

merged_income_df['DEC_MED14'].fillna(0,inplace=True)
province_income_df = merged_income_df.groupby("LIBIRIS",as_index=False)["DISP_MED14","DEC_MED14"].sum()
sum_column = province_income_df["DISP_MED14"] + province_income_df["DEC_MED14"]

province_income_df["total_median_income"] = sum_column

ile_de_france_df=province_income_df.loc[province_income_df['LIBIRIS'] == 'Ile de France']

ile_de_france_modified=ile_de_france_df.total_median_income

ile_de_france_df.head()
def income(x):

    if int(x) > int(ile_de_france_modified):

        return "1"

    else:

        return "0"



province_income_df['province_Exceding_iledefrance'] = province_income_df['total_median_income'].apply(lambda x: income(x))

province_income_df.head()