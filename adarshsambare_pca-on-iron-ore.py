## Importing the required libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
## Importing the data

df = pd.read_csv("../input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv",decimal=",",parse_dates=["date"],infer_datetime_format=True).drop_duplicates()
df.describe()
df.head()
## Checking null values in the data

df.isna().sum()
## Checking the corelation of all the inputs

plt.figure(figsize=(20, 15))

p = sns.heatmap(df.corr(), annot=True)
## Important parameters as described in the intro

imp = df.iloc[:,3:8]
# air flow and level is important but not that much

air_flow = df.iloc[:,8:15]

level    = df.iloc[:,15:22]
## Y is the output 

#  Dropping the % iron ore as it is more correalated to % silion

y = df.iloc[:,23:24]

y.describe()
imp.info()
imp.describe()
plt.figure(figsize=(15, 8))

p = sns.heatmap(imp.corr(), annot=True)
## Checking parameter graph

plt.figure(figsize=(10, 7))

#plt.hist(imp["Starch Flow"], alpha = 0.5, bins = 50, edgecolor = "Black", label = "01")

#plt.hist(imp["Amina Flow"], alpha = 0.5, bins = 50, edgecolor = "Black", label = "02")

#plt.hist(imp["Ore Pulp Flow"], alpha = 0.5, bins = 50, edgecolor = "Black", label = "03")

#plt.hist(imp["Ore Pulp pH"], alpha = 0.5, bins = 50, edgecolor = "Black", label = "04")

plt.hist(imp["Ore Pulp Density"], alpha = 0.5, bins = 50, edgecolor = "Black", label = "05")

plt.legend(loc = 'upper right')

plt.title("Histogram Of IMP")

plt.xlabel("")

plt.ylabel("Number of Occurence")

plt.show() 
air_flow.describe()
## Checking the corelation of air_flow

plt.figure(figsize=(15, 8))

p = sns.heatmap(air_flow.corr(), annot=True)
## Checking 2 parameter with each other

plt.figure(figsize=(10, 7))

#plt.hist(air_flow["Flotation Column 03 Air Flow"], alpha = 0.5, bins = 25, edgecolor = "Black", label = "03")

plt.hist(air_flow["Flotation Column 04 Air Flow"], alpha = 0.5, bins = 25, edgecolor = "Black", label = "04")

plt.hist(air_flow["Flotation Column 05 Air Flow"], alpha = 0.5, bins = 25, edgecolor = "Black", label = "05")

#plt.hist(air_flow["Flotation Column 06 Air Flow"], alpha = 0.5, bins = 25, edgecolor = "Black", label = "06")

plt.legend(loc = 'upper right')

plt.title("Histogram")

plt.xlabel("Cubic Meter Per Hour")

plt.ylabel("Number of Occurence")

plt.show() 
plt.figure(figsize=(15, 8))

p = sns.heatmap(level.corr(), annot=True)
plt.figure(figsize=(10, 7))

plt.hist(level["Flotation Column 01 Level"], alpha = 0.5, bins = 25, edgecolor = "Black", label = "03")

plt.hist(level["Flotation Column 02 Level"], alpha = 0.5, bins = 25, edgecolor = "Black", label = "04")

#plt.hist(level["Flotation Column 03 Level"], alpha = 0.5, bins = 25, edgecolor = "Black", label = "05")

#plt.hist(level["Flotation Column 04 Level"], alpha = 0.5, bins = 25, edgecolor = "Black", label = "06")

plt.legend(loc = 'upper right')

plt.title("Histogram")

plt.xlabel("Nm³/h")

plt.ylabel("Number of Occurence")

plt.show() 
from sklearn.decomposition import PCA

from sklearn.preprocessing import scale
air_flow_scaled_1 = scale(air_flow.iloc[:,0:3])

air_flow_scaled_2 = scale(air_flow.iloc[:,3:5])

air_flow_scaled_3 = scale(air_flow.iloc[:,5:])
print(air_flow_scaled_1.shape)

print(air_flow_scaled_2.shape)

print(air_flow_scaled_3.shape)
## Only giving n_components as 1

pca = PCA(n_components=1)
## Storing the PCA

pca_air_flow_1 = pca.fit_transform(air_flow_scaled_1)

pca_air_flow_2 = pca.fit_transform(air_flow_scaled_2)

pca_air_flow_3 = pca.fit_transform(air_flow_scaled_3)
## Checking the variance

var_1 = pca.explained_variance_ratio_
## The variance is around 93% which is okay

var_1
pca_air_flow_1
## Storing the PCA in a dataframe

pca_air_flow = pd.DataFrame(data = pca_air_flow_1,columns = ["PCA_air_1"])

pca_air_flow['PCA_air_2'] = pca_air_flow_2

pca_air_flow['PCA_air_3'] = pca_air_flow_3
## Checking the PCA relationship

plt.figure(figsize=(15, 8))

p = sns.heatmap(pca_air_flow.corr(), annot=True)
## Sorting the level according to their corelation

level_scaled_1 = scale(level.iloc[:,0:3])

level_scaled_2 = scale(level.iloc[:,3:])
## Taking the n_components = 2 as 1 is giving 72%

pca2 = PCA(n_components=2)
## Applying the PCA

pca_level_1 = pca2.fit_transform(level_scaled_1)

pca_level_2 = pca2.fit_transform(level_scaled_2)
## Checking the variance

var_2 = pca2.explained_variance_ratio_
var_2
## Storing the PCA in a dataframe

pca_level = pd.DataFrame(data = pca_level_1,columns = ["PCA_level_1_1","PCA_level_1_2"])
pca_level.shape
pca_level.isna().sum()
pca_level_pd_2 = pd.DataFrame(data = pca_level_2,columns = ["PCA_level_2_1","PCA_level_2_2"])

pca_level_pd_2.shape
pca_level_pd_2.isna().sum()
pca_level["PCA_level_2_1"] = pca_level_pd_2["PCA_level_2_1"]

pca_level["PCA_level_2_2"] = pca_level_pd_2["PCA_level_2_2"]

pca_level.shape
pca_level.isna().sum()
## Checking the corelation

plt.figure(figsize=(15, 8))

p = sns.heatmap(pca_level.corr(), annot=True)
pca_air_level = pd.DataFrame(pca_air_flow)
pca_air_level["PCA_level_1_1"] = pca_level["PCA_level_1_1"]

pca_air_level["PCA_level_1_2"] = pca_level["PCA_level_1_2"]

pca_air_level["PCA_level_2_1"] = pca_level["PCA_level_2_2"]

pca_air_level["PCA_level_2_2"] = pca_level["PCA_level_2_2"]
pca_air_level.isna().sum()
pca_air_level.shape
## Checking the overall corealtion

plt.figure(figsize=(15, 8))

p = sns.heatmap(pca_air_level.corr(), annot=True)
## Checking the shape of different parameter

print(imp.shape)

print(pca_air_level.shape)

print(y.shape)
## Checking the null values

print(imp.isna().sum())

print(pca_air_level.isna().sum())

print(y.isna().sum())

# no null values
new_df = pd.DataFrame(y)
# addition of air_flow pca

new_df['PCA_air_1'] = pca_air_flow_1

new_df['PCA_air_2'] = pca_air_flow_2

new_df['PCA_air_3'] = pca_air_flow_3



# addition of level pca

new_df["PCA_level_1_1"] = pca_level["PCA_level_1_1"]

new_df["PCA_level_1_2"] = pca_level["PCA_level_1_2"]

new_df["PCA_level_2_1"] = pca_level["PCA_level_2_1"]

new_df["PCA_level_2_2"] = pca_level["PCA_level_2_2"]
new_df.shape
new_df.isna().sum()
new_df
plt.figure(figsize=(15, 8))

p = sns.heatmap(new_df.corr(), annot=True)
from sklearn.model_selection import train_test_split
train,test = train_test_split(df,test_size = 0.2)