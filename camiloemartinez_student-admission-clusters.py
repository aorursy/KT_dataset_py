import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/Admission_Predict.csv")



# Print the head of df

print(df.head())



# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)
df.describe()
# Compute the correlation matrix

corr=df.iloc[:,1:9].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.jointplot(x="GRE Score", y="CGPA", data=df)
#Correlation for different deciles of the most important variable to be admitted

def corr_parts(data,x,y,z,z_cutoff):

    df_temp = data.loc[data[z] > z_cutoff]

    return df_temp[x].corr(df_temp[y])



dl_contrast = np.around(np.percentile(df['CGPA'], np.arange(0, 100, 10)),1)



corr_sop = []

for x in dl_contrast:

    corr_sop.append(corr_parts(df,'SOP','Chance of Admit ','CGPA', x ))

corr_lor = []

for x in dl_contrast:

    corr_lor.append(corr_parts(df,'LOR ','Chance of Admit ','CGPA', x ))

    

result = pd.DataFrame ({'decile': dl_contrast, 'sop': corr_sop, 'lor': corr_lor  })

result = result.melt('decile', var_name='vars',  value_name='corr')



# Set up the seaborn figure

sns.factorplot(x="decile", y="corr", hue='vars', data=result)
#Scaling the continuos variables

df_scale = df.copy()

scaler = preprocessing.StandardScaler()

columns =df.columns[1:7]

df_scale[columns] = scaler.fit_transform(df_scale[columns])

df_scale.head()
#Elbow graph

ks = range(1, 6)

inertias = []



for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=k)

    

    # Fit model to samples

    model.fit(df_scale.iloc[:,1:])

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

# Plot ks vs inertias

plt.plot(ks, inertias, '-o')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
# Create a KMeans instance with 3 clusters: model

model = KMeans(n_clusters=3)



# Fit model to points

model.fit(df_scale.iloc[:,2:9])



# Determine the cluster labels of new_points: labels

df_scale['cluster'] = model.predict(df_scale.iloc[:,2:9])



df_scale.head()
# Create PCA instance: model

model_pca = PCA()



# Apply the fit_transform method of model to grains: pca_features

pca_features = model_pca.fit_transform(df_scale.iloc[:,2:9])



# Assign 0th column of pca_features: xs

xs = pca_features[:,0]



# Assign 1st column of pca_features: ys

ys = pca_features[:,1]



# Scatter plot xs vs ys

sns.scatterplot(x=xs, y=ys, hue="cluster", data=df_scale)
sns.boxplot(x="cluster", y="Chance of Admit ", data=df_scale, palette="Set2" )
centroids = model.cluster_centers_

df_scale.iloc[:,1:10].groupby(['cluster']).mean()
sns.heatmap(df_scale.iloc[:,1:10].groupby(['cluster']).mean(), cmap="YlGnBu")
pd.DataFrame(df_scale['cluster'].value_counts(dropna=False))
g = sns.PairGrid(df_scale.iloc[:,1:10], hue="cluster", palette="Set2")

g.map(plt.scatter);