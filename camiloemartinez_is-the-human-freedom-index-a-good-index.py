import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

from sklearn.decomposition import PCA



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/hfi_cc_2018.csv")



# Print the head of df

print(df.head())



# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)



print(np.min(df.year))
df.head()
countries = df.dropna(axis=1, how='all')

countries = countries.fillna(countries.mean())

countries.iloc[:,4:110].head()
# Compute the correlation matrix

corr=countries.iloc[:,4:110].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Create a PCA instance: pca

pca = PCA(n_components=5)



# Fit the pipeline to 'samples'

pca.fit(countries.iloc[:,4:110])



pca_features = pca.transform(countries.iloc[:,4:110])



# Plot the explained variances

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.xticks(features)

plt.show()
pd.DataFrame(pca.components_)
# Assign 0th column of pca_features: xs

xs = pca_features[:,0]



# Assign 1st column of pca_features: ys

ys = countries['hf_score']



# Scatter plot xs vs ys

plt.scatter(xs, ys)

plt.axis('equal')

plt.show()