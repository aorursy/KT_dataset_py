import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

d0 = pd.read_csv('../input/Sales_Transactions_Dataset_Weekly.csv')
# save the labels into a variable l. Creating lables just for visulization purpose after doing PCA
l = d0['Product_Code']
# Drop the label feature and store the sales data in d. Separating the lable data
d = d0.drop("Product_Code",axis=1)
d.drop(d.columns[0:54], axis=1, inplace=True) # this is a normalized data
raw=d0[d0.columns[1:53]] # this is a raw data
raw.head() # how our data looks after seperating the lables
print(d.shape)
print(l.shape)
# initializing the pca
from sklearn import decomposition
pca = decomposition.PCA()
# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data = pca.fit_transform(raw)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)
pca.n_components = 52

pca_data_pca = pca.fit_transform(raw)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained,linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()
# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data = pca.fit_transform(d)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)
pca.n_components = 52

pca_data_pca = pca.fit_transform(d)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained,linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()