import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
d0 = pd.read_csv("/kaggle/input/mnist-dataset/train.csv")
print(d0.head(5))
l = d0['label']
d= d0.drop('label',axis=1)
print(d)
labels = l.head(1500)
data = d.head(1500)
print('the shape of sampledata=',data.shape)
from sklearn import decomposition
pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(data)
print('shape  of  pca-reduced.shape ',pca_data.shape)
import seaborn as sn
pca_data = np.vstack((pca_data.T,labels)).T
pca_df = pd.DataFrame(data= pca_data,columns=('1st_principal','2nd_principal','label'))
sn.FacetGrid(pca_df, hue= "label",height =6).map(plt.scatter,'1st_principal','2nd_principal').add_legend()
plt.show()
pca.n_components = 784

pca_data = pca.fit_transform(data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)

cum_var_explaind = np.cumsum(percentage_var_explained)

#plot the pca specturm
plt.figure(1, figsize=(6,4))
plt.clf()
plt.plot(cum_var_explaind,linewidth = 2)
plt.axis('tight')
plt.grid()
plt.show()
label_1500 = l.head(1500)
print(label_1500)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sn
standardized_data = StandardScaler().fit_transform(data)
data_1000 = standardized_data[0:1500,:]
model = TSNE(n_components = 2)
tane_data = model = model.fit_transform(data_1000)
# creating a data fram which help us in ploting the result
tane_data = np.vstack((tane_data.T,label_1500)).T
tane_df = pd.DataFrame(data=tane_data, columns=('Dim_1','Dim_2','lable'))
# ploting the result of tsne
sn.FacetGrid(tane_df,hue='lable',height = 6).map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show()
model = TSNE(n_components = 2 , random_state = 0 ,perplexity = 50 , n_iter = 5000)
tane_data = model = model.fit_transform(data_1000)
# creating a data fram which help us in ploting the result
tane_data = np.vstack((tane_data.T,label_1500)).T
tane_df = pd.DataFrame(data=tane_data, columns=('Dim_1','Dim_2','lable'))
# ploting the result of tsne
sn.FacetGrid(tane_df,hue='lable',height = 6).map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show()
model = TSNE(n_components = 2 , random_state = 0 ,perplexity = 100 , n_iter = 5000)
tane_data = model = model.fit_transform(data_1000)
# creating a data fram which help us in ploting the result
tane_data = np.vstack((tane_data.T,label_1500)).T
tane_df = pd.DataFrame(data=tane_data, columns=('Dim_1','Dim_2','lable'))
# ploting the result of tsne
sn.FacetGrid(tane_df,hue='lable',height = 6).map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show()
