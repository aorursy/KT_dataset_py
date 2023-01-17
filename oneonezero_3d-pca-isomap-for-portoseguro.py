import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



%matplotlib inline

plt.style.use('ggplot')
X = pd.read_csv("../input/train.csv")

y = X["target"]

X.drop(["target", "id"], axis = 1, inplace = True)

X.columns.values
def filter_cat(df):

    for x in df.columns.values:

        if x[-3:] == "cat":

            df.drop([x], axis = 1, inplace = True)

    return df



X_filt = filter_cat(X)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))



X_filt_scale = X_filt.copy()



for x in X_filt_scale.columns.values:

    if not x[-3:] == "bin":

        X_filt_scale[x] = scaler.fit_transform(X_filt[x].values.reshape(-1,1))
fig, axs = plt.subplots(1,2, figsize=(10, 5))



plot_data = pd.DataFrame(X_filt.max()-X_filt.min(), columns = ["magnitude"]).sort_values(by = "magnitude")

plot_data.plot.area(ax=axs[0], title = "Before Scaling", use_index = False, colormap = "Blues_r")



plot_data_2 = pd.DataFrame(X_filt_scale.max()-X_filt_scale.min(), columns = ["magnitude"]).sort_values(by = "magnitude")

plot_data_2.plot.area(ax=axs[1], title = "After Scaling", use_index = False, colormap = "Blues_r")
from sklearn.decomposition import PCA



pca = PCA(n_components = X_filt_scale.shape[1])

X_PCA = pca.fit_transform(X_filt_scale)

pca.explained_variance_ratio_.cumsum()[0:3]
def plot_pca(X, y, opacity_0, opacity_1):

    my_color = pd.DataFrame(np.zeros((len(y), 4)))

    my_color.iloc[:,0] = (1-y)*0.5

    my_color.iloc[:,1] = y*0.5

    my_color.iloc[:,2] = (1-y)*0.5

    my_color.iloc[:,3] = y*opacity_0 + (1-y)*opacity_1



    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')



    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=my_color)



    ax.set_xlabel('PC1')

    ax.set_ylabel('PC2')

    ax.set_zlabel('PC3')

    plt.title('PCA of Data from PortoSeguro')



plot_pca(X_PCA, y, 0.6, 0.05)
from sklearn.model_selection import train_test_split



X_iso_train, forget1, y_iso_train, forget2 = train_test_split(X_filt_scale, y, train_size=7000, random_state=4)

y_iso_train.reset_index(drop= True, inplace = True)



print("size of dataset before the split: " + str(len(X_filt_scale)))

print("size of dataset after the split: " + str(len(X_iso_train)))
X_PCA = pca.fit_transform(X_iso_train)

plot_pca(X_PCA, y_iso_train, 1, 0.5)
from sklearn.manifold import Isomap



iso = Isomap(n_neighbors = 5, n_components = 3)

X_iso = iso.fit_transform(X_iso_train)

plot_pca(X_iso, y_iso_train, 1, 0.5)