from sklearn.datasets import load_digits

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import numpy as np

import matplotlib.pyplot as plt



import plotly.graph_objects as go
# We're gonna use an 8x8 digit dataset provided by sklearn



digits = load_digits()

print(digits.data.shape)



# Plot samples from the dataset



samples = np.random.randint(0, digits.data.shape[0], size=10)

fig, ax = plt.subplots(1, 10, figsize=(20, 2))



for i, sample in enumerate(samples):

    ax[i].imshow(digits.data[sample].reshape(8, 8), cmap='gray')

    ax[i].set_title(digits.target[sample], fontsize=14, pad=10)

    ax[i].set_xticks([])

    ax[i].set_yticks([])



plt.show()
# Create a PCA instance & Standard scaler

pca = PCA(n_components=3)

scaler = StandardScaler()



# Fit & Transform the original dataset

pca_array = pca.fit_transform(scaler.fit_transform(digits.data))

print(pca_array.shape)



fig = go.Figure(data=[go.Scatter3d(

    x=pca_array[:, 0],

    y=pca_array[:, 1],

    z=pca_array[:, 2],

    mode='markers',

    marker=dict(

        size=8,

        opacity=0.7,

        color=digits.target,

        colorscale='portland'

    )

)])



fig.update_layout(

    autosize=False, width=800, height=800,

    title='sklearn PCA on the digits dataset')

fig.show()
class myPCA:



    def __init__(self, n_components):

        self.n_components = n_components

        self.original_data = None

        self.principal_cmps = None

    

    def fit(self, data):

        cov_matrix = np.cov(data.T)

        _, v = np.linalg.eig(cov_matrix)

        self.principal_cmps = v.T[:self.n_components]

    

    def transform(self, data):

        num_samples = data.shape[0]

        pca_data = np.zeros((num_samples, self.n_components))

        

        for i, p_cmp in enumerate(self.principal_cmps):

            pca_data[:,i] = np.sum(p_cmp * data, axis=1).reshape(1, -1)

        

        return pca_data
digits_data = scaler.fit_transform(digits.data)



pca = myPCA(n_components = 3)

pca.fit(digits_data)

pca_data = pca.transform(digits_data)
# PCA array from sklearn

print(abs(pca_array.round(5)))



print('\n**************************\n')



# PCA array from my version

print(abs(pca_data.round(5)))