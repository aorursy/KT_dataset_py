import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
print(os.listdir("../input/fashionmnist"))
df = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
df.head()
label = df.label.astype(np.int)
df.drop("label", axis=1, inplace=True)
df.shape
def show_images(ids, data=df):
    pixels = np.array(data.iloc[ids])
    pixels = pixels.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    print("label: ", label.iloc[ids])
    

show_images(10)
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(df)
standardized_data.shape
sample_data = standardized_data
covariance_matrix = np.matmul(sample_data.T, sample_data)
print(covariance_matrix.shape)
from scipy.linalg import eigh

values, vectors = eigh(covariance_matrix)
print(vectors.shape)
print("Last 10 eigen values:")
print(values[:][-10:])
print("\nCorresponding vectors:")
print(vectors[-10:])
values = values[-2:]
vectors = vectors[:,-2:]
vectors = vectors.T
print("Shape of eigen value: ", values.shape)
print("Shape of eigen vectors: ", vectors.shape)
reduced_data = np.matmul(vectors, sample_data.T)
print("Reduced data shape: ", reduced_data.shape)
reduced_data = np.vstack((reduced_data, label))
reduced_data = reduced_data.T
reduced_df = pd.DataFrame(reduced_data, columns=['X', 'Y', 'label'])
reduced_df.label = reduced_df.label.astype(np.int)
reduced_df.head()
import seaborn as sns
reduced_df.dtypes
g = sns.FacetGrid(reduced_df, hue='label', size=12).map(plt.scatter, 'X', 'Y').add_legend()