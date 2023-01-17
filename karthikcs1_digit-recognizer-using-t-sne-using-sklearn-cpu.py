from tsnecuda import TSNE as tsnecuda
import numpy as np 

import pandas as pd 

from  sklearn.manifold import TSNE as sktsne

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')
print('The shape  of training Dataset: ', df_train.shape, '  The shape of testing Dataset:  ', df_test.shape)
df_train = df_train.head(10000)
Y = df_train[['label']]

X = df_train.drop('label', axis=1)
def plot_digit(digits):

    fig, axs = plt.subplots(1,len(digits),figsize=(2,2))

    for i, pixels in enumerate(digits):

        ax = axs[i]

        digit_data = pixels.values.reshape(28,28)

        ax.imshow(digit_data,interpolation=None, cmap='gray')

    plt.show()
plot_digit([X.iloc[0], X.iloc[20], X.iloc[201]])
scaled_X = pd.DataFrame(StandardScaler().fit_transform(X))

scaled_X.head()

tsne_data = sktsne(n_components=2,random_state=42, perplexity=30.0, n_iter=1000).fit_transform(X)

# !export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:

# tsne_model = tsnecuda(n_components=2, perplexity=30.0).fit_transform(X)

# tsne_data = tsne_model.fit_transform(scaled_X)
tsne_df = pd.DataFrame(tsne_data)

tsne_df = pd.concat([tsne_df,Y], axis=1)

# tsne_df
sns.FacetGrid(tsne_df, hue="label" , size=6).map(plt.scatter, 0, 1).add_legend()

plt.show()