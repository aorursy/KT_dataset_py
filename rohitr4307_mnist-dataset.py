import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import datasets

from sklearn import manifold



%matplotlib inline
data = datasets.fetch_openml(

    'mnist_784',

    version=1,

    return_X_y=True

)



pixel_values, targets = data

targets = targets.astype(int)
print('Shape of Pixel Values {}'.format(pixel_values.shape))

print('Shape of Target Values {}'.format(targets.shape))
single_image = pixel_values[5, :].reshape(28, 28)

plt.imshow(single_image, cmap='gray')

plt.show()
tsne = manifold.TSNE(n_components=2, random_state=141)

transformed_data = tsne.fit_transform(pixel_values[:3000, :])
tsne_df = pd.DataFrame(

    np.column_stack((transformed_data, targets[:3000])),

    columns=['x', 'y', 'targets']

)



tsne_df.loc[:, 'targets'] = tsne_df.targets.astype(int)
tsne_df.head(10)
grid = sns.FacetGrid(tsne_df, hue='targets', size=8)

grid.map(plt.scatter, 'x', 'y').add_legend()

plt.show()