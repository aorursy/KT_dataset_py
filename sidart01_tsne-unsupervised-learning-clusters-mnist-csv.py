# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import manifold
%matplotlib inline

data=pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
target=data.label
type(target)

target_array= target.to_numpy()
target_array
data=data.drop(columns=['label'])
data
data_array= data.to_numpy()
data_array
single_image= data_array[241,:].reshape(28,28)

plt.imshow(single_image, cmap='gray')
tsne= manifold.TSNE(n_components=2, random_state=42)

transformed_data_array= tsne.fit_transform(data_array[:3000,:])
tsne_df= pd.DataFrame(np.column_stack((transformed_data_array, target_array[:3000])), columns=["x","y","label"])
tsne_df.loc[:,"label"] = tsne_df.label.astype(int)

grid = sns.FacetGrid(tsne_df, hue="label", size=8)
grid.map(plt.scatter,"x","y").add_legend()









