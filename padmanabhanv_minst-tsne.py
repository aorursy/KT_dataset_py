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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.manifold import TSNE
#../input/digit-recognizer/test.csv

train_data = pd.read_csv("../input/digit-recognizer/train.csv")
print(train_data.shape)
data = train_data.drop("label",axis=1)
print(data.shape)
target = train_data.loc[:, ['label']].values
print(target.shape)

tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000)
tsne_data = tsne.fit_transform(data)
print(tsne_data.shape)

result_data = np.vstack((tsne_data.T, target.T)).T
result_df = pd.DataFrame(data=result_data, columns=('first_principal', 'second_principal', 'label'))

sn.FacetGrid(result_df, hue='label', height=5).map(plt.scatter, 'first_principal', 'second_principal').add_legend()
