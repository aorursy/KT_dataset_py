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
dataset = pd.read_csv("../input/wine-customer-segmentation/Wine.csv")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(dataset)
s_data = sc.transform(dataset)
from sklearn.decomposition import PCA 
pca = PCA(n_components = 2) 
pca.fit(s_data) 
x_pca = pca.transform(s_data) 
pca = PCA(n_components = 2) 
pca.fit(s_data) 
x_pca = pca.transform(s_data) 
print('Previos',dataset.shape)
print('Reduced ',x_pca.shape)
plt.figure(figsize =(8, 6)) 
plt.scatter(x_pca[:, 0], x_pca[:, 1]) 
plt.xlabel('First Principal Component') 
plt.ylabel('Second Principal Component')