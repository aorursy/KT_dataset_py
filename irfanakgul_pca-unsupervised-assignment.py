# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from warnings import filterwarnings

filterwarnings('ignore')



import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp

from sklearn.cluster import KMeans



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/usarrests/USArrests.csv").copy()

df.index = df.iloc[:,0]

df = df.iloc[:,1:5]

del df.index.name

df.head()
#once standartlastirma islemi yapmamiz lazim. sonra asil modelimizi kuracagiz,

from sklearn.preprocessing import StandardScaler



df = StandardScaler().fit_transform(df)

df[0:5,0:5]
# asil PCA modelimiz

from sklearn.decomposition import PCA

pca = PCA(n_components = 3)  # kac bilesene ayrilacagini belirler 3 e ayrilsin dedik

pca_fit = pca.fit_transform(df)
bilesen_df = pd.DataFrame(data = pca_fit, 

                          columns = ["birinci_bilesen","ikinci_bilesen","ucuncu_bilesen"])



#yukaridaki 4 degiskeni 3 e indirgedik.
bilesen_df.head()
pca.explained_variance_ratio_  



# aciklanan varyans orani. her bilesenin varyansi. yani verinin aciklanma orani, 

# eger yuksekse,  bilesen sayisi arttirilabilir. kac degisken olacagini bu yorumlanaarak yapilabilir.

# genelde 2 yada 3 bilesen olmasi daha uygundur, 
pca = PCA().fit(df)
# kac bilesen olursa aciklanma oranini gosteren plot cizimi. 

plt.plot(np.cumsum(pca.explained_variance_ratio_))