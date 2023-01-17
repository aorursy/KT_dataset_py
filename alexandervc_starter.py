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
import time 

import matplotlib.pyplot as plt

import seaborn as sns
filename = 'GSE67123_v6_scrna_10x_143_23548_Mouse_Embryo_HSCs_invivo_fromCytotrace.csv'

t0 = time.time()

df = pd.read_csv('/kaggle/input/genes-expressions-datasets-collection/' + filename, index_col= 0)

print(time.time() - t0,'seconds passed')

df
y = np.array( [ s.split('||')[0] for s in df.index] )

print(y[:15])

pd.Series(y).value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

#import umap 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X = df.values



t0=time.time()

fig = plt.figure(figsize = (20,4) )

fig.suptitle(filename.split('_')[0] + '\n'+str(X.shape) )

c = 0;  n_x_subplots = 2



c += 1; fig.add_subplot(1,n_x_subplots,c)  

X2 = PCA().fit_transform(X)

sns.scatterplot(x=X2[:,0], y=X2[:,1] , hue = y) 

plt.title('PCA')

plt.grid()



c += 1; fig.add_subplot(1,n_x_subplots,c)  

X2 = PCA().fit_transform(scaler.fit_transform(X) ) 

sns.scatterplot(x=X2[:,0], y=X2[:,1] , hue = y) 

plt.title('StandardScaler+PCA')

plt.grid()



plt.show()
list_files = os.listdir('/kaggle/input/genes-expressions-datasets-collection/') 



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

#import umap 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()





for filename in list_files:

    t0 = time.time()

    df = pd.read_csv('/kaggle/input/genes-expressions-datasets-collection/' + filename, index_col= 0)

    print(filename, 'shape:',  df.shape, np.round(time.time() - t0,2) ,'seconds passed')

    



    X = df.values

    y = np.array( [ s.split('||')[0] for s in df.index] )    



    t0=time.time()

    fig = plt.figure(figsize = (20,4) )

    fig.suptitle(filename.split('_')[0] + '\n'+str(X.shape) )

    c = 0;  n_x_subplots = 2



    c += 1; fig.add_subplot(1,n_x_subplots,c)  

    X2 = PCA().fit_transform(X)

    sns.scatterplot(x=X2[:,0], y=X2[:,1] , hue = y) 

    plt.title('PCA')

    plt.grid()



    c += 1; fig.add_subplot(1,n_x_subplots,c)  

    X2 = PCA().fit_transform(scaler.fit_transform(X) ) 

    sns.scatterplot(x=X2[:,0], y=X2[:,1] , hue = y) 

    plt.title('StandardScaler+PCA')

    plt.grid()



    plt.show()    