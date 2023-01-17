# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

wine = pd.read_csv("../input/wine-data/wine_data.csv")

wine.head()
#Dropping Index

wine = wine.iloc[:,1:] 

wine.head()
#normalizing the values



from sklearn.preprocessing import scale 

wine_norm = scale(wine) 

wine_norm


from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



pca = PCA()

pca_values = pca.fit_transform(wine_norm)



# The amount of variance that each PCA explains

var = pca.explained_variance_ratio_ 

plt.plot(var)

pd.DataFrame(var)

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100) 

var1

# Variance plot for PCA components obtained 

plt.plot(var1,color="red")



#storing PCA values to a data frame

new_df = pd.DataFrame(pca_values[:,0:4])

new_df
