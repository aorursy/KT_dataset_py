# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.decomposition import PCA #principal component analysis module

from sklearn.cluster import KMeans #KMeans clustering

import matplotlib.pyplot as plt #Python defacto plotting library

import seaborn as sns # More snazzy plotting library

%matplotlib inline

# Any results you write to the current directory are saved as output.
battle=pd.read_csv('../input/battles.csv') #reads csv and creates the dataframe called battle

battle.shape

death=pd.read_csv('../input/character-deaths.csv') #reads csv and creates the dataframe called death

death.shape

pred=pd.read_csv('../input/character-predictions.csv')#reads csv and creates the dataframe called pred

pred.shape
battle.head(10)
death.head(10)
pred.head(10)
battle.corr()

"""

1. defender_size is negatively correlated with battle #: not sure if this means anything 

2. major_death and major_capture are negatively correlated with battle_number: again not meaningful

3. attacker_size and major capture are positively correlated: so more attackers translates to higher major_capture;



    """
death.corr()
pred.corr()
correlation=battle.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')