# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mushroom = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

mushroom.shape
mushroom.head()
list(mushroom)

mushroom.dtypes
mushroom.info()
data = mushroom[mushroom['stalk-root'] != '?'] 

print(data)
data.shape


features = [

 'cap-shape',

 'cap-surface',

 'cap-color',

 'bruises',

 'odor',

 'gill-attachment',

 'gill-spacing',

 'gill-size',

 'gill-color',

 'stalk-shape',

 'stalk-root',

 'stalk-surface-above-ring',

 'stalk-surface-below-ring',

 'stalk-color-above-ring',

 'stalk-color-below-ring',

 'veil-type',

 'veil-color',

 'ring-number',

 'ring-type',

 'spore-print-color',

 'population',

 'habitat']



x = mushroom.loc[:, features]



y = mushroom.loc[:,['class'] ]
y['class'].value_counts()
X_enc = pd.get_dummies(x) 

X_enc.head() 

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

y_enc = le.fit_transform(y.values.ravel()) 

yDf = pd.DataFrame(y_enc) 

yDf.rename(columns={0: "class"}, inplace = True) 

yDf.head()


scaler = StandardScaler()

x_std=scaler.fit_transform(X_enc)





pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x_std)

principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1',

'pc2'])



principalDf.head()
import seaborn as sns

sns_plot = sns.scatterplot(principalDf.pc1, principalDf.pc2)

fig = sns_plot.get_figure()

fig.savefig('output.jpg')
