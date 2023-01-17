# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')

df.head()
# method : {‘pearson’, ‘kendall’, ‘spearman’} or callable

corr = df.corr(method='pearson')

corr.head()
import seaborn as sns

sns.heatmap(corr) # this will give you a basic heat map
labels = {

    'age': 'Age (in years)', # Can be made categorical,

    'sex': 'Sex', # Categorical

    'cp': 'Chest Pain Type', # Categorical

    'trestbps': 'Resting blood pressure',

    'chol': 'Serum Cholestoral',

    'fbs': 'Fasting Blood Sugar',

    'restecg': 'Resting ECG',

    'thalach': 'Max Heart Rate Achieved',

    'exang': 'Exercise Induced Angina', # Categorical,

    'oldpeak': 'ST depression',

    'slope': 'Slope of peak exercise ST Seg',

    'ca': 'No. of major vessels colored by flourospy', # Range (0-3)

    'thal': 'Thal', # Categorical,

    'target': 'Target'

}



corr = corr.rename(labels)



# This is to remove the top right triange - because that's just duplicate information

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Colors

cmap = sns.diverging_palette(240, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0)
from sklearn.decomposition import PCA



component_var = {}

for i in range(2, 6):

    pca = PCA(n_components=i)

    res = pca.fit(df)

#     print('At components: ', i)

#     display(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

    component_var[i] = sum(pca.explained_variance_ratio_)

print(component_var)
