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
pulsar_data = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')   

exoplanet_data = pd.read_csv('../input/kepler-exoplanet-search-results/cumulative.csv')
pd.options.display.max_columns = None

pulsar_data.describe()


exoplanet_data.dtypes
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

import missingno as msno
msno.matrix(exoplanet_data)
missing_exoplanet_data = exoplanet_data.columns[exoplanet_data.isnull().any()].tolist()

msno.bar(exoplanet_data[missing_exoplanet_data], color="red", log=True, figsize=(30,18))
msno.heatmap(exoplanet_data[missing_exoplanet_data], figsize=(20,20))
plt.figure(figsize=(10, 7))

sns.heatmap(pulsar_data.corr(), annot=True, cmap=sns.color_palette('colorblind', 15))

plt.show
exoplanet_data['koi_depth'].plot(kind='hist')

plt.xscale('log',basex=10) 

plt.show()

exoplanet_data['koi_depth'].plot(kind='density')

plt.xscale('log',basex=10) 

plt.show()
pulsar_data["Mean of the integrated profile"]=pulsar_data.index

sns.boxplot(x="target_class",y="Mean of the integrated profile",data=pulsar_data)
sns.scatterplot(x="target_class",y="Mean of the integrated profile",data=pulsar_data)
sns.scatterplot(x="koi_depth",y="koi_duration",data=exoplanet_data)
sns.pairplot(pulsar_data, hue = 'target_class')
sns.jointplot(x='koi_depth', y='koi_duration', data=exoplanet_data, kind='kde')
#Let's try to create a PCA model with 5 components from pulsar data



from sklearn import decomposition



pca = decomposition.PCA(n_components=5)



#remove target class

pulsar_data_pca= pulsar_data.loc[:, pulsar_data.columns != 'target_class']



#normalize data

pulsar_data_pca = pulsar_data_pca*1./np.max(pulsar_data_pca, axis=0)



pc = pca.fit_transform(pulsar_data_pca)



#create a new data frame with principal components

pc_df = pd.DataFrame(data = pc , 

        columns = ['PC1', 'PC2','PC3','PC4','PC5'])



#Assign target class value as Cluster

pc_df['Cluster'] = pulsar_data['target_class']



pc_df.head(5)





pca.explained_variance_ratio_




df = pd.DataFrame({'Variance':pca.explained_variance_ratio_,

             'Principal Component':['PC1', 'PC2','PC3','PC4','PC5']})

sns.barplot(x='Principal Component',y="Variance", 

           data=df, color="r");
