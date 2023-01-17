# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import numpy as np

import time

from glob import glob

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



patient_path = glob('/kaggle/input/coronavirusdataset/patient.csv')[0]

patient = pd.read_csv(patient_path)



patient.head()
patient['age'] = 2020 - patient['birth_year']

patient['sex'] = patient['sex'].map({'female' : 0, 'male' : 1})
str_cols = ['country', 'region', 'group', 'infection_reason']

num_cols = ['sex', 'disease', 'infection_order', 'infected_by', 'contact_number', 'age']

label = 'state'
num_dset = patient[num_cols].fillna(0)

str_dset = pd.get_dummies(patient[str_cols])

all_df = pd.concat([num_dset, str_dset], axis=1)

all_df[label] = patient[label]

all_df.head()
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

import seaborn as sns



rdi_df = all_df.copy()

rdi_df.fillna(0)

scaler = MinMaxScaler()



rdi_X = rdi_df[list(rdi_df.columns)[:-1]]

rdi_y = rdi_df['state']

scaled_X = pd.DataFrame(scaler.fit_transform(rdi_X), columns=rdi_X.columns)
pca = PCA(n_components=2)



y_numeric = rdi_y.map({"isolated": 0, "released" : 1, 'deceased' : 2})

PCA_data = scaled_X.copy()

PCA_data['state'] = list(y_numeric)

principalComponents = pca.fit_transform(PCA_data)



principalDf = pd.DataFrame(principalComponents

             , columns = ['principal component 1', 'principal component 2'])



labeled_Df = pd.concat([principalDf, rdi_y], axis=1)



ax = sns.scatterplot(x = 'principal component 1', 

                     y = 'principal component 2', 

                     hue = 'state',

                     data = labeled_Df, 

                     palette ='Spectral')
PC1_feature_importance = sorted(zip(map(

                                        lambda x : int(x * 1000) / 1000, 

                                        pca.components_[0]), 

                                    PCA_data.columns), 

                                reverse=True)



PC2_feature_importance = sorted(zip(map(

                                        lambda x : int(x * 1000) / 1000, 

                                        pca.components_[1]), 

                                    PCA_data.columns), 

                                reverse=True)
all_set = set()

for n, z in enumerate(zip(PC1_feature_importance, PC2_feature_importance)):

    all_set.add(z[0][1])

    all_set.add(z[1][1])

    if n == 2: break

print(all_set)
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(n_components=2)



y_numeric = rdi_y.map({"isolated": 0, "released" : 1, 'deceased' : 2})

PLS_data = scaled_X.copy()

pls.fit(PLS_data, y_numeric)



x_scores = pd.DataFrame(pls.x_scores_, columns=['x_scores_PC1', 'x_scores_PC2'])

y_scores = pd.DataFrame(pls.y_scores_, columns=['y_scores_PC1', 'y_scores_PC2'])

xy_scores_l = pd.concat([x_scores, y_scores, y_numeric], axis=1)



x_loadings = pd.DataFrame(pls.x_loadings_, columns=['x_loadings_PC1', 'x_loadings_PC2'])

x_weights = pd.DataFrame(pls.x_weights_, columns=['x_weights_PC1', 'x_weights_PC2'])



x_loading_weight = pd.concat([x_loadings, x_weights], axis=1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.scatterplot(x = 'x_scores_PC1', y = 'x_scores_PC2',

                data = xy_scores_l, hue = 'state', palette ='Spectral',

                ax=axes[0])



sns.scatterplot(x = 'x_scores_PC1', y = 'y_scores_PC1',

                data = xy_scores_l, hue = 'state', palette ='Spectral',

                ax=axes[1])



sns.scatterplot(x = 'y_scores_PC1', y = 'y_scores_PC2',

                data = xy_scores_l, hue = 'state', palette ='Spectral',

                ax=axes[2])



plt.show()



fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.scatterplot(x = 'x_loadings_PC1', y = 'x_loadings_PC2',

                data = x_loading_weight, palette ='Spectral',

                ax=axes[0])



sns.scatterplot(x = 'x_weights_PC1', y = 'x_weights_PC2',

                data = x_loading_weight, palette ='Spectral',

                ax=axes[1])



sns.scatterplot(x = 'x_loadings_PC1', y = 'x_weights_PC1',

                data = x_loading_weight, palette ='Spectral',

                ax=axes[2])

plt.show()
importance_sumup = pd.concat([x_loading_weight['x_weights_PC1'], 

                              x_loading_weight['x_weights_PC2']])



feature_importance = sorted(zip(map(

                                    lambda x : int(x * 1000) / 1000, 

                                    importance_sumup), 

                                PCA_data.columns), 

                            reverse=True)

feature_importance[:6]
pearson_df = scaled_X.copy()

desc = pearson_df.describe()



std_s = [desc[l][2] for l in desc.columns]

sorted_list = sorted(zip(std_s, desc.columns), reverse=True)

good_col_set = sorted_list[:len(sorted_list) // 3]

good_cols = [ll[1] for ll in good_col_set]



sel_pearson = pearson_df[good_cols]

sel_pearson['state'] = list(y_numeric)

sel_pearson = sel_pearson.corr()
fig = plt.figure(figsize=(10,5))

sns.heatmap(sel_pearson.sort_values(by='state', ascending=True))

plt.show()