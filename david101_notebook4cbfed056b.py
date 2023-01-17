import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotting

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



# scikit-learn

from sklearn import preprocessing

from sklearn import svm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import model_selection

from sklearn.decomposition import PCA



import xgboost as xgb



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv('../input/train.csv')

df = pd.concat([pd.read_csv('../input/train.csv'), 

                pd.read_csv('../input/test.csv')], ignore_index=True)
mean_species = df_train.groupby('species').aggregate(np.mean)

for feature in ('margin', 'shape', 'texture'):

    fig = plt.figure(figsize=(13, 13))

    for spec in range(len(mean_species)):

        plt.subplot(15, 7, spec + 1)

        feature_map = np.zeros((64, 1))

        for i in range(1, 65): 

            feature_map[i-1] = mean_species.iloc[spec][feature + str(i)]

        feature_map = feature_map.reshape((8, 8))

        plt.imshow(feature_map, cmap='hot')

        plt.grid(False)

        plt.axis('off')

        plt.title(mean_species.index[spec], fontsize=6)

    fig.tight_layout()

    print(feature + ' maps')