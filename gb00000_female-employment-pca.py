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


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor

import matplotlib.pyplot as plt

#import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from scipy.spatial.distance import euclidean

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor



from sklearn.preprocessing import  MinMaxScaler

mlr2 = pd.read_csv('/kaggle/input/female-employment-vs-socioeconimic-factors/MLR2.csv')
mlr2
fertility_rate_train_y =  mlr2['FertilityRate'].dropna()
fertility_rate_train_y
fertility_rate_train_X = mlr2.iloc[0:23].drop('FertilityRate',axis=1)

fertility_rate_test_X = mlr2.iloc[23:].drop('FertilityRate',axis=1)
fertility_rate_train_X 
fertility_rate_train_X_norm =  pd.DataFrame(MinMaxScaler().fit_transform( fertility_rate_train_X ), columns = fertility_rate_train_X.columns)

fertility_rate_test_X_norm =  pd.DataFrame(MinMaxScaler().fit_transform( fertility_rate_test_X ), columns = fertility_rate_test_X.columns)
fertility_rate_train_X_norm 
#Use models to fill in the missing fertility rate values with their average, since there isn't a lot of data

def reg_model(mod):

    model = mod()

    model.fit(fertility_rate_train_X_norm.values,fertility_rate_train_y.values)

    return model.predict(fertility_rate_test_X_norm)
models = [RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,KNeighborsRegressor]
preds = []



for i in models:

    print(i)

    print(reg_model(i))

    preds.append(reg_model(i))



# try one model to see which was the bigger predictors of fertility rate

model = GradientBoostingRegressor()

model.fit(fertility_rate_train_X_norm.values,fertility_rate_train_y.values)

model.predict(fertility_rate_test_X_norm)

pd.Series(model.feature_importances_, fertility_rate_train_X_norm.columns).sort_values(ascending=False)
#take the average of the model predictions to fill values

fr_avg23 = np.mean([i[0] for i in preds ]).round(2)

fr_avg24 = np.mean([i[1] for i in preds ]).round(2)
fr_avg23
mlr2_full = mlr2.copy()

mlr2_full['FertilityRate'].iloc[23] = fr_avg23

mlr2_full['FertilityRate'].iloc[24] = fr_avg24
mlr2_full
ss =  StandardScaler()

mlr2_ss = pd.DataFrame(ss.fit_transform(mlr2_full), columns = mlr2_full.columns)
mlr2_ss
pca = PCA(n_components=5)

mlr2_pca = pca.fit_transform(mlr2_ss)
pca.explained_variance_
plt.figure()

x = np.linspace(0,len(pca.explained_variance_ratio_  ), num = len(pca.explained_variance_ratio_  ))

plt.plot(x, pca.explained_variance_ratio_, 'r.')

plt.xlabel('components')

plt.ylabel('explained variance ratio')

plt.show()
pca2 = PCA(n_components=2)

mlr2_pca2 = pca2.fit_transform(mlr2_ss)

#mlr2_pca2
plt.figure()

plt.scatter(mlr2_pca2[:,0],mlr2_pca2[:,1])

plt.xlabel('first principal component')

plt.ylabel('second Principal Component')

plt.show()