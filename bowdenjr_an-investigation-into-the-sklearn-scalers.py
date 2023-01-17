import numpy as np

import pandas as pd

from sklearn import preprocessing

from matplotlib import pyplot as plt
def show_graphs(feature1,feature2):

    """

    Creates subplots of two Pandas series

    """

    fig = plt.figure(figsize=(12,8), dpi= 60, facecolor='w', edgecolor='k')



    for i,feature in enumerate([feature1,feature2]):

        ax = fig.add_subplot(2, 2, (2*(i+1))-1)

        ax.boxplot(feature,labels=[feature.name])

        ax.set_ylim(-5,20)

        ax = fig.add_subplot(2, 2, 2*(i+1))

        ax.hist(feature,50,facecolor='g')

        plt.xlabel(feature.name)



    fig.show()



    return 
from sklearn.datasets import load_wine



data = pd.DataFrame(load_wine()['data'],columns = load_wine()['feature_names'])

target = pd.DataFrame(load_wine()['target'])



data.describe()
show_graphs(data['alcohol'],data['color_intensity'])
alcohol_simple_scaled = pd.Series(preprocessing.scale(data['alcohol']),name='alcohol')

color_intensity_simple_scaled = pd.Series(preprocessing.scale(data['color_intensity']),name='color_intensity')



show_graphs(alcohol_simple_scaled,color_intensity_simple_scaled)
scaler = preprocessing.MinMaxScaler()



alcohol_minmax_scaled = pd.Series(scaler.fit_transform(data[['alcohol']]).ravel(),name='alcohol')

color_intensity_minmax_scaled = pd.Series(scaler.fit_transform(data[['color_intensity']]).ravel(),name='color_intensity')



show_graphs(alcohol_minmax_scaled,color_intensity_minmax_scaled)
scaler = preprocessing.MaxAbsScaler()



alcohol_MaxAbs_scaled = pd.Series(scaler.fit_transform(data[['alcohol']]).ravel(),name='alcohol')

color_intensity_MaxAbs_scaled = pd.Series(scaler.fit_transform(data[['color_intensity']]).ravel(),name='color_intensity')



show_graphs(alcohol_MaxAbs_scaled,color_intensity_MaxAbs_scaled)
scaler = preprocessing.RobustScaler()



alcohol_robust_scaled = pd.Series(scaler.fit_transform(data[['alcohol']]).ravel(),name='alcohol')

color_intensity_robust_scaled = pd.Series(scaler.fit_transform(data[['color_intensity']]).ravel(),name='color_intensity')



show_graphs(alcohol_robust_scaled,color_intensity_robust_scaled)
scaler = preprocessing.PowerTransformer(method = 'box-cox')



alcohol_power_scaled = pd.Series(scaler.fit_transform(data[['alcohol']]).ravel(),name='alcohol')

color_intensity_power_scaled = pd.Series(scaler.fit_transform(data[['color_intensity']]).ravel(),name='color_intensity')



show_graphs(alcohol_power_scaled,color_intensity_power_scaled)
scaler = preprocessing.QuantileTransformer(output_distribution = 'normal', n_quantiles=len(data))



alcohol_Quantile_scaled = pd.Series(scaler.fit_transform(data[['alcohol']]).ravel(),name='alcohol')

color_intensity_Quantile_scaled = pd.Series(scaler.fit_transform(data[['color_intensity']]).ravel(),name='color_intensity')



show_graphs(alcohol_Quantile_scaled,color_intensity_Quantile_scaled)
scaler = preprocessing.Normalizer()



alcohol_Normalizer_scaled = pd.Series(scaler.fit_transform(data[['alcohol']]).ravel(),name='alcohol')

color_intensity_Normalizer_scaled = pd.Series(scaler.fit_transform(data[['color_intensity']]).ravel(),name='color_intensity')



show_graphs(alcohol_Normalizer_scaled,color_intensity_Normalizer_scaled)
from mpl_toolkits.mplot3d import Axes3D



three_data = data[['alcohol','color_intensity','flavanoids']]

scaled_three_data = pd.DataFrame(scaler.fit_transform(three_data),columns=three_data.columns)



fig = plt.figure(figsize=(16,8), dpi= 60, facecolor='w', edgecolor='k')



ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(data['alcohol'],data['color_intensity'],data['flavanoids'])



ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.scatter(scaled_three_data['alcohol'],scaled_three_data['color_intensity'],scaled_three_data['flavanoids'])
fig = plt.figure(figsize=(12,8), dpi= 60, facecolor='w', edgecolor='k')



ax = plt.scatter(data['color_intensity'],data['flavanoids'])