import pandas as pd

from IPython.display import Image

import numpy as np

from sklearn import preprocessing

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

matplotlib.style.use('ggplot')
np.random.seed(1)

df = pd.DataFrame({

    'x1': np.random.normal(0, 2, 10000),

    'x2': np.random.normal(5, 3, 10000),

    'x3': np.random.normal(-5, 5, 10000)

})



scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))



ax1.set_title('Before Scaling')

sns.kdeplot(df['x1'], ax=ax1)

sns.kdeplot(df['x2'], ax=ax1)

sns.kdeplot(df['x3'], ax=ax1)

ax2.set_title('After Standard Scaler')

sns.kdeplot(scaled_df['x1'], ax=ax2)

sns.kdeplot(scaled_df['x2'], ax=ax2)

sns.kdeplot(scaled_df['x3'], ax=ax2)

plt.show()
df = pd.DataFrame({

    # positive skew

    'x1': np.random.chisquare(8, 1000),

    # negative skew 

    'x2': np.random.beta(8, 2, 1000) * 40,

    # no skew

    'x3': np.random.normal(50, 3, 1000)

})



scaler = preprocessing.MinMaxScaler()

scaled_df = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')

sns.kdeplot(df['x1'], ax=ax1)

sns.kdeplot(df['x2'], ax=ax1)

sns.kdeplot(df['x3'], ax=ax1)

ax2.set_title('After Min-Max Scaling')

sns.kdeplot(scaled_df['x1'], ax=ax2)

sns.kdeplot(scaled_df['x2'], ax=ax2)

sns.kdeplot(scaled_df['x3'], ax=ax2)

plt.show()
x = pd.DataFrame({

    # Distribution with lower outliers

    'x1': np.concatenate([np.random.normal(20, 1, 1000), np.random.normal(1, 1, 25)]),

    # Distribution with higher outliers

    'x2': np.concatenate([np.random.normal(30, 1, 1000), np.random.normal(50, 1, 25)]),

})



scaler = preprocessing.RobustScaler()

robust_scaled_df = scaler.fit_transform(x)

robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])



scaler = preprocessing.MinMaxScaler()

minmax_scaled_df = scaler.fit_transform(x)

minmax_scaled_df = pd.DataFrame(minmax_scaled_df, columns=['x1', 'x2'])



fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 5))

ax1.set_title('Before Scaling')

sns.kdeplot(x['x1'], ax=ax1)

sns.kdeplot(x['x2'], ax=ax1)

ax2.set_title('After Robust Scaling')

sns.kdeplot(robust_scaled_df['x1'], ax=ax2)

sns.kdeplot(robust_scaled_df['x2'], ax=ax2)

ax3.set_title('After Min-Max Scaling')

sns.kdeplot(minmax_scaled_df['x1'], ax=ax3)

sns.kdeplot(minmax_scaled_df['x2'], ax=ax3)

plt.show()
from mpl_toolkits.mplot3d import Axes3D



df = pd.DataFrame({

    'x1': np.random.randint(-100, 100, 1000).astype(float),

    'y1': np.random.randint(-80, 80, 1000).astype(float),

    'z1': np.random.randint(-150, 150, 1000).astype(float),

})



scaler = preprocessing.Normalizer()

scaled_df = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_df, columns=df.columns)



fig = plt.figure(figsize=(9, 5))

ax1 = fig.add_subplot(121, projection='3d')

ax2 = fig.add_subplot(122, projection='3d')

ax1.scatter(df['x1'], df['y1'], df['z1'])

ax2.scatter(scaled_df['x1'], scaled_df['y1'], scaled_df['z1'])

plt.show()