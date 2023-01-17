# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



plt.style.use('fivethirtyeight')
# Import pandas

#import pandas as pd

#import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize=(15,7))

# Read 'gapminder.csv' into a DataFrame: df

df = pd.read_csv('../input/gm2008/gm_2008_region.csv')



# Print the columns of df

print(df.columns)



# Create a boxplot of life expectancy per region

df.boxplot('life', 'Region', rot=60,ax=ax)



# Show the plot

plt.show()

# Print the columns of df_region

print("Dataframe with Region:\n ",df.info())



print("Dataframe head containing Region column :\n", df['Region'].head(10))



# Create dummy variables: df_region

df_region1 = pd.get_dummies(df)



print("Dataframe after creating dummy columns without dropping region :\n ",df_region1.info())



# Create dummy variables with drop_first=True: df_region

df_region2 = pd.get_dummies(df,drop_first=True)



# Print the new columns of df_region

print("Dataframe Region with Dummy Columns but dropping first dummy column : \n",df_region2.info())



print("Dataframe Region columns after dummification step :\n ", df_region2.iloc[:10,9:11])
from sklearn.datasets import load_boston

boston = load_boston()

X = boston.data

Y = boston.target

print("X Shape : ",X.shape)

print("Y Shape : ",Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1000)
import numpy as np

X = np.random.uniform(0.0, 1.0, size=(10, 2))

Y = np.random.choice(('Male','Female'), size=(10))

print("X : ",X)

print("Y : ",Y)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

yt = le.fit_transform(Y)

print(yt)
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

Yb = lb.fit_transform(Y)

print("Yb : ",Yb)

print("Inverse Transformation : ",lb.inverse_transform(Yb))
from sklearn.preprocessing import Imputer

data = np.array([[1, np.nan, 2], [2, 3, np.nan], [-1, 4, 2]])

print("data : ",data)

# Mean Strategy

imp = Imputer(strategy='mean')

trans_data = imp.fit_transform(data)

print("Transformed Data using mean strategy : \n",trans_data)

# Median Strategy

imp = Imputer(strategy='median')

trans_data = imp.fit_transform(data)

print("Transformed Data using median strategy : \n",trans_data)

# Most Frequent

imp = Imputer(strategy='most_frequent')

trans_data = imp.fit_transform(data)

print("Transformed Data using most frequent strategy : \n",trans_data)
# Import pandas

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



# Read 'gapminder.csv' into a DataFrame: df

df = pd.read_csv('../input/housingvotes/house-votes-84.csv',header=None)



# Convert '?' to NaN

df[df == '?'] = np.nan



# Print the number of NaNs

print("The number of NaNs :\n",df.isnull().sum())



# Print shape of original DataFrame

print("Shape of Original DataFrame: {}".format(df.shape))



# Drop missing values and print shape of new DataFrame

df = df.dropna()



# Print shape of new DataFrame

print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

from pandas import read_csv

from sklearn.preprocessing import Imputer

#from sklearn.impute import SimpleImputer

import numpy as np

dataset = read_csv('../input/pimaindian/pima-indians-diabetes.data.csv', header=None)

# mark zero values as missing or NaN

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, np.NaN)

# fill missing values with mean column values

values = dataset.values



imputer = Imputer()



transformed_values = imputer.fit_transform(values)



# count the number of NaN values in each column

print("NaN values count :- ",np.isnan(transformed_values).sum())



from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_wine



ss = StandardScaler()

features, target = load_wine(return_X_y=True)



scaled_data = ss.fit_transform(features)

print('Unscaled Data:\n',features)

print("Scaled Data :\n",scaled_data)
from sklearn.preprocessing import Normalizer

import numpy as np



data = np.array([1.0, 2.0])

n_max = Normalizer(norm='max')

norm_data = n_max.fit_transform(data.reshape(1, -1))

print("Norm Data(max) :\n ",norm_data)

n_l1 = Normalizer(norm='l1')

norm_data = n_l1.fit_transform(data.reshape(1, -1))

print("Norm Data(l1) :\n ",norm_data)

n_l2 = Normalizer(norm='l2')

n_l2.fit_transform(data.reshape(1, -1))

print("Norm Data(l2) :\n ",norm_data)
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.DataFrame({

    # positive skew

    'x1': np.random.chisquare(8, 1000),

    # negative skew 

    'x2': np.random.beta(8, 2, 1000) * 40,

    # no skew

    'x3': np.random.normal(50, 3, 1000)

})



scaler = MinMaxScaler()

scaled_df = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

ax1.set_title('Before Scaling')

sns.kdeplot(df['x1'], ax=ax1)

sns.kdeplot(df['x2'], ax=ax1)

sns.kdeplot(df['x3'], ax=ax1)

ax2.set_title('After Min-Max Scaling')

sns.kdeplot(scaled_df['x1'], ax=ax2)

sns.kdeplot(scaled_df['x2'], ax=ax2)

sns.kdeplot(scaled_df['x3'], ax=ax2)

plt.show()
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



x = pd.DataFrame({

    # Distribution with lower outliers

    'x1': np.concatenate([np.random.normal(20, 1, 1000), np.random.normal(1, 1, 25)]),

    # Distribution with higher outliers

    'x2': np.concatenate([np.random.normal(30, 1, 1000), np.random.normal(50, 1, 25)]),

})



scaler = RobustScaler()

robust_scaled_df = scaler.fit_transform(x)

robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])



scaler = MinMaxScaler()

minmax_scaled_df = scaler.fit_transform(x)

minmax_scaled_df = pd.DataFrame(minmax_scaled_df, columns=['x1', 'x2'])



fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 5))



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

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import Normalizer

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.DataFrame({

    'x1': np.random.randint(-100, 100, 1000).astype(float),

    'y1': np.random.randint(-80, 80, 1000).astype(float),

    'z1': np.random.randint(-150, 150, 1000).astype(float),

})



scaler = Normalizer()

scaled_df = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_df, columns=df.columns)



fig = plt.figure(figsize=(9, 5))

ax1 = fig.add_subplot(121, projection='3d')

ax2 = fig.add_subplot(122, projection='3d')

ax1.scatter(df['x1'], df['y1'], df['z1'])

ax2.scatter(scaled_df['x1'], scaled_df['y1'], scaled_df['z1'])

plt.show()
#SelectKBest features

from sklearn.datasets import load_boston, load_iris

from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_regression

regr_data = load_boston()

print(regr_data.data.shape)

kb_regr = SelectKBest(f_regression)

X_b = kb_regr.fit_transform(regr_data.data, regr_data.target)

print(X_b.shape)

print(kb_regr.scores_)
class_data = load_iris()

print(class_data.data.shape)

perc_class = SelectPercentile(chi2, percentile=15)

X_p = perc_class.fit_transform(class_data.data, class_data.target)

print(X_p.shape)

print(perc_class.scores_)
from sklearn.datasets import load_digits

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



digits = load_digits()



# Show some random digits

selection = np.random.randint(0, 1797, size=100)



fig, ax = plt.subplots(10, 10, figsize=(10, 10))



samples = [digits.data[x].reshape((8, 8)) for x in selection]



for i in range(10):

    for j in range(10):

        ax[i, j].set_axis_off()

        ax[i, j].imshow(samples[(i * 8) + j], cmap='gray')



plt.show()





pca = PCA(n_components=36, whiten=True)

X_pca = pca.fit_transform(digits.data / 255)

print(pca.explained_variance_ratio_)



# Plot the explained variance ratio

fig, ax = plt.subplots(1, 2, figsize=(16, 6))



ax[0].set_xlabel('Component')

ax[0].set_ylabel('Variance ratio (%)')

ax[0].bar(np.arange(36), pca.explained_variance_ratio_ * 100.0)



ax[1].set_xlabel('Component')

ax[1].set_ylabel('Cumulative variance (%)')

ax[1].bar(np.arange(36), np.cumsum(pca.explained_variance_)[::-1])



plt.show()
X_rebuilt = pca.inverse_transform(X_pca)

# Rebuild from PCA and show the result

fig, ax = plt.subplots(10, 10, figsize=(10, 10))



samples = [pca.inverse_transform(X_pca[x]).reshape((8, 8)) for x in selection]



for i in range(10):

    for j in range(10):

        ax[i, j].set_axis_off()

        ax[i, j].imshow(samples[(i * 8) + j], cmap='gray')



plt.show()
