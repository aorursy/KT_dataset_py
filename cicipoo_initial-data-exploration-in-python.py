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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
train = pd.read_csv('../input/train.csv',index_col='Id')
test = pd.read_csv('../input/test.csv',index_col='Id')
test.head()
nan_cols=train.columns[train.isna().any()]
train[nan_cols].isna().sum().sort_values(ascending=False) / train.shape[0] * 100
sns.set(style="white", palette="muted", color_codes=True)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(7, 5))
sns.despine(left=True)

sns.distplot(train.SalePrice, kde=False, color="b", ax=ax)
plt.title('Distribution of Home Prices',fontdict=dict(size=18,weight='bold'))
plt.setp(ax, yticks=[])
plt.tight_layout();
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(7, 5))
sns.despine(left=True)

sns.distplot(np.log(train.SalePrice), kde=False, color="b", ax=ax)
plt.title('Log-Norm Distribution of Home Prices',fontdict=dict(size=18,weight='bold'))
plt.setp(ax, yticks=[])
plt.tight_layout();
# built-in formula:
print('Skew: '+str(stats.skew(train.SalePrice)))
print('Log-Norm Skew: '+str(stats.skew(np.log(train.SalePrice))))
# built-in kurtosis formula:
print('Kurtosis: '+str(stats.kurtosis(train.SalePrice)))
print('Log-Norm Kurtosis: '+str(stats.kurtosis(np.log(train.SalePrice))))
q1,q3 = train.SalePrice.describe()[['25%','75%']]
iqr = q3 - q1
max_right = q3 + 1.5*iqr
min_left = q1 - 1.5*iqr
print('Lower Bound: '+str(min_left))
print('Upper Bound: '+str(max_right))
outliers = train.loc[(train.SalePrice < min_left) | (train.SalePrice > max_right)] 
outliers.shape[0] / train.shape[0]
left_outliers = train.loc[(train.SalePrice < min_left)]
left_outliers.shape[0]
numeric_features = train[[train.columns[i] for i in range(len(train.columns)) if train.dtypes[i] in [int,float]]]
numeric_features.head()
sns.set(style="white")

# Compute the correlation matrix
corr = numeric_features.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 11))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr.loc['MiscVal'].loc['SalePrice']
# with hot-coding
train_cleaned = pd.get_dummies(train[[c for c in train if c != 'MiscVal']],dummy_na=False,drop_first=True)
train_cleaned_cols = train_cleaned.columns
# subset categorical dummy vars for analysis
dummy_features = train_cleaned[[c for c in train_cleaned.columns if '_' in c or c == 'SalePrice']]
dummy_features.head()
# Compute the correlation matrix
corr = dummy_features.corr()

sale_correlations = pd.DataFrame(corr.loc['SalePrice'][abs(corr.loc['SalePrice']) >.1])
sale_correlations['R_2'] = sale_correlations ** 2
sale_correlations.sort_values('R_2',ascending=False).head(10)
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
new_data = pd.DataFrame(my_imputer.fit_transform(train_cleaned))
new_data.columns = train_cleaned.columns
# This script comes from sklearn @ 
# https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=[0, 1])

# scaling dummy variables won't change them so we don't have to worry about them.
new_data = scaler.fit_transform(new_data[[c for c in new_data.columns if c != 'SalePrice']])
new_data = pd.DataFrame(new_data,columns=[c for c in train_cleaned_cols if c != 'SalePrice'])
new_data.head()
#Fitting the PCA algorithm with our Data
pca = PCA().fit(new_data)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Home Prices Dataset Explained Variance',fontsize=18)
plt.show();
explained_var = pd.DataFrame({'n_components':list(np.arange(new_data.shape[1])),
                              'cumsum':list(np.cumsum(pca.explained_variance_ratio_))})
explained_var.loc[explained_var['cumsum'] > .99].iloc[:5,:]
pca = PCA(n_components=156)
transformed_features = pca.fit_transform(new_data)
transformed_features[:5,:5]
transformed_data = pd.DataFrame(transformed_features)
transformed_data['SalePrice'] = list(np.log(train.SalePrice))
transformed_data.head()
transformed_data.to_csv('transformed_pca.csv',index=False)