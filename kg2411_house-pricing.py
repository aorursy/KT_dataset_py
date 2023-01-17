
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import xgboost as xgb  #GBM algorithm
from xgboost import XGBRegressor
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from IPython.display import display

# remove warnings
import warnings
warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', na_values=['?',''],delimiter=',',delim_whitespace=False)
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', na_values=['?',''],delimiter=',',delim_whitespace=False)


# Correlation Matrix
data_aux = train_data
# Correlation Matrix all features
correlation_matrice = data_aux.corr()
f, ax = plt.subplots( figsize=(15, 12))
sns.heatmap(correlation_matrice,vmin=0.2, vmax=0.8, square= True, cmap= 'BuPu')
plt.xlabel('The house features in the x axis',fontsize= 13)
plt.ylabel('The house features in the y axis',fontsize= 13)
plt.title('Figure 1 - The correlation matrix between all the featues ', fontsize= 16);

#Scatter plot of thr most important features
cols = ['SalePrice', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'GrLivArea', 'GarageCars']
sns.pairplot(data_aux[cols], size = 2.8)
plt.suptitle('Figure 2 - The scatter plot of the top features ',x=0.5, y=1.01, verticalalignment='top', fontsize= 18)
plt.tight_layout()
plt.show();

# regplot of GrLivArea/SalePrice

ax = sns.regplot(x=data_aux['GrLivArea'], y=data_aux['SalePrice'])
plt.ylabel('SalePrice', fontsize= 10)
plt.xlabel('GrLivArea', fontsize= 10)
plt.title('Figure 3 - regplot of the GrLivArea with the SalePrice', fontsize= 12)
plt.show();
# Removing the outliers
# We sort the values by GrLivArea and select the two lager values, and we locate the index number 
# to use it in order to drop corresponding rows.
g_out = data_aux.sort_values(by="GrLivArea", ascending = False).head(2)
g_out
data_aux.drop([523,1298], inplace = True)
data_aux.reset_index(inplace=True)

