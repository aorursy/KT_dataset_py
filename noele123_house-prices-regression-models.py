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
# Importing all library to use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
%matplotlib inline
from subprocess import check_output
print(check_output(["ls", "../input/house-prices-advanced-regression-techniques"]).decode("utf8")) #check the files available in the directory
# Reading data from file to proes the s
# file is at same location where we have this python code sheet
# Using panda lib to read file
trainData = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
trainData.shape
#correlation matrix for all Important Features
import seaborn as sns
corrmat = trainData.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,cmap="Blues", square=True);
## Getting the correlation of all the features with target variable. 
(trainData.corr()**2)["SalePrice"].sort_values(ascending = False)[1:]
# correlation matrix for 10 important features
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(trainData[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.1f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
def customized_scatterplot(y, x):
        ## Sizing the plot. 
    plt.style.use('fivethirtyeight')
    plt.subplots(figsize = (12,8))
    ## Plotting target variable with predictor variable(OverallQual)
    sns.scatterplot(y = y, x = x);
#More visulization
#Lets check with scatter plot

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(trainData[cols], height = 2.5)
plt.show();
#histogram and normal probability plot
from scipy.stats import norm
from scipy import stats
sns.distplot(trainData['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainData['SalePrice'], plot=plt)

# Lets check CoRelation

## Plot fig sizing. 
import matplotlib.style as style
import seaborn as sns

style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))
## Plotting heatmap. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(trainData.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(trainData.corr(), 
            cmap=sns.diverging_palette(20, 220, n=200), 
            mask = mask, 
            annot=True, 
            center = 0, 
           );
## Give title. 
plt.title("Heatmap of all the Features", fontsize = 30);