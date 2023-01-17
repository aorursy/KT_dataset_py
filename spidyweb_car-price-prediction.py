# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import all libraries and dependencies for dataframe



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as ticker

import matplotlib.ticker as plticker



# import all libraries and dependencies for machine learning

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.base import TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '../input/car-price-prediction/'

data= pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')

data1= pd.read_excel('../input/car-price-prediction/Data Dictionary - carprices.xlsx')

data.head()
data1.head()
data.describe()
data.isnull().sum()
len(data)
data.shape
data.info()
# dropping car_ID based on business knowledge



data = data.drop('car_ID',axis=1)
# Outlier Analysis of target variable with maximum amount of Inconsistency



outliers = ['price']

plt.rcParams['figure.figsize'] = [8,8]

sns.boxplot(data=data[outliers], orient="v", palette="Set1" ,whis=1.5,saturation=1, width=0.7)

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Price Range", fontweight = 'bold')

plt.xlabel("Continuous Variable", fontweight = 'bold')

data.shape
data.CarName.unique()


data['CarName'] = data['CarName'].str.split(' ',expand=True)
data.CarName.unique()
# Renaming the typo errors in Car Company names

data['CarName'] = data['CarName'].replace({'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 

                            'vokswagen': 'volkswagen', 'vw': 'volkswagen'})
data.CarName.unique()