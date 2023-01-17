# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

import seaborn as sns

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.ensemble import GradientBoostingRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv'); 

data.SalePrice = np.log10(data.SalePrice)

data.head()
# data = data.replace(np.nan, 0)
data = data.append( data , ignore_index = True )

data.describe()
data.head()
uniques = data.Street.unique()

uniques
Street = pd.get_dummies( data.Street , prefix='Street' )

Street.head()
data.boxplot('SalePrice','Street',rot = 30,figsize=(5,6))
# converting street to numberival values

StreetsList = {

    'street':{

        'Pave':1,

        'Grvl':2

    }

}

print(StreetsList)
convertedStreet = pd.Series( np.where( data.Street == 'Pave' , 1 , 0 ) , name = 'Streetno' )

data = pd.concat( [ convertedStreet,data ] , axis=1 )
# data = data.drop(columns=['OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','Street','MSSubClass','MSZoning','LotFrontage','LotArea','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','HouseStyle'])

# col_list = ['Streetno','SalePrice']

# data = data[col_list]

data.head().T
data.plot('SalePrice','Streetno')
def get_some_data():

    y = data.SalePrice

    X = data[['Streetno','SalePrice']]

    my_imputer = SimpleImputer()

    imputed_X = my_imputer.fit_transform(X)

    return imputed_X, y
X, y = get_some_data()

my_model = GradientBoostingRegressor()

my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model,       

                                   features=[0, 1], # column numbers of plots we want to show

                                   X=X,            # raw predictors data.

                                   feature_names=['Streetno', 'SalePrice'], # labels on graphs

                                   grid_resolution=10) 