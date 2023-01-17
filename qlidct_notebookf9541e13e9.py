# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

#train.columns
test = train[['YearBuilt','GarageYrBlt']]

test = test[test.GarageYrBlt != test.YearBuilt]

NoGarage = train[['Id','GarageArea','GarageYrBlt']]

nbNoGarage = NoGarage[NoGarage.GarageYrBlt.isnull()].shape

nbtest = test.shape

nbtest ##371

#nbNoGarage ##81

#Number_NoGarage
Garage_YrBlt_rem = train

Garage_YrBlt_id = Garage_YrBlt_rem[Garage_YrBlt_rem.GarageYrBlt.isnull()]['Id'].tolist()

Garage_YrBlt_rem['GarageYrBlt'] = Garage_YrBlt_rem[Garage_YrBlt_rem['Id'].isin(Garage_YrBlt_id)]['YearBuilt']

test = Garage_YrBlt_rem[Garage_YrBlt_rem.GarageYrBlt != Garage_YrBlt_rem.YearBuilt]

test.shape
MasVnr_NoValue = train[train['MasVnrType'].isnull() == True][['Id','MasVnrType','MasVnrArea','SalePrice']]

MasVnr_WithValue = train[train['MasVnrType'].isnull() == False][['Id','MasVnrType','MasVnrArea','SalePrice']]


#MasVnr_WithValue[MasVnr_WithValue.columns[2]].apply(lambda x: x.corr(MasVnr_WithValue['SalePrice']))

MasVnr_WithValue['MasVnrType'].corr(MasVnr_WithValue['SalePrice'], method='kendall')
test_total = train['SalePrice'].groupby(train['MasVnrType'])

plt.boxplot([test_total.apply(tolist)])

#test_total
test_NoValue['SalePrice'].describe()