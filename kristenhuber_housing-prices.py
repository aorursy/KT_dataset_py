# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import statsmodels.api as sm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dftrain=pd.DataFrame(pd.read_table("../input/train.csv",delimiter=',',encoding="utf-8-sig"))

# I ran into some problems with encoding on my pc with this data set, so I'll play it safe

dftest=pd.DataFrame(pd.read_table("../input/test.csv",delimiter=',',encoding="utf-8-sig"))

print(dftrain.columns)

print(dftest.columns)
#def clean(dataframe):

#    for column in dataframe:

#        if dataframe.dtypes(column[1])=='string':

#            pd.get_dummies(dataframe, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)

dftrain.select_dtypes(include=['bool','object']).apply(pd.get_dummies(dftrain,prefix=None,prefix_sep='_',dummy_na=False, columns=None,sparse=False,drop_first=False))



#df.select_dtypes(include=['float64']).apply(your_function)

#df.select_dtypes(exclude=['string','object']).apply(your_other_function)



print(dftrain.head)
dftrain.head()
dftrain.iloc[:,10:20]
neighborhood = pd.get_dummies(dftrain['Neighborhood'], columns=True)
neighborhood
#

price=dftrain["SalePrice"]

year=dftrain["YearBuilt"]



y=np.matrix(price).transpose()

#x1=np.matrix(year).transpose()

x=np.matrix(neighborhood).transpose()



#x=np.column_stack([x1,x2])



X=sm.add_constant(x)

model=sm.OLS(y,X)
dftrain.shape
neighborhood.shape
dftrain = pd.get_dummies(dftrain["Neighborhood"])
newdf=pd.concat(dftrain, neighborhood,left_index=True, right_index=True)



newdf.columns