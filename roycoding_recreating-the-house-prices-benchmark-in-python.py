# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load the training and test data sets

traindf = pd.read_csv('../input/train.csv')

testdf = pd.read_csv('../input/test.csv')
# Select the benchmark models features

X_train = traindf[['YrSold','MoSold','LotArea','BedroomAbvGr']]

y_train = traindf['SalePrice']

X_test = testdf[['YrSold','MoSold','LotArea','BedroomAbvGr']]
from sklearn import linear_model
clf = linear_model.LinearRegression()

clf.fit(X_train,y_train)
yhat = clf.predict(X_test)
# Create a dataframe with the row ID and price predictions

yhatdf = pd.DataFrame(data={'Id':testdf.Id, 'SalePrice': yhat})
# Write to CSV file

filename = 'benchmark.csv'

yhatdf.to_csv(filename,index=False)