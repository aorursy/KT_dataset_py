# This Python3 environment comes with many helpful analytics libraries installed

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
# traindf.dtypes
# reg_vars = traindf.columns[traindf.dtypes != 'objects']

# reg_vars

# pd.get_dummies(traindf.BldgType, prefix='BldgType').head()

# training_binarized_df = pd.get_dummies(traindf)

# training_binarized_df = training_binarized_df.set_index('Id')
# training_binarized_df.corr()['SalePrice'].sort_values(ascending=False)
# pd.get_dummies(traindf)
import seaborn as sns;

sns.set(color_codes=True)

sns.regplot('OverallQual', 'SalePrice', traindf)
# Select the benchmark models features

x_train = traindf[['YrSold','MoSold','LotArea','BedroomAbvGr','BedroomAbvGr']]

y_train = traindf['SalePrice']

x_test = testdf[['YrSold','MoSold','LotArea','BedroomAbvGr','BedroomAbvGr']]
from sklearn import linear_model
clf = linear_model.LinearRegression()

clf.fit(x_train,y_train)
from sklearn.cross_validation import cross_val_score

test_scores = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='mean_squared_error'))

test_scores
np.mean(test_scores)
x_test = x_test.fillna(x_test.mean())
yhat = clf.predict(x_test)
# yhat
# Create a dataframe with the row ID and price predicitons

yhatdf = pd.DataFrame(data={'Id':testdf.Id,'SalePrice':yhat})
# Write to CSV file

filename = 'benchmark.csv'

yhatdf.to_csv(filename,index=False)
resultdf = pd.read_csv('./benchmark.csv')

resultdf