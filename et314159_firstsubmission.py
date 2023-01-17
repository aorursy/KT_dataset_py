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
traindf= pd.read_csv("../input/train.csv")
testdf= pd.read_csv("../input/test.csv")
from sklearn import linear_model
x_train = traindf[['YrSold','MoSold','LotArea','BedroomAbvGr']]

y_train = traindf['SalePrice']

x_test = testdf[['YrSold','MoSold','LotArea','BedroomAbvGr']]
x_train.head(n=5)

x_train.shape
y_train.head(n=5)
from sklearn import linear_model
clf = linear_model.LinearRegression()

clf.fit(x_train,y_train)
yhat = clf.predict(x_test)
yhatdf = pd.DataFrame(data={'Id':testdf.Id, 'SalePrice': yhat})
yhatdf.head(n=5)
#filename = 'benchmark.csv'

#yhatdf.to_csv(filename,index=False)
(traindf.loc[:, traindf.dtypes != object]).info()

numVar_traindf=traindf.loc[:, traindf.dtypes != object]

numVar_traindf.shape
corrmat = numVar_traindf.corr()
import seaborn as sns

sns.heatmap(corrmat)
alldf = pd.concat((traindf.iloc[:, :-1], testdf))

alldf.shape
df_quality = pd.get_dummies(alldf['OverallQual'], prefix='OverallQual')

df_quality.head()
df_neighborhood = pd.get_dummies(alldf['Neighborhood'], prefix='Neighborhood')

df_neighborhood.head()
numVar_alldf = alldf.loc[:, alldf.dtypes != object]

numVar_traindf = pd.concat([numVar_alldf, df_quality,df_neighborhood], axis=1)

numVar_traindf = numVar_alldf[:1460]

numVar_traindf.shape
numVar_testdf = numVar_alldf[1460:]

numVar_testdf.shape
newTraindf  = numVar_traindf.fillna(numVar_traindf.mean())

newTraindf.isnull().sum().sort_values(ascending=False).head()
newTestdf  = numVar_testdf.fillna(numVar_traindf.mean())

newTestdf.isnull().sum().sort_values(ascending=False).head()
clf = linear_model.LinearRegression()

clf.fit(newTraindf,y_train)
yhat = clf.predict(newTestdf)
yhatdf = pd.DataFrame(data={'Id':testdf.Id, 'SalePrice': yhat})

yhatdf['SalePrice'] = yhatdf['SalePrice'].abs()

yhatdf.head()
yhatdf.isnull().sum().sort_values(ascending=False).head()
yhatdf.hist()
filename = 'benchmark.csv'

yhatdf.to_csv(filename,index=False)