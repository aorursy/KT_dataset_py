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
from xgboost.sklearn import XGBRegressor
df1 = pd.read_csv('../input/train.csv')



df2 = pd.read_csv('../input/test.csv')



frame = df1.append(df2)



frame = pd.get_dummies(frame)
mycolumns = frame.columns.tolist()

mycolumns.remove('SalePrice')

for column in mycolumns:

    frame[column] = frame[column].fillna(frame[column].median())
clf = XGBRegressor(n_estimators=1000,max_depth=4)
train = frame[frame.SalePrice.notnull()].dropna()
test = frame[frame.SalePrice.isnull()]
frame.SalePrice
clf = clf.fit(train[mycolumns],train['SalePrice'])
preds = clf.predict(test[mycolumns],)
test['SalePrice'] = preds
test['SalePrice']
preds
test[['Id','SalePrice']].to_csv('upload.csv',index=False)
train