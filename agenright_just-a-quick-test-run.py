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

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor



%matplotlib inline



trainData = pd.read_table('../input/train.tsv')

testData = pd.read_table('../input/test.tsv')
trainData.head()



trainData['brand_name'].fillna('missing',inplace=True)

testData['brand_name'].fillna('missing',inplace=True)



trainData[['category1_label', 'category2_label', 'category3_label', 'category4_label', 'category5_label']] = trainData['category_name'].str.split('/', expand=True)

testData[['category1_label', 'category2_label', 'category3_label', 'category4_label', 'category5_label']] = testData['category_name'].str.split('/', expand=True)



trainData['category1_label'].fillna('missing',inplace=True)

testData['category1_label'].fillna('missing',inplace=True)

trainData['category2_label'].fillna('missing',inplace=True)

testData['category2_label'].fillna('missing',inplace=True)

trainData['category3_label'].fillna('missing',inplace=True)

testData['category3_label'].fillna('missing',inplace=True)





le = LabelEncoder()



trainData['brand'] = le.fit_transform(trainData['brand_name'])

testData['brand'] = le.fit_transform(testData['brand_name'])

trainData['category1'] = le.fit_transform(trainData['category1_label'])

testData['category1'] = le.fit_transform(testData['category1_label'])

trainData['category2'] = le.fit_transform(trainData['category2_label'])

testData['category2'] = le.fit_transform(testData['category2_label'])

trainData['category3'] = le.fit_transform(trainData['category3_label'])

testData['category3'] = le.fit_transform(testData['category3_label'])

lr = RandomForestRegressor()



predictors = ['item_condition_id', 'shipping', 'brand', 'category1', 'category2', 'category3']

target = 'price'

lr.fit(trainData[predictors], trainData[target])

testData[target] = lr.predict(testData[predictors])

testData.head()
testData.to_csv('./mysubmission.csv', columns=['test_id','price'],index=False)