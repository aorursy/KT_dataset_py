import pandas as pd
trainData = pd.read_csv('../input/train.csv')

y = np.array(trainData['SalePrice'])

trainData.drop(['Id', 'SalePrice'], axis=1, inplace=True)

testData = pd.read_csv('../input/test.csv')

testData.drop(['Id'], axis=1, inplace=True)

data = pd.concat([trainData, testData])

print(trainData.shape, testData.shape, data.shape)

data.head()