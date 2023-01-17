import pandas as pd
trainData = pd.read_csv('../input/train.csv')
trainData.columns
trainData.head()
trainData.isnull().any()
trainData.isnull().sum()
missingDataPer = (trainData.isnull().sum()/trainData.shape[0])*100
print(missingDataPer)
trainData = trainData[trainData.Embarked.notnull()]

missingDataPer = (trainData.isnull().sum()/trainData.shape[0])*100
print(missingDataPer)
trainData = trainData.drop('Cabin',axis = 1)

trainData.columns
trainData = trainData.fillna(trainData.mean())

trainData.Age.hist(bins=20)
trainData[trainData.Survived == 1].Age.hist()
trainData.Pclass.hist()
trainData[trainData.Survived == 1].Pclass.hist()
trainData.Sex.hist()
trainData[trainData.Survived == 1].Sex.hist()
trainData.dtypes
