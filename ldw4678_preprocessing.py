import pandas as pd
from collections import Counter

%pylab inline

trainPath = r'../input/train.csv'
testPath = r'../input/test.csv'
train_df = pd.read_csv(trainPath)
test_df = pd.read_csv(testPath)
train_df.info()
test_df.info()
#deal character data
train_df['Year']  = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
train_df['Month'] = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
train_df['Week']  = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

test_df['Year']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
test_df['Month'] = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
test_df['Week']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

train_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
test_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
#get character data columns
notNumTypeCol = [col for col in train_df.columns if train_df[col].dtype == dtype('O')]
notNumTypeCol
def charToInt(data_df,convDict):
    #create map dict
    for colName in convDict.keys():
        valuesCounter = convDict[colName]
        valuesCounterSize = len(valuesCounter)
        mapDict = dict(zip(valuesCounter.keys(),range(valuesCounterSize)))
        for index,record in enumerate(data_df[colName]):
            data_df[colName][index] = mapDict[record]
notNumTypeCol
import pandas as pd
from collections import Counter

%pylab inline

trainPath = r'../input/train.csv'
testPath = r'../input/test.csv'

train_df = pd.read_csv(trainPath)
test_df = pd.read_csv(testPath)

#deal character data
train_df['Year']  = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
train_df['Month'] = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
train_df['Week']  = train_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

test_df['Year']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
test_df['Month'] = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
test_df['Week']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

train_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
test_df.drop(['Original_Quote_Date'], axis=1,inplace=True)

#get character data columns
notNumTypeCol = [col for col in train_df.columns if train_df[col].dtype == dtype('O')]


def charToInt(data_df,convDict):
    #create map dict
    for colName in convDict.keys():
        valuesCounter = convDict[colName]
        valuesCounterSize = len(valuesCounter)
        mapDict = dict(zip(valuesCounter.keys(),range(valuesCounterSize)))
        for index,record in enumerate(data_df[colName]):
            data_df[colName][index] = mapDict[record]

train_x_df = train_df.drop("QuoteConversion_Flag",axis=1)
train_y_df = train_df["QuoteConversion_Flag"]
charToInt(train_x_df,notNumTypeCol)
notNumValueCounter = {}
for key in col:
    notNumValueCounter[key] = Counter(train_x_df[key])
notNumValueCounter = {}
for key in notNumTypeCol:
    notNumValueCounter[key] = Counter(train_x_df[key])
charToInt(train_x_df,notNumValueCounter)
