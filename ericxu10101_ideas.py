from sklearn import linear_model, metrics, ensemble, tree, preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import json
import csv
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# %%
trainData = json.load(open("../input/train.json"))

allCuisineInTrain = [record['cuisine'] for record in trainData]

# %%
labelEncoder = preprocessing.LabelEncoder()
labelEncodedCuisine = labelEncoder.fit_transform(allCuisineInTrain)

# %%
allIngredientsInTrain = [record['ingredients'] for record in trainData]
allIngredientsInTrain = np.hstack(allIngredientsInTrain)

# %%
ingredientLabelEncoder = preprocessing.LabelEncoder()
labeledIngredient = ingredientLabelEncoder.fit_transform(allIngredientsInTrain)

# %%
featureSize = len(ingredientLabelEncoder.classes_)
trainDataSize = len(trainData)

# %%
trainDataMatrix = np.zeros((trainDataSize, featureSize))


# %%

def setTrainDataMatrix(row, cols):
    for col in cols:
        trainDataMatrix[row][col] = 1


for i in range(trainDataSize):
    if i % 200 == 0:
        print("completed %.2f%%" % (i * 100 / trainDataSize))
    setTrainDataMatrix(i, ingredientLabelEncoder.transform(trainData[i]['ingredients']))
print("done !")

# %%
testData = json.load(open("../input/test.json"))

# %%
testDataSize = len(testData)
testDataMatrix = np.zeros((testDataSize, featureSize))


# %%
def setTestDataMatrix(row, cols):
    for col in cols:
        testDataMatrix[row][col] = 1


for i in range(testDataSize):
    if i % 200 == 0:
        print("completed %.2f%%" % (i * 100 / testDataSize))
    try:
        setTestDataMatrix(i, ingredientLabelEncoder.transform(testData[i]['ingredients']))
    except ValueError as err:
#         print("value error: {0}".format(err))
        pass
print("done !")
xgbTrain = xgb.DMatrix(trainDataMatrix, label=labelEncodedCuisine)

param = {
    'max_depth': 5,
    'eta': 0.5,
    'silent': 1,
    'objective': 'multi:softmax',
    'num_class': 20,
    'nthread': 200
}

num_round = 60

watchlist = [(xgbTrain, 'train')]
gbdt = xgb.train(param, xgbTrain, num_round, watchlist)
xgbTest = xgb.DMatrix(testDataMatrix)

# %%
predictedLabel = gbdt.predict(xgbTest)
predictedLabel = [int(a) for a in predictedLabel]
predictedCuisine = labelEncoder.inverse_transform(predictedLabel)
with open('gbdt_d5_e04_n30_omultisoftmax.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'cuisine'])
    for i in range(testDataSize):
        writer.writerow([testData[i]['id'], predictedCuisine[i]])
    print('done !')