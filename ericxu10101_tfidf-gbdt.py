from sklearn import preprocessing, feature_extraction
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
import warnings
import logging
import time

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] : %(message)s ',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

logging.info('happy machine learning')
# %%
trainData = json.load(open("../input/train.json"))
# allIngredientsInTrain = [record['ingredients'] for record in trainData]
allIngredientsTextInTrain = [" ".join(record['ingredients']).lower() for record in trainData]

# %%
allCuisineInTrain = [record['cuisine'] for record in trainData]
labelEncoder = preprocessing.LabelEncoder()
labelEncodedCuisine = labelEncoder.fit_transform(allCuisineInTrain)
# %%
logging.info('init TfidfVectorizer on train ...')

tfidfVectorizer = feature_extraction.text.TfidfVectorizer(binary=True)
trainDataCSRMatrix = tfidfVectorizer.fit_transform(allIngredientsTextInTrain)
testData = json.load(open("../input/test.json"))
testDataSize = len(testData)
# allIngredientsInTest = [record['ingredients'] for record in testData]
allIngredientsTextInTest = [" ".join(record['ingredients']).lower() for record in testData]
testDataCSRMatrix = tfidfVectorizer.transform(allIngredientsTextInTest)
xgbTrain = xgb.DMatrix(trainDataCSRMatrix, label=labelEncodedCuisine)

param = {
    'max_depth': 5,
    'eta': 0.3,
    'silent': 1,
    'objective': 'multi:softmax',
    'num_class': 20,
    'nthread': 200
}

num_round = 100

watchlist = [(xgbTrain, 'train')]
logging.info('cross validating ...')
xgb.cv(param, xgbTrain, num_round,
       nfold=5,
       metrics={'merror'},
       seed=0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
plt.plot(np.arange(num_round), validationResult['train-merror-mean'], validationResult['test-merror-mean'])
plt.show()
start = time.time()
logging.info('start train gbdt at %.3f', time.time())
gbdt = xgb.train(param, xgbTrain, num_round, watchlist)
end = time.time()
logging.info('train gbdt done at %.3f, time = %.3f', end, end - start)
# %%
xgbTest = xgb.DMatrix(testDataCSRMatrix)

# %%
predictedLabel = gbdt.predict(xgbTest)
predictedLabel = [int(a) for a in predictedLabel]
predictedCuisine = labelEncoder.inverse_transform(predictedLabel)
# %%
with open('gbdt_tfidf.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'cuisine'])
    for i in range(testDataSize):
        writer.writerow([testData[i]['id'], predictedCuisine[i]])
    logging.info('prepare submit done !')