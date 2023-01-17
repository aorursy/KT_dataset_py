import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plots
import graphviz # print tree
from sklearn import datasets, tree, model_selection, linear_model

from sklearn.metrics import mean_absolute_error

from __future__ import absolute_import, division, print_function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
X = pd.read_csv("../input/train.csv", usecols=['field','age','type', 'harvest_month']).values # harvest year foi retirado devido a análise 2
y = pd.read_csv("../input/train.csv", usecols=['production']).values
#X.head() #nao e possivel usar com o .values
#y.head() #nao e possivel usar com o .values

traindata = pd.read_csv("../input/train.csv", usecols=['field','age','type','harvest_month'])
traindata.head()

dataselect = pd.read_csv("../input/train.csv", usecols=['field','age','type','harvest_year','harvest_month'])
dataselect.head()
data = traindata.values
dataselect = dataselect.values


field_min = 0
field_max = 27
fieldData = {}
for i in range(field_min,field_max+1):
    file = "../input/field-"+str(i)+".csv"
    fieldData[i] = pd.read_csv(file, usecols=['temperature','dewpoint','windspeed','Soilwater_L1','Precipitation']).values

def getDataField(field, month, year):
    #todos os dados possuem tamanho 192, e vão de 01/2002 a 12/2017, o que significa que com base no mês e ano podemos calcular o índice de onde queremos os dados
    offsetano = int(year) - 2002
    offsetmes = int(month) #
    return fieldData[int(field)][((offsetano*12) + offsetmes)-1]


dataP = []
for i in range(len(data)):
    climatic = getDataField(dataselect[i][0], dataselect[i][4], dataselect[i][3]).tolist()
    el = data[i].tolist()
    el = el + climatic
    dataP.append(el)
#print(dataP[5:])
dataP = np.array(dataP)

print('done')
y = pd.read_csv("../input/train.csv", usecols=['production']).values
X = dataP
print(y.shape)
print(X.shape)
y = y.flatten()
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2, random_state = 33)
X_train = X
y_train = y

X_test = X
y_test = y

print(X_train.shape, y_train.shape)
clf = linear_model.Lasso(alpha=0.01)

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

mean_absolute_error(y_test, y_predict)
testdfRead = pd.read_csv('../input/test.csv')
testdf = testdfRead.values

testdata = pd.read_csv("../input/test.csv", usecols=['field','age','type','harvest_month'])
testdata.head()
testdata = testdata.values

dataTest = []
for i in range(len(testdata)):
    climatic = getDataField(testdf[i][1], testdf[i][5], testdf[i][4]).tolist()
    el = testdata[i].tolist()
    el = el + climatic
    dataTest.append(el)
dataTest = np.array(dataTest)
print(dataTest.shape)

test_predictions = clf.predict(dataTest)
test_predictions = test_predictions.flatten().tolist()
for i in range(len(test_predictions)):
    if test_predictions[i] < 0:
        test_predictions[i] = 0
    if test_predictions[i] > 1:
        test_predictions[i] = 1
submissiondf = submissiondf = pd.DataFrame(pd.DataFrame({'Id':testdfRead['Id'].values.flatten().tolist(),'production':test_predictions}))
submissiondf.to_csv('fit.csv', index=False)
import os
print(os.listdir("../"))

