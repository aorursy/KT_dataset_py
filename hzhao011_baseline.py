# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

%matplotlib inline
traindf = pd.read_csv('../input/train.csv')

testdf = pd.read_csv('../input/test.csv')



# check training data

totalRows = traindf[' y'].max()

totalCols = traindf['x'].max()

print("room size ", totalRows, ' ',totalCols)

trainData = traindf.iloc[:, 2:]

trainlabels = traindf.iloc[:, 0:2]



testData = testdf.iloc[:, 2:]

testlabels = testdf.iloc[:, 0:2]



dimOfdata = trainData.shape[1]

print("number of  features: ", dimOfdata)

numberOfSamples = len(trainlabels)

print('total samples: ', numberOfSamples)
train_X = trainData.values

train_Y = trainlabels.values

test_X = testData.values

test_Y = testlabels.values
#interpolate 插值

from scipy.interpolate import griddata

room_y = np.zeros(((totalRows+1)*(totalCols+1), 2))

k = 0

for i in range(totalCols+1):

    for j in range((totalRows+1)):

        room_y[k, :] = [i, j]

        k += 1

room_x = griddata(train_Y, train_X, room_y, method='cubic')

room_x = np.nan_to_num(room_x)
train_x, val_x, train_y, val_y = room_x, test_X, room_y,test_Y 
k = 4

model = MultiOutputRegressor(KNeighborsRegressor(k))

#model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100))

model.fit(train_x, train_y)
#predict

pred_results = model.predict(val_x)
#write to csv

resultdf = pd.read_csv('../input/sample_submission.csv')

realy, realx = resultdf.iloc[:, 0], resultdf.iloc[:, 1]

testdf[" y"] = pred_results[:, 1]

testdf["x"] = pred_results[:, 0]

data = {"realy":realy, "realx":realx, "predicty": pred_results[:, 0], "predictx":pred_results[:, 1]}

df = pd.DataFrame(data)
from sklearn.metrics import mean_squared_error

predicts = testdf.iloc[:, 0:2]

trues = resultdf.iloc[:, 0:2]



print("total mse: ", mean_squared_error(trues, predicts))



#testdf.loc['mse'] = testdf.apply(lambda x: x.sum())



df["mse"] = df.apply(lambda x: mean_squared_error([x["realx"], x["realy"]],[x["predictx"], x["predicty"]]), axis=1)

df.to_csv('./submission.csv')

df.head()
def to_int(x):

    return int(round(x))

imageSize = (totalRows+1,totalCols+1)

testlabels = resultdf.iloc[:, 0:2]

def getRouteImage():

    image = np.full(imageSize, 128)

    for index, lables in testlabels.iterrows():

        image[lables[' y']][lables['x']] = 80

    return image



image = getRouteImage()

for predict_y, predict_x in predicts.iloc[:, 0:2].values:

    predict_y, predict_x = to_int(predict_y), to_int(predict_x)

    image[predict_x][predict_y] = 255 



plt.figure(figsize=(20,20))

plt.title('images')

plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')