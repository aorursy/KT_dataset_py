# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



# seed

np.random.seed(1337)



# load avocado file

avocadoCSV = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv', parse_dates=['Date'], dtype={"region": "category","type": "category","year": "category"})  



import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('seaborn-darkgrid')

fig1 = plt.figure(1, figsize=(14,7))



sns.distplot(avocadoCSV['AveragePrice'],color='b',  axlabel='Average Price')

fig2 = plt.figure(2, figsize=(5,5))

sns.boxplot(x="type", y="AveragePrice", data=avocadoCSV, palette="Set1")



fig3 = plt.figure(3, figsize=(8,7))

sns.boxplot(x="type", y="AveragePrice", hue='year', data=avocadoCSV, palette="Set1")
avocadoCSV = avocadoCSV.sample(frac=1) # randomize sample

avocadoCSV.head()
print("dataset shape = ",avocadoCSV.shape)

avocadoCSV.describe()
avocadoCSV['month'] = avocadoCSV['Date'].map(lambda x: x.month)

avocadoCSV['month'] = avocadoCSV['month'].astype('category')



#avocado=avocadoCSV[[ 'month' ,'year', 'region', 'Small Bags', 

#'Large Bags','XLarge Bags', 'type','AveragePrice']]
def makeAvocadoWithCategory(data, categoryColumns, fieldsToKeep):



	allFields = categoryColumns + fieldsToKeep

	df = data[allFields]



	dfCategories = [ pd.get_dummies(df[column], prefix=column) for column in categoryColumns ]

	df = pd.concat([df] + dfCategories, axis=1)

	df = df.drop(columns=categoryColumns)



	return df



avocado = makeAvocadoWithCategory(

	avocadoCSV,

	['month' ,'year', 'region','type'],

	['Small Bags','Large Bags','XLarge Bags','AveragePrice']

)
avocado.shape
avocado.columns
def makeTrainAndTestSet(data):

	dataAveragePrice = data['AveragePrice']

	dataNoAveragePrice = data.drop(columns=['AveragePrice'])



	dataTrain = dataNoAveragePrice[:15000]

	dataYTrain = dataAveragePrice[:15000]





	dataTest = dataNoAveragePrice[15001:]

	dataYTest = dataAveragePrice[15001:]

	return dataTrain, dataYTrain, dataTest,dataYTest



avocadoTrain, avocadoYTrain, avocadoTest, avocadoYTest = makeTrainAndTestSet(avocado)
avocadoTrain.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten



model = Sequential()

model.add(Dense(4, activation='relu', input_dim=75))

model.add(Dense(6, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(6, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1))

model.compile('adam', loss='mean_squared_error')
history1 = model.fit(avocadoTrain, avocadoYTrain, epochs=15)
def makePredictSummary(modelTo, XtestData, YtestData):

	pred = modelTo.predict(XtestData).T[0]

	real = YtestData.values

	# compute relative error

	err = np.abs((real - pred) / real)

	predictionSummary = pd.DataFrame({'real': real, 'pred': pred, 'err(%)': err})



	return predictionSummary



summary = makePredictSummary(model, avocadoTest, avocadoYTest)

summary[:20]
from keras.layers import BatchNormalization



model2 = Sequential()

model2.add(Dense(6, activation='relu', input_dim=75))

model2.add(Dense(6, activation='relu'))

model2.add(BatchNormalization())

model2.add(Dense(10, activation='relu'))

model2.add(Dropout(0.25))

model2.add(Dense(16, activation='elu'))

model2.add(Dense(10, activation='elu'))

model2.add(Dropout(0.5))

model2.add(Dense(6, activation='relu'))

model2.add(BatchNormalization())

model2.add(Dense(4, activation='relu'))

model2.add(Dense(4, activation='relu'))

model2.add(Dense(1))



model2.compile("adam", loss='mean_squared_error')
history2 = model2.fit(avocadoTrain, avocadoYTrain, epochs=20, batch_size=64)
summary2 = makePredictSummary(model2, avocadoTest, avocadoYTest)

summary2[:20]
avocadoYTrainExp = np.power(3, avocadoYTrain)

avocadoYTestExp = np.power(3, avocadoYTest)

historyExp2 = model2.fit(avocadoTrain, avocadoYTrainExp, epochs=20, batch_size=64)
summaryExp = makePredictSummary(model2, avocadoTest, avocadoYTestExp)

summaryExp
plt.figure(4, figsize = (7,4))



plt.plot(history1.history['loss'], '-p', markersize=6, linewidth=2)

plt.title('First Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['learning rate 0.001'], loc='upper left')
from keras import optimizers





adam2 = optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=adam2, loss='mean_squared_error')

historyExp2_2 = model.fit(avocadoTrain,  avocadoYTrainExp, epochs=20, batch_size=64, verbose=0)



adam3 = optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=adam3, loss='mean_squared_error')

historyExp2_3 = model.fit(avocadoTrain,  avocadoYTrainExp, epochs=20, batch_size=64, verbose=0)



plt.figure(5, figsize = (9,4))

plt.plot(historyExp2_2.history['loss'], '-p', markersize=6, linewidth=2)

plt.plot(historyExp2_3.history['loss'], '-p', markersize=6, linewidth=2)

plt.title('Second Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['0.0001', '0.0005'], loc='upper left')
# remove the year column as a category



avocadoNew = makeAvocadoWithCategory(

	avocadoCSV,

	['month' ,'region','type'],

	['Small Bags','Large Bags','XLarge Bags','AveragePrice', 'year']

)



avocadoNewTrain, avocadoNewYTrain, avocadoNewTest, avocadoNewYTest = makeTrainAndTestSet(avocado)



avocadoNewYTrain = np.power(3, avocadoNewYTrain)

avocadoNewYTest = np.power(3, avocadoNewYTest)
model3 = Sequential()

model3.add(Dense(6, activation='relu', input_dim=75))

model3.add(Dense(6, activation='relu'))

model3.add(BatchNormalization())

model3.add(Dense(10, activation='relu'))

model3.add(Dropout(0.25))

model3.add(Dense(16, activation='elu'))

model3.add(Dense(10, activation='elu'))

model3.add(Dropout(0.5))

model3.add(Dense(6, activation='relu'))

model3.add(BatchNormalization())

model3.add(Dense(4, activation='relu'))

model3.add(Dense(4, activation='relu'))

model3.add(Dense(1))



adam = optimizers.Adam(learning_rate=0.0003) #new learning rate

model3.compile(adam, loss='mean_squared_error')