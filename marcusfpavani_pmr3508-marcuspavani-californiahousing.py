import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import copy as cp

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
import os 
os.listdir('../input')
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
trainData.shape
trainData.head()
# Análise da Idade Média

medianAge = trainData['median_age']
maxMedianAge = medianAge.max()
minMedianAge = medianAge.min()
print('minMedianAge: ', minMedianAge, ', maxMedianAge: ', maxMedianAge)

# Divisão em Grupos:
leg = '0-10','15-20','20-30','30-40','40-max'
medianAgeGroups = np.array([0,0,0,0,0])
for i in range(len(medianAge)):
    if medianAge.iloc[i] <= 10:
        medianAgeGroups[0] = medianAgeGroups[0] + 1
    if medianAge.iloc[i] > 10 and medianAge.iloc[i] <= 20:
        medianAgeGroups[1] = medianAgeGroups[1] + 1
    if medianAge.iloc[i] > 20 and medianAge.iloc[i] <= 30:
        medianAgeGroups[2] = medianAgeGroups[2] + 1
    if medianAge.iloc[i] > 30 and medianAge.iloc[i] <= 40:
        medianAgeGroups[3] = medianAgeGroups[3] + 1
    if medianAge.iloc[i] > 40:
        medianAgeGroups[4] = medianAgeGroups[4] + 1
plt.pie(medianAgeGroups, labels = leg);
# Análise da quantidade de cômodos

totalRooms = trainData['total_rooms']
maxRooms = totalRooms.max()
minRooms = totalRooms.min()
print('maxRooms: ', maxRooms, ', minRooms: ', minRooms)

# Divisão em Grupos:
leg = '0-1','1-2.5','2.5-5','5-10','10-40'
totalRoomsGroups = np.array([0,0,0,0,0])
for i in range(len(totalRooms)):
    if totalRooms.iloc[i] <= 1000:
        totalRoomsGroups[0] = totalRoomsGroups[0] + 1
    if totalRooms.iloc[i] > 1000 and totalRooms.iloc[i] <= 2500:
        totalRoomsGroups[1] = totalRoomsGroups[1] + 1
    if totalRooms.iloc[i] > 2500 and totalRooms.iloc[i] <= 5000:
        totalRoomsGroups[2] = totalRoomsGroups[2] + 1
    if totalRooms.iloc[i] > 5000 and totalRooms.iloc[i] <= 10000:
        totalRoomsGroups[3] = totalRoomsGroups[3] + 1
    if totalRooms.iloc[i] > 10000:
        totalRoomsGroups[4] = totalRoomsGroups[4] + 1
plt.pie(totalRoomsGroups, labels = leg);
# Análise da quantidade de Quartos

totalBedrooms = trainData['total_bedrooms']
maxBedrooms = totalBedrooms.max()
minBedrooms = totalBedrooms.min()
print('maxBedrooms: ', maxBedrooms, ', minBedrooms: ', minBedrooms)

# Divisão em Grupos:
leg = '0-0.25','0.25-0.5','0.5-1','1-2.5','2.5-max'
totalBedroomsGroups = np.array([0,0,0,0,0])
for i in range(len(totalBedrooms)):
    if totalBedrooms.iloc[i] <= 250:
        totalBedroomsGroups[0] = totalBedroomsGroups[0] + 1
    if totalBedrooms.iloc[i] > 250 and totalBedrooms.iloc[i] <= 500:
        totalBedroomsGroups[1] = totalBedroomsGroups[1] + 1
    if totalBedrooms.iloc[i] > 500 and totalBedrooms.iloc[i] <= 1000:
        totalBedroomsGroups[2] = totalBedroomsGroups[2] + 1
    if totalBedrooms.iloc[i] > 1000 and totalBedrooms.iloc[i] <= 2500:
        totalBedroomsGroups[3] = totalBedroomsGroups[3] + 1
    if totalBedrooms.iloc[i] > 2500:
        totalBedroomsGroups[4] = totalBedroomsGroups[4] + 1
plt.pie(totalBedroomsGroups, labels = leg);
# Análise da População

population = trainData['population']
maxPopulation = population.max()
minPopulation = population.min()
print('minPopulation: ', minPopulation, ', maxPopulation: ', maxPopulation)

# Divisão em Grupos:
leg = '0-1','1-2','2-3','3-5','5-max'
populationGroups = np.array([0,0,0,0,0])
for i in range(len(population)):
    if population.iloc[i] <= 1000:
        populationGroups[0] = populationGroups[0] + 1
    if population.iloc[i] > 1000 and population.iloc[i] <= 2000:
        populationGroups[1] = populationGroups[1] + 1
    if population.iloc[i] > 2000 and population.iloc[i] <= 3000:
        populationGroups[2] = populationGroups[2] + 1
    if population.iloc[i] > 3000 and population.iloc[i] <= 5000:
        populationGroups[3] = populationGroups[3] + 1
    if population.iloc[i] > 5000:
        populationGroups[4] = populationGroups[4] + 1
plt.pie(populationGroups, labels = leg);
# Análise da quantidade de Propriedades

households = trainData['households']
maxHouseholds = households.max()
minHouseholds = households.min()
print('minHouseholds: ', minHouseholds, ', maxHouseholds: ', maxHouseholds)

# Divisão em Grupos:
leg = '0-0.25','0.25-0.5','0.5-1','1-2','2-max'
householdGroups = np.array([0,0,0,0,0])
for i in range(len(households)):
    if households.iloc[i] <= 250:
        householdGroups[0] = householdGroups[0] + 1
    if households.iloc[i] >  250 and households.iloc[i] <= 500:
        householdGroups[1] = householdGroups[1] + 1
    if households.iloc[i] >  500 and households.iloc[i] <= 1000:
        householdGroups[2] = householdGroups[2] + 1
    if households.iloc[i] > 1000 and households.iloc[i] <= 2000:
        householdGroups[3] = householdGroups[3] + 1
    if households.iloc[i] > 2000:
        householdGroups[4] = householdGroups[4] + 1
plt.pie(householdGroups, labels = leg);
# Análise da Renda Média

medianIncome = trainData['median_income']
maxMedianIncome = medianIncome.max()
minMedianIncome = medianIncome.min()
print('minMedianIncome: ', minMedianIncome, ', maxMedianIncome: ', maxMedianIncome)

# Divisão em Grupos:
leg = '0-20','20-30','30-40','40-50','50-max'
medianIncomeGroups = np.array([0,0,0,0,0])
for i in range(len(medianIncome)):
    if medianIncome.iloc[i] <= 20000:
        medianIncomeGroups[0] = medianIncomeGroups[0] + 1
    if medianIncome.iloc[i] > 20000 and medianIncome.iloc[i] <= 30000:
        medianIncomeGroups[1] = medianIncomeGroups[1] + 1
    if medianIncome.iloc[i] > 30000 and medianIncome.iloc[i] <= 40000:
        medianIncomeGroups[2] = medianIncomeGroups[2] + 1
    if medianIncome.iloc[i] > 40000 and medianIncome.iloc[i] <= 50000:
        medianIncomeGroups[3] = medianIncomeGroups[3] + 1
    if medianIncome.iloc[i] > 50000:
        medianIncomeGroups[4] = medianIncomeGroups[4] + 1
plt.pie(medianIncomeGroups, labels = leg);
# Relação entre a Renda Média e o valor médio da propriedade
medianHouseValue = trainData['median_house_value']
plt.scatter(medianIncome,medianHouseValue);
# Relação entre a Latitude, Longitude e o valor médio da propriedade
latitude = trainData['latitude']
longitude = trainData['longitude']
scaledValues = (medianHouseValue - medianHouseValue.min())/medianHouseValue.ptp()
colors = plt.cm.coolwarm(scaledValues)
plt.scatter(latitude,longitude,marker='.',color=colors)
trainDataX = trainData[["latitude","longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
trainDataY = medianHouseValue
testData  = testData[["latitude","longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
meanScoreManhattan = np.zeros(15)
stdScoreManhattan  = np.zeros(15)
for k in range(1,16):
    knnRegressor = KNeighborsRegressor(n_neighbors=k, p=1)
    score = cross_val_score(knnRegressor, trainDataX, trainDataY, cv=10)
    meanScoreManhattan[k-1] = np.mean(score)
    stdScoreManhattan[k-1]  = np.std(score)
    
np.amax(meanScoreManhattan)

meanScoreEuclidean = np.zeros(15)
stdScoreEuclidean  = np.zeros(15)
for k in range(1,16):
    knnRegressor = KNeighborsRegressor(n_neighbors=k, p=2)
    score = cross_val_score(knnRegressor, trainDataX, trainDataY, cv=10)
    meanScoreEuclidean[k-1] = np.mean(score)
    stdScoreEuclidean[k-1]  = np.std(score)
    
np.amax(meanScoreEuclidean)

if np.amax(meanScoreManhattan) > np.amax(meanScoreEuclidean):
    chosenK = np.argmax(meanScoreManhattan)+1
    chosenP = 1
else:
    chosenK = np.argmax(meanScoreEuclidean)+1
    chosenP = 2
    
print('Chosen K: ', chosenK, ', chosen P: ', chosenP)
knnRegressor = KNeighborsRegressor(n_neighbors=chosenK,p=chosenP)
knnRegressor.fit(trainDataX,trainDataY)

knnPredictedData = knnRegressor.predict(testData)
knnPredictedData
lassoRegressor = linear_model.Lasso(alpha=0.5)
lassoRegressor.fit(trainDataX,trainDataY)

lassoPredictedData = lassoRegressor.predict(testData)

for i in range(len(lassoPredictedData)):
    if lassoPredictedData[i] < 0:
        lassoPredictedData[i] = -lassoPredictedData[i]

lassoPredictedData
ridgeRegressor = Ridge(alpha=1.0)
ridgeRegressor.fit(trainDataX,trainDataY)

ridgePredictedData = ridgeRegressor.predict(testData)

for i in range(len(ridgePredictedData)):
    if ridgePredictedData[i] < 0:
        ridgePredictedData[i] = -ridgePredictedData[i]
        
ridgePredictedData
submissionId    = testData.index
knnSubmission   = pd.DataFrame({'Id':submissionId[:],'median_house_value':knnPredictedData[:]})
lassoSubmission = pd.DataFrame({'Id':submissionId[:],'median_house_value':lassoPredictedData[:]})
ridgeSubmission = pd.DataFrame({'Id':submissionId[:],'median_house_value':ridgePredictedData[:]})

knnSubmission.to_csv('knnSubmission.csv', index = False)
lassoSubmission.to_csv('lassoSubmission.csv', index = False)
ridgeSubmission.to_csv('ridgeSubmission.csv', index = False)
