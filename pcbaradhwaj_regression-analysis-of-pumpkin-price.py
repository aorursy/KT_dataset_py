# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt



def processDataFrame(df):

    size_map = {'jbo': 1,    'sml': 2,    'med': 3,

                'med-lge': 4,    'lge': 5,    'xlge': 6,

                'exjbo': 7}

    df = df.assign(

         size = df['Item Size'].map(size_map),

         avgPrice = (df['Low Price'] + df['High Price'])/2,

         sizeClass =(df['Item Size'].map(size_map) >= 3).astype(int)

         )

    df = df[['size','avgPrice','sizeClass']]  

    return df

    

# Read all cssv files    

atlanta = pd.read_csv('../input/atlanta_9-24-2016_9-30-2017.csv')

atlanta = processDataFrame(atlanta)

baltimore = pd.read_csv('../input/baltimore_9-24-2016_9-30-2017.csv')

baltimore = processDataFrame(baltimore)

boston = pd.read_csv('../input/boston_9-24-2016_9-30-2017.csv')

boston = processDataFrame(boston)

chicago = pd.read_csv('../input/chicago_9-24-2016_9-30-2017.csv')

chicago = processDataFrame(chicago)

colombia = pd.read_csv('../input/columbia_9-24-2016_9-30-2017.csv')

colombia = processDataFrame(colombia)

dallas = pd.read_csv('../input/dallas_9-24-2016_9-30-2017.csv')

dallas = processDataFrame(dallas)

detroit = pd.read_csv('../input/detroit_9-24-2016_9-30-2017.csv')

detroit = processDataFrame(detroit)

losAngles = pd.read_csv('../input/los-angeles_9-24-2016_9-30-2017.csv')

losAngles = processDataFrame(losAngles)

miami = pd.read_csv('../input/miami_9-24-2016_9-30-2017.csv')

miami = processDataFrame(miami)

newYork = pd.read_csv('../input/new-york_9-24-2016_9-30-2017.csv')

newYork = processDataFrame(newYork)

philD = pd.read_csv('../input/philadelphia_9-24-2016_9-30-2017.csv')

philD = processDataFrame(philD)

sanFran = pd.read_csv('../input/san-fransisco_9-24-2016_9-30-2017.csv')

sanFran = processDataFrame(sanFran)

stLouis = pd.read_csv('../input/st-louis_9-24-2016_9-30-2017.csv')

stLouis = processDataFrame(stLouis)
# Perform Train test split in all data

def trainTestSegregation(df) : 

    all_records= np.arange(df.shape[0])

    trainingRecordCount = round(0.7 *df.shape[0])

    testRecordCount = round(0.3 * df.shape[0])

    np.random.seed(100)

    trainingRecordsIds = np.random.choice(all_records,trainingRecordCount ,replace=False)

    testingRecordsIds =all_records[~np.in1d(all_records,trainingRecordsIds)] 

    trainingRecords = df.iloc[testingRecordsIds,:]

    testRecords = df.iloc[testingRecordsIds,:]

    return trainingRecords, testRecords



atlantaTraining,atlantaTesting = trainTestSegregation(atlanta)

baltimoreTraining,baltimoreTesting = trainTestSegregation(baltimore)

bostonTraining,bostonTesting = trainTestSegregation(boston)

chicagoTraining,chicagoTesting = trainTestSegregation(chicago)

newYorkTraining,newYorkTesting = trainTestSegregation(newYork)



# Identify Direct and Indirect Variable

#Based on size estimate price 

# DV : Price IDV : size

# Correlation

print(atlantaTraining['size'].corr(atlantaTraining['avgPrice'])) # 0.004489 or 0.01228

print(baltimoreTraining['size'].corr(baltimoreTraining['avgPrice']))#0.715499939368 -> strong +

print(bostonTraining['size'].corr(bostonTraining['avgPrice']))#0.472942356909 ->

print(chicagoTraining['size'].corr(chicagoTraining['avgPrice']))# 0.603218651914 -> strong +

print(newYorkTraining['size'].corr(newYorkTraining['avgPrice'])) # 0.785152670629 -> strong +

 # 0: Weak correlation (no relationship).

## Concluding that no relationship can be built

## Step 4: Build Regression Model

# Least Squares method

# Formula: DV ~ IDV

baltimoreModel = smf.ols(formula='avgPrice ~ size',data = atlantaTraining).fit()

baltimoreModel.summary()

# R - Squared : 0.000 (# 0 - Bad model)

# avgPrice =  147.5515 + 0.3750 * size

# P  value : 0.963 -> size

# only IDVs with p values less than 0.05 should be included

# p value should be less than 0.05 for the IDV to be significant

# size  is insignificant - can be ignored

# Intercept is significant 

### Boston Model

bostonModel = smf.ols(formula='avgPrice ~ size',data = bostonTraining).fit()

bostonModel.summary()

# price   = 100.49 + 20.4214 * size

# R square: 0.224 - poor model.

# P value is significant for both intercept and idv



chicagoModel = smf.ols(formula='avgPrice ~ size',data = chicagoTraining).fit()

chicagoModel.summary()

# R squqare : 0.364

# P - Intercept : 0.102 - Insignificant

# P - IDV : 0.000 - significant

#44.3997 * size - 45.1840

newYorkModel = smf.ols(formula='avgPrice ~ size',data = newYorkTraining).fit()

newYorkModel.summary()

# R sqaure  : 0.616 

# P Intercept : 0.299 - Insignificant

# P size: 33.2642

#avgWeight = 33.2642 * size -22.6877

# Since p value is insignificant  for intercept we ignore ,  avgWeight = 33.2642 * size
# Build new York model : All values decently matching

# Copy data to new dataframe : newYorkTraining,newYorkTesting

def MAPE(actual,predicted):

    abs_percent_diff = abs(actual-predicted)/actual

    # 0 actual values might lead to infinite errors

    # replacing infinite with nan

    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)

    median_ape = np.median(abs_percent_diff)

    mean_ape = np.mean(abs_percent_diff)

    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])

    return mape

newYorkTestingCopy  =newYorkTesting.copy()

del newYorkTestingCopy['avgPrice']

predictedAvgPrice = newYorkModel.predict(newYorkTestingCopy)

#MAPE(newYorkTesting['avgPrice'],predictedAvgPrice) # Mean: 0.603389 Median : Nan 60 % error :(  - Bad Model

newYorkTrainingasTestingCopy  =newYorkTraining.copy()

del newYorkTrainingasTestingCopy['avgPrice']

newYorkTrainingasTestingCopy['predictedavgPrice'] = newYorkModel.predict(newYorkTrainingasTestingCopy)

MAPE(newYorkTesting['avgPrice'],newYorkTrainingasTestingCopy['predictedavgPrice']) # Mean: 0.603389 Median : Nan 60 % error :(  - Bad Model

plt.scatter('size','avgPrice',data=newYorkTraining)

plt.scatter('size','predictedavgPrice',data=newYorkTrainingasTestingCopy,color='r')

#newYorkModel

print(colombia['size'].corr(colombia['avgPrice'])) # Weak +  Corr:0.18116

print(dallas['size'].corr(dallas['avgPrice'])) # Weak - correlation : -0.2775

print(detroit['size'].corr(detroit['avgPrice'])) # weak + corr : 0.1705

print(losAngles['size'].corr(losAngles['avgPrice'])) # weal + corr : 0.2573

print(miami['size'].corr(miami['avgPrice'])) # strong + correlation : 1.0

print(philD['size'].corr(philD['avgPrice'])) # weak + corr : 0.18772

print(sanFran['size'].corr(sanFran['avgPrice'])) # weak - corr : -0.10569

print(stLouis['size'].corr(stLouis['avgPrice'])) # st louis negative corr : -0.56222
# Build Model for st Louis

# Perform Train test split 

# Get all row index using arange

stLouisCount = np.arange(stLouis.shape[0])

stLouisTrCount = round(0.7 *stLouis.shape[0]) # 72

stLouisTeCount = round(0.3 * stLouis.shape[0]) #31

np.random.seed(10)

stLouisTrRows = np.random.choice(stLouisCount,stLouisTrCount,replace=False)

stLouisTeRows = stLouisCount[~np.in1d(stLouisCount,stLouisTrRows)]

stLouisTrValue = stLouis.iloc[stLouisTrRows,:]

stLouisTeValue = stLouis.iloc[stLouisTeRows,:]



# Build Model

stLouisModel = smf.ols(formula = 'avgPrice ~ size', data =stLouis).fit()

stLouisModel.summary()

# Rvalue : 0.316 # Bad model fit

#avgprice = 220.3333 -17.6667 * size

# P value is significant for both values 
# Test the effieicny of the model

# Test for testing data

stLouisTeValueCopy = stLouisTeValue.copy()

del stLouisTeValueCopy['avgPrice']

predictedstLouisavgPrice = stLouisModel.predict(stLouisTeValueCopy)

#MAPE(stLouisTeValue['avgPrice'],predictedstLouisavgPrice)  # MEan : 0.19389 Median : Nan -  19 % error

# Decent model with little variation



## Create a copy of training data as test and verify against this model

stLouisTrTestCopy = stLouisTrValue.copy()

del stLouisTrTestCopy['avgPrice']

stLouisTrTestCopy['predictedavgPrice'] = stLouisModel.predict(stLouisTrTestCopy)

MAPE(stLouisTrValue['avgPrice'],stLouisTrTestCopy['predictedavgPrice'])  # 18 % - 0.180719

## Plot a graph with traning data avg Price and predicted avgPrice in training as test data

plt.scatter('size','avgPrice',data=stLouisTrValue)

plt.scatter('size','predictedavgPrice',data=stLouisTrTestCopy,color='g')
