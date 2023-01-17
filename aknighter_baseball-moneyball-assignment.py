'''Notebook Setup:
This notebook presents the data analysis for the Moneyball Baseball Problem.

The summary report can be found in this file: 411 Assignment 1 Andrew Knight.pdf

The data file used: DataDictionary_Baseball.xlsx
'''

# Python imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf  
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import stats
#from sklearn import datasets, linear_model
#from sklearn.feature_selection import f_regression

from statsmodels.graphics.gofplots import ProbPlot

plt.style.use('seaborn') # pretty matplotlib plots

plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

print('Imports completed')
# Load the train and test data
try:
    traindf = pd.read_csv('../input/train.csv')
    testdf = pd.read_csv('../input/test.csv')
    print('train and test dataframes loaded')
except:
    print('file read error')
print(traindf.info())
print(testdf.info())
# The number of NaN is important to understand how many of each variables we have.
#traindf.head(10)
count_nan = len(traindf) - traindf.count()
print(count_nan)

#Also for the test set
count_nan_test = len(testdf) - testdf.count()
print(count_nan_test)
#traindf.tail(50)
#Note the missing values are recorded as NaN, we need to replace these with something (median? or your choice)
#start by converting all NaN values to 0
traindf = traindf.fillna(0)
#m = np.median(traindf.TEAM_BATTING_HBP[traindf.TEAM_BATTING_HBP>0])
#traindf['TEAM_PITCHING_IMP'] = 0 # flag for values that get imputed, default of zero indicates original values are used
train1 = traindf
train1 = train1.fillna(0)

#train1 = traindf.replace({'TEAM_BATTING_HBP': {0: m}}) 
#print(train1)

#Make sure whatever you do to the training data you also do to the test data otherwise your model will not score properly
testdf = testdf.fillna(0)
#testdf['TEAM_PITCHING_IMP'] = 0 # flag for values that get imputed, default of zero indicates original values are used
test1 = testdf
test1 = test1.fillna(0)
#test1 = testdf.replace({'TEAM_BATTING_HBP': {0: m}})
#print(test1)

# NOTE: I decided not to use this variable, but rather to simply change in place for this assignment.
# modified from template
print('')
print('----- Summary of Input Data -----')
print('')

# show the object is a DataFrame
print('Object type: ', type(traindf))

# show number of observations in the DataFrame
print('Number of observations: ', len(traindf))

# show variable names
print('Variable names: ', traindf.columns)

# show descriptive statistics
print(traindf.describe())

# show a portion of the beginning of the DataFrame
#print(traindf.head())

#Some quick plots of the data
traindf.hist(figsize=(20,16))
traindf.plot(kind= 'box' , subplots=True, layout=(5,5), sharex=False, sharey=False, figsize=(10,8))
'''
These data do not provide dates but rather a complete listing of team stats between years 1871 and 2006. However, we don't neccesarily need 
this dated information to build a model for predicted number of wins. That said, it might be nice to come up with an estiamte of the number of 
teams represented per year. Also, we are assuming from the given instructions that data has been standardized for a 162 game season, for each year.
This means that we will assume each record represents a team that played a total of 162 that year.

'''
years = 2006 - 1871
print('years of baseball in data: ' + str(years))

teams = round(len(traindf) / years, 0)
print('avg teams represented in data: ' + str(teams))
# Section 1 Results:
'''
These following items should be addressed before building the model.
'''
# 1. Due to the number of missing values in variables TEAM_BATTING_HBP, TEAM_BASERUN_CS I will likely not be using these variables in my models.

# 2. Bsaed on the distributions shown above, these variables may be candidates for using a transformation
# TEAM_BASERUN_SB, TEAM_PITCHING_BB, TEAM_PITCHING_H, TEAM_PITCHING_HR

# These variables
# First lets see how many records contain zero wins
train1[train1['TARGET_WINS'] < 10]
# We have one record that contains zero wins, this seems suspect for a 100+ game season, no matter how bad they are.
# Let's drop this record as I believe this data is invalid
train1 = train1.drop([1210])

# Also considered updating the target wins for this one record to 0.1 to avoid a division by zero situation
# train1.iloc[1210, 2] = 0.1
# Next, let's check some of the other pitching stats with zero values that seem suspect.
for v in traindf.columns:
    print('Zeros in ' + str(v) + ': ' + str(len(train1[train1[str(v)] < 1])))

print(len(train1[train1['TEAM_PITCHING_HR'] < 1]))
print(len(train1[train1['TEAM_PITCHING_BB'] < 1]))
print(len(train1[train1['TEAM_PITCHING_SO'] < 1]))
print(len(train1[train1['TEAM_PITCHING_SO'] > (3 * np.mean(train1['TEAM_PITCHING_SO']))]))
print(len(train1[train1['TEAM_FIELDING_E'] > (3 * np.mean(train1['TEAM_FIELDING_E']))]))

# Now let's check extreme outliers for var:
print('Number of outliers greater than 3 x mean for...')
for v in traindf.columns:
    print(str(v) + ': ' + str(len(train1[train1[str(v)] > (3 * np.mean(train1[str(v)]))])))
# Next, let's address the outliers for some specific predictors.

#The first extreme outlier is in the TEAM_PITCHING_SO, the max value clearly has an issue. The 75th percentile is 968 while the max value is 19,278. 
#I'm going to cap this max value by imputing value with new max value of 1000.
# if(train1[train1['TEAM_PITCHING_SO']] > (3 * np.mean(train1['TEAM_PITCHING_SO']))):
#     train1['TEAM_PITCHING_SO_1'] = (3 * np.mean(train1['TEAM_PITCHING_SO']))
#     train1['TEAM_PITCHING_IMP'] = 1

#     m = np.median(train.TEAM_PITCHING_SO[train.TEAM_PITCHING_SO > 0])
# train1=train.replace({'team_batting_hbp': {0: m}}) 
#print(3 * np.mean(train1['TEAM_PITCHING_SO']))

# train1['TEAM_PITCHING_SO'].min()
# sort_teampitchingso = train1.sort_values('TEAM_PITCHING_SO')
# print(sort_teampitchingso.tail(n=3))

print('TRAIN DATA\n------------')

# TEAM_PITCHING_SO trim extreme high outliers
a = np.array(train1['TEAM_PITCHING_SO'].values.tolist())
train1['TEAM_PITCHING_SO'] = np.where(a > (3 * np.mean(train1['TEAM_PITCHING_SO'])), round(3 * np.mean(train1['TEAM_PITCHING_SO']), 0), a).tolist()
print('New TEAM_PITCHING_SO max values is: ' + str(train1['TEAM_PITCHING_SO'].max()))

# TEAM_FIELDING_E trim extreme high outliers
a = np.array(train1['TEAM_FIELDING_E'].values.tolist())
train1['TEAM_FIELDING_E'] = np.where(a > (3 * np.mean(train1['TEAM_FIELDING_E'])), round(3 * np.mean(train1['TEAM_FIELDING_E']), 0), a).tolist()
print('New TEAM_FIELDING_E max values is: ' + str(train1['TEAM_FIELDING_E'].max()))

# TEAM_FIELDING_DP zero values - replace with the mean
a = np.array(train1['TEAM_FIELDING_DP'].values.tolist())
train1['TEAM_FIELDING_DP'] = np.where(a == 0, np.mean(train1['TEAM_FIELDING_DP']), a).tolist()
print('New TEAM_FIELDING_DP min value is: ' + str(train1['TEAM_FIELDING_DP'].min()))

# TEAM_PITCHING_BB zero values - replace with the mean
a = np.array(train1['TEAM_PITCHING_BB'].values.tolist())
train1['TEAM_PITCHING_BB'] = np.where(a == 0, np.mean(train1['TEAM_PITCHING_BB']), a).tolist()
print('New TEAM_PITCHING_BB min value is: ' + str(train1['TEAM_PITCHING_BB'].min()))

# TEAM_PITCHING_HR zero values - replace with the mean
a = np.array(train1['TEAM_PITCHING_HR'].values.tolist())
train1['TEAM_PITCHING_HR'] = np.where(a == 0, np.mean(train1['TEAM_PITCHING_HR']), a).tolist()
print('New TEAM_PITCHING_HR min value is: ' + str(train1['TEAM_PITCHING_HR'].min()))

# TEAM_BASERUN_SB zero values - replace with the mean
a = np.array(train1['TEAM_BASERUN_SB'].values.tolist())
train1['TEAM_BASERUN_SB'] = np.where(a == 0, np.mean(train1['TEAM_BASERUN_SB']), a).tolist()
print('New TEAM_BASERUN_SB min value is: ' + str(train1['TEAM_BASERUN_SB'].min()))

# TEAM_BATTING_3B zero values - replace with the mean
a = np.array(train1['TEAM_BATTING_3B'].values.tolist())
train1['TEAM_BATTING_3B'] = np.where(a == 0, np.mean(train1['TEAM_BATTING_3B']), a).tolist()
print('New TEAM_BATTING_3B min value is: ' + str(train1['TEAM_BATTING_3B'].min()))

# TEAM_BATTING_BB zero values - replace with the mean
a = np.array(train1['TEAM_BATTING_BB'].values.tolist())
train1['TEAM_BATTING_BB'] = np.where(a == 0, np.mean(train1['TEAM_BATTING_BB']), a).tolist()
print('New TEAM_BATTING_BB min value is: ' + str(train1['TEAM_BATTING_BB'].min()))

# TEAM_BATTING_SO zero values - replace with the mean
a = np.array(train1['TEAM_BATTING_SO'].values.tolist())
train1['TEAM_BATTING_SO'] = np.where(a == 0, np.mean(train1['TEAM_BATTING_SO']), a).tolist()
print('New TEAM_BATTING_SO min value is: ' + str(train1['TEAM_BATTING_SO'].min()))

# TEAM_BATTING_HR zero values - replace with the mean
a = np.array(train1['TEAM_BATTING_HR'].values.tolist())
train1['TEAM_BATTING_HR'] = np.where(a == 0, np.mean(train1['TEAM_BATTING_HR']), a).tolist()
print('New TEAM_BATTING_HR min value is: ' + str(train1['TEAM_BATTING_HR'].min()))

# TEAM_PITCHING_SO zero values - replace with the mean
a = np.array(train1['TEAM_PITCHING_SO'].values.tolist())
train1['TEAM_PITCHING_SO'] = np.where(a == 0, np.mean(train1['TEAM_PITCHING_SO']), a).tolist()
print('New TEAM_PITCHING_SO min value is: ' + str(train1['TEAM_PITCHING_SO'].min()))

'''
DO THE SAME FOR THE TEST SET
'''
print('\nTEST DATA\n------------')
# 1. TEAM_PITCHING_SO extreme high outliers
a = np.array(test1['TEAM_PITCHING_SO'].values.tolist())
test1['TEAM_PITCHING_SO'] = np.where(a > (3 * np.mean(test1['TEAM_PITCHING_SO'])), round(3 * np.mean(test1['TEAM_PITCHING_SO']), 0), a).tolist()
print('New TEAM_PITCHING_SO max values is: ' + str(test1['TEAM_PITCHING_SO'].max()))

# TEAM_FIELDING_E trim extreme high outliers
a = np.array(test1['TEAM_FIELDING_E'].values.tolist())
test1['TEAM_FIELDING_E'] = np.where(a > (3 * np.mean(test1['TEAM_FIELDING_E'])), round(3 * np.mean(test1['TEAM_FIELDING_E']), 0), a).tolist()
print('New TEAM_FIELDING_E max values is: ' + str(test1['TEAM_FIELDING_E'].max()))

# TEAM_FIELDING_DP zero values - replace with the mean
a = np.array(test1['TEAM_FIELDING_DP'].values.tolist())
test1['TEAM_FIELDING_DP'] = np.where(a == 0, np.mean(test1['TEAM_FIELDING_DP']), a).tolist()
print('New TEAM_FIELDING_DP min value is: ' + str(test1['TEAM_FIELDING_DP'].min()))

# 2 TEAM_PITCHING_BB zero values - replace with the mean
a = np.array(test1['TEAM_PITCHING_BB'].values.tolist())
test1['TEAM_PITCHING_BB'] = np.where(a == 0, np.mean(test1['TEAM_PITCHING_BB']), a).tolist()
print('New TEAM_PITCHING_BB min value is: ' + str(test1['TEAM_PITCHING_BB'].min()))

# 3 TEAM_PITCHING_HR zero values - replace with the mean
a = np.array(test1['TEAM_PITCHING_HR'].values.tolist())
test1['TEAM_PITCHING_HR'] = np.where(a == 0, np.mean(test1['TEAM_PITCHING_HR']), a).tolist()
print('New TEAM_PITCHING_HR min value is: ' + str(test1['TEAM_PITCHING_HR'].min()))

# 3 TEAM_BASERUN_SB zero values - replace with the mean
a = np.array(test1['TEAM_BASERUN_SB'].values.tolist())
test1['TEAM_BASERUN_SB'] = np.where(a == 0, np.mean(test1['TEAM_BASERUN_SB']), a).tolist()
print('New TEAM_BASERUN_SB min value is: ' + str(test1['TEAM_BASERUN_SB'].min()))

# TEAM_BATTING_3B zero values - replace with the mean
a = np.array(test1['TEAM_BATTING_3B'].values.tolist())
test1['TEAM_BATTING_3B'] = np.where(a == 0, np.mean(test1['TEAM_BATTING_3B']), a).tolist()
print('New TEAM_BATTING_3B min value is: ' + str(test1['TEAM_BATTING_3B'].min()))

# TEAM_BATTING_BB zero values - replace with the mean
a = np.array(test1['TEAM_BATTING_BB'].values.tolist())
test1['TEAM_BATTING_BB'] = np.where(a == 0, np.mean(test1['TEAM_BATTING_BB']), a).tolist()
print('New TEAM_BATTING_BB min value is: ' + str(test1['TEAM_BATTING_BB'].min()))

# TEAM_BATTING_SO zero values - replace with the mean
a = np.array(test1['TEAM_BATTING_SO'].values.tolist())
test1['TEAM_BATTING_SO'] = np.where(a == 0, np.mean(test1['TEAM_BATTING_SO']), a).tolist()
print('New TEAM_BATTING_SO min value is: ' + str(test1['TEAM_BATTING_SO'].min()))

# TEAM_BATTING_HR zero values - replace with the mean
a = np.array(test1['TEAM_BATTING_HR'].values.tolist())
test1['TEAM_BATTING_HR'] = np.where(a == 0, np.mean(test1['TEAM_BATTING_HR']), a).tolist()
print('New TEAM_BATTING_HR min value is: ' + str(test1['TEAM_BATTING_HR'].min()))

# TEAM_PITCHING_SO zero values - replace with the mean
a = np.array(test1['TEAM_PITCHING_SO'].values.tolist())
test1['TEAM_PITCHING_SO'] = np.where(a == 0, np.mean(test1['TEAM_PITCHING_SO']), a).tolist()
print('New TEAM_PITCHING_SO min value is: ' + str(test1['TEAM_PITCHING_SO'].min()))

print(train1.describe())
# Now, let's add some additional columns to the training data frame
#train1['LOG_TARGET_WINS'] = np.log(train1['TARGET_WINS']) #log response
#train1['SQRT_TARGET_WINS'] = np.sqrt(train1['TARGET_WINS']) #sqrt response

train1['LOG_TEAM_BASERUN_SB'] = np.log(train1['TEAM_BASERUN_SB']) #log baserun
train1['LOG_TEAM_PITCHING_BB'] = np.log(train1['TEAM_PITCHING_BB']) #log walks allowed
train1['LOG_TEAM_PITCHING_H'] = np.log(train1['TEAM_PITCHING_H']) #log hits allowed
train1['LOG_TEAM_PITCHING_HR'] = np.log(train1['TEAM_PITCHING_HR']) #log home runs allowed
#train1['TEAM_BATTING_1B'] = (train1['TEAM_BATTING_H'] - train1['TEAM_BATTING_HR'] - train1['TEAM_BATTING_2B'] - train1['TEAM_BATTING_3B']) #1B hits only

# train1['LOG_TEAM_BATTING_1B'] = np.log(train1['TEAM_BATTING_1B'])
# train1['LOG_TEAM_BATTING_2B'] = np.log(train1['TEAM_BATTING_2B'])
# train1['LOG_TEAM_BATTING_3B'] = np.log(train1['TEAM_BATTING_3B'])
# train1['LOG_TEAM_BATTING_HR'] = np.log(train1['TEAM_BATTING_HR'])
# train1['LOG_TEAM_BATTING_BB'] = np.log(train1['TEAM_BATTING_BB'])
# Also add same predictor variable columns to the test data frame, do not add any of the new response vars (TARGET_WINS not included in test)
#test1['LOG_TEAM_BATTING_H'] = np.log(test1['TEAM_BATTING_H']) #log hitting pred var
#test1['SQRT_TEAM_BATTING_H'] = np.sqrt(test1['TEAM_BATTING_H']) #sqrt hitting pred var
#test1['TEAM_BATTING_1B'] = (test1['TEAM_BATTING_H'] - test1['TEAM_BATTING_HR'] - test1['TEAM_BATTING_2B'] - test1['TEAM_BATTING_3B']) #1B hits only

test1['LOG_TEAM_BASERUN_SB'] = np.log(test1['TEAM_BASERUN_SB']) #log baserun
test1['LOG_TEAM_PITCHING_BB'] = np.log(test1['TEAM_PITCHING_BB']) #log walks allowed
test1['LOG_TEAM_PITCHING_H'] = np.log(test1['TEAM_PITCHING_H']) #log hits allowed
test1['LOG_TEAM_PITCHING_HR'] = np.log(test1['TEAM_PITCHING_HR']) #log hoe runs allowed
# Now let's verify new additions to train1 and test1 data frames
print(train1.info())
print(test1.info())
train1.describe()
# Now let's take another look at the cleaned and transformed dataset
test1.hist(figsize=(20,16))
test1.plot(kind= 'box' , subplots=True, layout=(5,5), sharex=False, sharey=False, figsize=(10,8))
# This can be used to quickly change the variables for testing plots in this cell only
#myresponsevar = 'TARGET_WINS'
#myresponsevar = 'LOG_TARGET_WINS'

# goodhitting = ['TEAM_BATTING_HR', 'TEAM_BATTING_1B', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB', 'TEAM_BATTING_HBP']
# goodother = ['TEAM_PITCHING_SO', 'TEAM_FIELDING_DP', 'TEAM_BASERUN_SB']
# badpitching = ['TEAM_PITCHING_H', 'TEAM_PITCHING_HR', 'TEAM_PITCHING_BB']
# badoffense = ['TEAM_BATTING_SO', 'TEAM_BASERUN_CS']
# baddefense = ['TEAM_FIELDING_E']

# This was used for first Kaggle submissions 1-3
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_HR', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB',
#               'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO', 'TEAM_BATTING_SO', 'TEAM_BASERUN_CS', 'TEAM_FIELDING_E', 'TEAM_PITCHING_BB',
#               'TEAM_PITCHING_H', 'TEAM_PITCHING_HR']

# Use all vars - predictions4
# myresponsevar = 'TARGET_WINS'
# mypredvars = all

# New Test - predictions5
# myresponsevar = 'LOG_TARGET_WINS'
# mypredvars = ['TEAM_BATTING_1B', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_HR', 'TEAM_BATTING_BB', 
#               'TEAM_PITCHING_BB', 'TEAM_FIELDING_E', 'TEAM_PITCHING_H']
#preds5 tried but not used
#mypredvars = ['TEAM_BATTING_H', 'TEAM_BATTING_1B', 'TEAM_BATTING_BB', 'TEAM_PITCHING_SO', 'TEAM_FIELDING_E', 'TEAM_PITCHING_H']
# mypredvars = ['TEAM_BATTING_HR', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB',
#               'TEAM_PITCHING_SO', 'TEAM_FIELDING_E', 'TEAM_PITCHING_BB', 'TEAM_PITCHING_H']

# Predictions 6
# myresponsevar = 'SQRT_TARGET_WINS'
# mypredvars = ['TEAM_BATTING_H', 'TEAM_BATTING_HR', 'TEAM_BATTING_3B', 'TEAM_BATTING_SO', 'TEAM_BATTING_HBP',
#               'TEAM_BASERUN_SB', 'TEAM_PITCHING_SO', 'TEAM_PITCHING_BB', 'TEAM_PITCHING_H', 'TEAM_FIELDING_DP', 'TEAM_FIELDING_E']

# Predictions 7
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_HR', 'TEAM_BATTING_1B', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_HBP',
#               'TEAM_PITCHING_SO', 'TEAM_FIELDING_DP', 'TEAM_BASERUN_SB', 'TEAM_PITCHING_H', 'TEAM_PITCHING_BB']

# Predictions 8
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_HR', 'TEAM_BATTING_1B', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB', 'TEAM_BATTING_HBP',
#               'TEAM_PITCHING_SO', 'TEAM_BASERUN_SB'] #all good items

# Predictions 9
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_H', 'TEAM_BATTING_BB', 'TEAM_BATTING_SO', 'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO',
#               'TEAM_FIELDING_E', 'TEAM_PITCHING_BB', 'TEAM_PITCHING_H', 'TEAM_PITCHING_HR']

# Predictions 10
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_HR', 'TEAM_BATTING_1B', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B',
#               'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO', 'TEAM_BATTING_SO', 'TEAM_FIELDING_E', 'TEAM_PITCHING_BB',
#               'TEAM_PITCHING_H', 'TEAM_PITCHING_HR']

# Predictions 11 / 12
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_HR', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB',
#               'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO', 'TEAM_BATTING_SO', 'TEAM_BASERUN_CS', 'TEAM_FIELDING_E', 'TEAM_PITCHING_BB',
#               'TEAM_PITCHING_H', 'TEAM_PITCHING_HR']

# Predictions 13
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_H', 'TEAM_BATTING_HR', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB', 'TEAM_BASERUN_SB',
#               'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO', 'TEAM_BATTING_SO', 'TEAM_FIELDING_E', 'TEAM_PITCHING_BB',
#               'TEAM_PITCHING_H', 'TEAM_PITCHING_HR']

# Predictions 14
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_HR', 'TEAM_BATTING_H', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB', 'LOG_TEAM_BASERUN_SB',
#               'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO', 'TEAM_BATTING_SO', 'TEAM_FIELDING_E', 'LOG_TEAM_PITCHING_BB',
#               'LOG_TEAM_PITCHING_H', 'LOG_TEAM_PITCHING_HR']

# Predictions 15
# myresponsevar = 'TARGET_WINS'
# mypredvars = ['TEAM_BATTING_H', 'TEAM_BATTING_BB', 'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO', 'TEAM_BATTING_SO',
#               'TEAM_FIELDING_E', 'TEAM_PITCHING_BB', 'TEAM_PITCHING_H', 'TEAM_PITCHING_HR']


# Other comparison tests used - remove before submitting
#mypredvars = ['TEAM_BATTING_H', 'LOG_TEAM_BATTING_H', 'SQRT_TEAM_BATTING_H']
# mypredvars = ['TEAM_BATTING_H', 'TEAM_BATTING_HR', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB', 'TEAM_BASERUN_SB',
#               'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO', 'TEAM_BATTING_SO', 'TEAM_BASERUN_CS', 'TEAM_FIELDING_E', 'TEAM_PITCHING_BB',
#               'TEAM_PITCHING_H', 'TEAM_PITCHING_HR']

#submit 2 trials
myresponsevar = 'TARGET_WINS'
mypredvars = ['TEAM_BATTING_HR', 'TEAM_BATTING_2B', 'TEAM_BATTING_3B', 'TEAM_BATTING_BB',
              'TEAM_FIELDING_DP', 'TEAM_PITCHING_SO', 'TEAM_BATTING_SO', 'TEAM_BASERUN_CS', 'TEAM_FIELDING_E', 'LOG_TEAM_PITCHING_BB',
              'LOG_TEAM_PITCHING_H', 'LOG_TEAM_PITCHING_HR']


predstr = ''
for i in mypredvars:
    predstr = predstr + ' + ' + str(i)

# Now build the string for the linear model
modelstr = myresponsevar + predstr
modelstr = modelstr.replace('+', '~', 1) #replace only the first instance
print(modelstr) # this is the string replesenting the MLR model for use below
# Section 2 Results:

# test1.hist(figsize=(20,16))
# test1.plot(kind= 'box' , subplots=True, layout=(5,5), sharex=False, sharey=False, figsize=(10,8))
#Just in case there are some variables that are highly correlated look at some scatter plots
# # option 1
# for i in mypredvars:
#     train1.plot.scatter(y=myresponsevar, x=i, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, 
#                         linewidths=None, verts=None, edgecolors=None )

# option 2 - shown with regr line
# for i in mypredvars:
#     sm.graphics.plot_partregress(myresponsevar, i, [], data = train1, obs_labels = False)
#From sample code
#Take a look at a possible model
print(modelstr + str('\n'))

#previous tests
#regres1 = smf.ols('LOG_TARGET_WINS ~ TEAM_BATTING_H', data = train1)
#regres1 = smf.ols('TARGET_WINS ~ TEAM_BATTING_H + TEAM_BATTING_BB', data = train1)
#regres1 = smf.ols('TARGET_WINS ~ TEAM_BATTING_2B + TEAM_BATTING_3B + TEAM_BATTING_HR + TEAM_BATTING_BB + TEAM_BASERUN_SB', data = train1)
#regres1 = smf.ols('LOG_TARGET_WINS ~ TEAM_BATTING_H + TEAM_BATTING_BB + TEAM_BASERUN_SB + TEAM_FIELDING_DP', data = train1)
#regres1 = smf.ols('TARGET_WINS ~ TEAM_BATTING_HR + TEAM_BATTING_HBP + TEAM_BATTING_BB + TEAM_BASERUN_SB', data = train1)

'''
Function: fit model
Takes a model string & dataframe, runs the linear model calc and returns the fit result
'''
def fit_model(fullmodelstr, dataset):
    return smf.ols(fullmodelstr, data = dataset).fit()

#     regres1 = smf.ols(...
#     fit1 = regres1.fit() 
#     return fit1


# Run the model
result1 = fit_model(modelstr, train1)
#print(dir(result1)) #use this if you want to see teh available attributes
print (result1.summary())
print (result1.conf_int())
#dir(result1) lists other print options

# # Now run pred model on the test1 set
# predictions = result1.predict(test1)
# print(predictions)
'''
This code is for performing a step-wise forward selection on the defined model. 
Input: modelstr
Output: string with best AIC/BIC scores
Descr: Uses the sample code for smf.ols fit to fit the model and obtain the summary
'''

def stepwise_fwd_selection(fullmodelstr, dataset):
    #first sep string into components and save the response
    respons = fullmodelstr[:fullmodelstr.find(' ~')]
    allpredictors = fullmodelstr[fullmodelstr.find(' ~')+3:]
    print(allpredictors)
    
    #ok, now split preds based on number of '+' char found in allpredictors string + 1
    num_of_preds = 0
    if(fullmodelstr.count('+') < 1):
        num_of_preds = 1
    else:
        num_of_preds = fullmodelstr.count('+') + 1
    
    #separate out preditors from the string
    preds = []
    tmpstr = allpredictors
    for i in list(range(num_of_preds)):
        #print('tmpstr is: ' + str(tmpstr))
        if(tmpstr.count('+') < 1):
            preds.append(tmpstr)
        else:
            preds.append(tmpstr[:tmpstr.find(' +')])
        tmpstr = tmpstr[tmpstr.find(' +')+3:]
    
    #ok, now we need to run the linear model for each iteration starting with the first and sequentially adding all predictors
    # fit the partial model for each step starting
    adjR2 = []
    aic = []
    bic = []
    
    steppreds = ''
    for i in preds:
        stepmodelstr = str(respons) + ' ~ ' + steppreds + str(i)
        stepresult = fit_model(stepmodelstr, train1)
        steppreds = steppreds + str(i) + ' + '
        adjR2.append(stepresult.rsquared_adj)
        aic.append(stepresult.aic)
        bic.append(stepresult.bic)
        #print (stepresult.summary())
        #print('+++++++++++++++++++++++++++++++++++++++++\n\n')
    
    print(adjR2)
#     print('AIC values: ' + str(aic))
#     print('BIC values: ' + str(bic))
#     return respons, preds
    return np.argmax(adjR2), np.argmin(aic), np.argmin(bic)

# Next try it out
output1, output2, output3 = stepwise_fwd_selection(modelstr, None)
print(output1)
print(output2)
print(output3)



# fitted values (need a constant term for intercept)
model_fitted_y = result1.fittedvalues

# model residuals
model_residuals = result1.resid

# normalized residuals
model_norm_residuals = result1.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)

# leverage, from statsmodels internals
model_leverage = result1.get_influence().hat_matrix_diag

# cook's distance, from statsmodels internals
model_cooks = result1.get_influence().cooks_distance[0]
# Print each plot output

#1 Residual Plot
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'TEAM_BATTING_H', data = train1, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i, 
                               xy=(model_fitted_y[i], 
                                   model_residuals[i]));
#2 Q-Q Plot
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   model_norm_residuals[i]));
#3 Scale-Location Plot

plot_lm_3 = plt.figure(3)
plot_lm_3.set_figheight(8)
plot_lm_3.set_figwidth(12)

plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

# annotations
abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

for i in abs_norm_resid_top_3:
    plot_lm_3.axes[0].annotate(i, 
                               xy=(model_fitted_y[i], 
                                   model_norm_residuals_abs_sqrt[i]));
#4 Leverage Plot
plot_lm_4 = plt.figure(4)
plot_lm_4.set_figheight(8)
plot_lm_4.set_figwidth(12)

plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
sns.regplot(model_leverage, model_norm_residuals, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_4.axes[0].set_xlim(0, 0.20)
plot_lm_4.axes[0].set_ylim(-3, 5)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

# annotations
leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

for i in leverage_top_3:
    plot_lm_4.axes[0].annotate(i, 
                               xy=(model_leverage[i], 
                                   model_norm_residuals[i]))
    
# shenanigans for cook's distance contours
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(result1.params) # number of model parameters

graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 
      'Cook\'s distance') # 0.5 line
graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50)) # 1 line
plt.legend(loc='upper right');
print(modelstr)

#Take a look at the final model
regres_final = smf.ols(modelstr , data = train1)  
result_final = regres_final.fit()  
print (result_final.summary())
print (result_final.conf_int())
#dir(result) lists other print options

final_predictions = result_final.predict(test1)
print(final_predictions)
# TODO: Write out model equation in explicit form
# Note: Just disply it here no code to run - copy to report.
modelstr
# From provided sample code

#Convert the array predictions to a data frame then merge with the index for the test data to create your file
d = {'P_TARGET_WINS': round(final_predictions, 1)}
#df1 = testdf[['INDEX']]
df1 = pd.to_numeric(testdf['INDEX'], downcast = 'integer').to_frame()

df2 = pd.DataFrame(data = d)
output_file = pd.concat([df1,df2], axis = 1, join_axes = [df1.index])

#Submit your file as csv using the following code to save on your computer
#output_file.to_csv('output/predictions.csv', index = False)
#output_file.to_csv('output/andrew_knight_predictions1.csv') # This file was my first submission to Kaggle on 20180710-0858 - failed
#output_file.to_csv('output/andrew_knight_predictions2.csv', index = False) # Second attempt on 20180710-0928 still has formatting issues - failed
#output_file.to_csv('output/andrew_knight_predictions3.csv', index = False) # Third attempt on 20180710 - THIS WORKED FINALLY!

#new test
#output_file.to_csv('output/andrew_knight_predictions4.csv', index = False) # First attempt on 20180711-2155
#output_file.to_csv('output/andrew_knight_predictions5.csv', index = False) # Using LOG_TARGET_WINS on 20180711-2315
#output_file.to_csv('output/andrew_knight_predictions6.csv', index = False) # Using SQRT_TARGET_WINS on 20180711-2332
#output_file.to_csv('output/andrew_knight_predictions7.csv', index = False) # Preds7 subm on 20180712-0048
#output_file.to_csv('output/andrew_knight_predictions8.csv', index = False) # Preds8 subm on 20180712-0053
#output_file.to_csv('output/andrew_knight_predictions9.csv', index = False) # Preds9 subm on 20180712-0152 and -2256
#output_file.to_csv('output/andrew_knight_predictions10.csv', index = False) # Pred10 subm on 20180712-2301
#output_file.to_csv('output/andrew_knight_predictions11.csv', index = False) # Pred11 subm on 20180712-2311
#output_file.to_csv('output/andrew_knight_predictions12.csv', index = False) # Pred12 subm on 20180712-2318
#output_file.to_csv('output/andrew_knight_predictions13.csv', index = False) # Pred13 subm on 20180712-2326
#output_file.to_csv('output/andrew_knight_predictions14.csv', index = False) # Pred14
output_file.to_csv('output/andrew_knight_predictions15.csv', index = False) # Pred15


# sanity check section
