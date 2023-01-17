import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from kaggle.competitions import twosigmanews #TwoSigmaAPI
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

import random
import math
from datetime import timedelta
#Crete our environment
env = twosigmanews.make_env()
(market_label_df, news_label_df) = env.get_training_data()
def kFoldIndices(df, folds):
    """
    This function creates a list of training and validation indices for our K-Folds based on splitting the data on year-months
    """
    #Get a set of all the available year months
    df.loc[:, 'yearmonth'] = df.loc[:, 'time'].dt.strftime('%Y-%m')
    yearmonths = set(df['yearmonth'].unique())
    #Work out the fold size from the total number of folds available and the number of folds we want to use
    #tk - Need to add error handling here for number of folds greather than available even though this is unlikely to occur
    foldSize = int(round(len(yearmonths)/folds, 0))
    #Get the training yearmonths
    foldYearMonths = []
    for i in range(folds):
        #Final fold
        if i == (folds-1):
            #Need to expand to a single list of yearmonths as otherwise we have a list of lists
            foldYearMonthsExpanded = [j for i in foldYearMonths for j in i]
            #Calculate the yearmonths for our final fold
            fold = list(yearmonths-set(foldYearMonthsExpanded))
            foldYearMonths.append(fold)
        else:
            #Take a random sample from the remaining yearmonths 
            if len(foldYearMonths) > 0:
                #Need to expand to a single list of yearmonths as otherwise we have a list of lists
                foldYearMonthsExpanded = [j for i in foldYearMonths for j in i]
                options = yearmonths - set(foldYearMonthsExpanded)
                fold = random.sample(options, foldSize)
                foldYearMonths.append(fold)
            #If it is our first fold just take a random sample of the yearmonths
            else: 
                fold = random.sample(yearmonths, foldSize)
                foldYearMonths.append(fold)
    
    #Now that we have our folds we need to split them into K combinations of training and validiation sets 
    kFolds = []
    
    for i in range(folds):
        #Set our training months to be all except for the ith fold
        trainMonths = foldYearMonths[:i] + foldYearMonths[i+1:]
        #Expand into single list from list of lists
        trainMonths = [j for i in trainMonths for j in i]
        #Set our valid months to be the ith fold
        validMonths = foldYearMonths[i]
        #Set our indices from the months
        trainIndices = df[df['yearmonth'].isin(trainMonths)].index
        validIndices = df[df['yearmonth'].isin(validMonths)].index
        #Create our tuple and append 
        kFold = (trainMonths, validMonths, trainIndices, validIndices)
        kFolds.append(kFold)
    
    return kFolds
def purgeTrain(train, valid, yearmonths, validMonths):
    """
    This function purges all training data within 10 days of any validation data. This prevents leakage between
    the two data sets. 
    """
    # Get a list of our yearmonths 
    ymSort = list(yearmonths)
    # Sort in ascending order
    ymSort.sort()
    # Make a copy of our training data
    trn = train.copy(deep=True)
    # Iterate over our validation months
    for month in validMonths:
        # Get the position of the month in our list of year months
        index = ymSort.index(month)
        # Get the position of the month before it and after it i.e under and over
        under = index-1
        over = index+1
        
        # If no months are before or after this month set it that side to false
        if under < 0:
            under = False
        if over > (len(ymSort) - 1):
            over = False
        
        # Look for the first training month before our validition month, this handled multiple validation months joined together
        while True:
            if ymSort[under] not in validMonths:
                break
            if (under - 1) < 0:
                under = False
                break
            else:
                under -= 1
            
        # Look for the first training month after our validition month, this handled multiple validation months joined together
        while True:
            if ymSort[over] not in validMonths:
                break
            if (over + 1) > (len(ymSort) - 1):
                over = False
                break
            else:
                over += 1
        
        # If there is a training month before our validation month, remove all training observations from the 20th onwards of that month
        if under != False:
            monthUnder = ymSort[under]
            test = trn[trn['yearmonth'] == monthUnder]
            test = test[test['time'].dt.day > 20]
            trn = trn.drop(test.index)          
            
        # If there is a training month after our validation month, remove all training observations up to the 10th of that month
        if over != False:
            monthOver = ymSort[over]
            test = trn[trn['yearmonth'] == monthOver]
            test = test[test['time'].dt.day < 10]
            trn = trn.drop(test.index)
    
    # Return our indices to purge from our training sets
    return trn.index
# Create our KFolds
kFolds = kFoldIndices(market_label_df, 10)

# Get the indices and year months for our first fold
trainMonths = kFolds[0][0]
trainIndices = kFolds[0][2]
validMonths = kFolds[0][1]
validIndices = kFolds[0][3]

# Create our training and validation sets
train = market_label_df.loc[trainIndices, :]
valid = market_label_df.loc[validIndices, :]
print (train.shape, valid.shape)

# Work out the indices we need to purge to ensure no data leakage
trainIndicesPurged = purgeTrain(train, valid, trainMonths+validMonths, validMonths)

# Purge our training set to ensure no data leakage
trainPurged = train.loc[trainIndicesPurged, :]
print (trainIndicesPurged.shape)
print (trainPurged.shape)
# Plot our fold
plt.figure(figsize=(40,10))
plt.plot_date(trainPurged.groupby('time').count().index, trainPurged.groupby('time').count()['close'], markersize=3)
plt.plot_date(valid.groupby('time').count().index, valid.groupby('time').count()['close'], markersize=3)
# Initialise our list for holding our scores
scores = []

# Create a simple decision tree model
decisionTree = DecisionTreeClassifier(max_depth=10)

# Iterate over our folds we created earlier
for kFold in kFolds:
    trainMonths = kFold[0]
    validMonths = kFold[1]
    trainIndices = kFold[2]
    validIndices = kFold[3]
    
    train = market_label_df.loc[trainIndices, :]
    valid = market_label_df.loc[validIndices, :]
    
    trainIndicesPurged = purgeTrain(train, valid, trainMonths+validMonths, validMonths)
    
    valid = valid.dropna()
    train = market_label_df.loc[trainIndicesPurged, :].dropna()
    
    X_train = train.drop(['returnsOpenNextMktres10', 'universe', 'yearmonth', 'time', 'assetName', 'assetCode'], axis=1).values
    X_valid = valid.drop(['returnsOpenNextMktres10', 'universe', 'yearmonth', 'time', 'assetName', 'assetCode'], axis=1).values
    y_train = train.loc[:, 'returnsOpenNextMktres10']
    y_train[y_train < 0] = 0
    y_train[y_train > 0] = 1
    y_train = y_train.values.squeeze()
    y_valid = valid.loc[:, 'returnsOpenNextMktres10']
    y_valid[y_valid < 0] = 0
    y_valid[y_valid > 0] = 1
    y_valid = y_valid.values.squeeze()
    
    print ('got kfold')
    estimatorCopy = clone(decisionTree)
    print ('cloned')
    estimatorCopy.fit(X_train, y_train)
    print ('fit')
    score = estimatorCopy.score(X_valid, y_valid)
    print ('scored')
    scores.append(score)
print (scores)