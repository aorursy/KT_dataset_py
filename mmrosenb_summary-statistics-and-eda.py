#imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import display, HTML, Markdown, display #display settings

import warnings #for filtering warnings



#constants

%matplotlib inline

sns.set_style("dark")

#to ignore warnings in output

warnings.filterwarnings('ignore')

#global information settings

sigLev = 2 #three significant digits

percentMul = 100 #for percentage multiplication

figWidth = figHeight = 8
#load in dataset

trainFrame = pd.read_csv("../input/train.csv")
lossFigure = plt.figure(figsize = (figWidth,figHeight))

lossHistogram = plt.hist(trainFrame["loss"])

plt.xlabel("Loss")

plt.ylabel("Count")

plt.title("Distribution of Loss")
trainFrame["logLoss"] = np.log(trainFrame["loss"])

logLossFigure = plt.figure(figsize = (figWidth,figHeight))

logLossHistogram = plt.hist(trainFrame["logLoss"])

plt.xlabel("$\log(Loss)$")

plt.ylabel("Count")

plt.title("Distribution of $\log(Loss)$")
#check number of categorical variables

categoricalColumns = [col for col in trainFrame.columns if "cat" in col]
#get num unique for each

categoricalTrainFrame = trainFrame[categoricalColumns]

uniqueVec = categoricalTrainFrame.apply(lambda x : x.nunique(),axis = 0)

nuniqueMode = uniqueVec.mode()

#then plot the distribution of number of unique levels

categoricalUniqueFigure = plt.figure(figsize = (figWidth,figHeight))

categoricalNUniqueHist = plt.hist(uniqueVec)

plt.xlabel("Number of Unique Levels")

plt.ylabel("Count")

plt.title("Distribution of Number of Unique Levels\nGiven Categorical Variable")
#make mapper functions

def integerizeCol(catColumn):

    #helper for integerizing a categorical column

    levels = catColumn.unique()

    counter = 0

    mapDict = {} #will add to this

    for lev in levels:

        mapDict[lev] = counter

        counter += 1

    #then integerize the column

    intCol = catColumn.map(mapDict)

    return intCol



def propMode(catColumn):

    #helper that finds the proportion of a given column is the mode

    #integerize it

    intCatColumn = integerizeCol(catColumn)

    #then get the mode

    modeOfCategory = int(intCatColumn.mode())

    #then get proportion

    numMode = intCatColumn[intCatColumn == modeOfCategory].shape[0]

    propMode = float(numMode) / intCatColumn.shape[0]

    return propMode

#then apply over categorical columns

propModeVec = categoricalTrainFrame.apply(propMode,axis = 0)
#then plot

givenFigure = plt.figure(figsize = (figWidth,figHeight))

plt.hist(propModeVec)

plt.xlabel("Proportion Mode")

plt.ylabel("Count")

plt.title("Distribution of\nProportion Mode Across Categorical Variables")
categoricalCutoff = 100 #won't consider categorical variables with fewer than

#this amount of features

propModeBelowCutoff = (

    propModeVec[(1 - propModeVec) * trainFrame.shape[0] < categoricalCutoff])

catToRemove = list(propModeBelowCutoff.index)

categoricalColumns = [catCol for catCol in categoricalColumns if

                        catCol not in catToRemove]
#get continuous variables

continuousColumns = [col for col in trainFrame.columns if "cont" in col]
#get standard deviations of continuous columns

continuousTrainFrame = trainFrame[continuousColumns]

continuousVarVec = continuousTrainFrame.apply(lambda x: x.std(),axis = 0)

display(continuousVarVec)
def numNull(colVec):

    #helper that gets the number of null values in a given columns

    nullValColVec = colVec[colVec.isnull()]

    return nullValColVec.shape[0]

numMissingVec = trainFrame.apply(numNull,axis = 0)

anyMissing = (numMissingVec.sum() > 0)
#reencode our categorical variables

intTrainFrame = trainFrame

intTrainFrame = intTrainFrame.loc[:,

                                  ~trainFrame.columns.isin(categoricalColumns)]

for catCol in categoricalColumns:

    intTrainFrame["int_" + catCol] = integerizeCol(trainFrame[catCol])
#check for correlations

trainCorFrame = intTrainFrame.corr()

#clear up irrelevant variable

trainCorFrame = trainCorFrame.drop(["id","loss"],axis=0)

trainCorFrame = trainCorFrame.drop(["id","loss"],axis=1)

#then get relevant vector

logLossCorVec = trainCorFrame.loc["logLoss",:]

logLossCorVec = logLossCorVec.drop("logLoss",axis = 0)

logLossAbsCorVec = abs(logLossCorVec)

#then plot it

givenFigure = plt.figure(figsize = (figWidth,figHeight))

plt.hist(logLossAbsCorVec)

plt.ylabel("Count")

plt.xlabel("Absolute Correlation with $\log(Loss)$")

plt.title("Distribution of\nAbsolute Correlation (Pearson) with $\log(Loss)$")
#order absolute correlation vector

logLossAbsCorVec = logLossAbsCorVec.sort_values(ascending = False)

#get top 5 features

numFeaturesConsidered = 5

topFiveFeatures = list(logLossAbsCorVec.index)[0:numFeaturesConsidered]

display(topFiveFeatures)
#clean up feature names for full train frame

def plotCategory(catName):

    #helper for plotting our categories

    figure = plt.figure(figsize = (figWidth,figHeight))

    newBoxplot = sns.boxplot(x = catName,y = "logLoss",data = trainFrame)

    #then set labels

    sns.plt.xlabel(catName)

    sns.plt.ylabel("$\log(Loss)$")

    sns.plt.title("$\log(Loss)$ on " + catName)

topFiveFilteredFeatures = [feat[len("int_"):] for feat in topFiveFeatures]

for category in topFiveFilteredFeatures:

    plotCategory(category)