#init



%matplotlib inline



import math

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from mpl_toolkits.mplot3d import Axes3D

import scipy

from scipy import stats

import numpy as np

from numpy import linalg as LA

from sklearn import preprocessing 

import random

import time



scaler = preprocessing.MinMaxScaler()
def display(df):

    

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print(df)
# DATA SELECTION

numerical = ["PRICE", "GBA", "LANDAREA", "AYB"]

categorical = ["AC", "BATHRM", "BEDRM", "ROOMS"]               

colnames = numerical + categorical



#flag missing data - the key attributes consi

missing = {'PRICE':[0,''],'AYB':[0],'AC':[0]}



#read data frame

dataFrame = pd.read_csv("../input/residential2/Computer_Assisted_Mass_Appraisal__Residential.csv", na_values = missing, usecols = colnames)



numerical.extend(['Trans_Score', 'Walk_Score', 'School_Score'])



dataFrame['Walk_Score'] = np.random.randint(1, 100, size=len(dataFrame))

dataFrame['School_Score'] =  np.random.randint(1, 100, size=len(dataFrame))

dataFrame['Trans_Score'] =  np.random.randint(1, 100, size=len(dataFrame))



#drop rows with missing values for a simplicity 

dropped = dataFrame.dropna()

dropped = dropped.iloc[0:500000]



dropped = dropped.reset_index(drop = True)

dropped.head()
#Handle numerical attributes

numericalData = dropped[numerical] 



def histograms(data):

    

    ##NUMERICAL ATTRIBUTES DISTRIBUTION

    for ii in range(0, len(data)):



        minimum = min(data.iloc[:, ii])

        

        if data.columns[ii] == "PRICE" or data.columns[ii] == "LANDAREA" or data.columns[ii] == "GBA":



            maximum = max(data.iloc[:, ii]) / 1.5



        else: 



            maximum = max(data.iloc[:, ii])



        data.iloc[:, ii].plot.hist(grid=True, bins=100, rwidth=0.9, color= "#607c8e", range = (minimum, maximum))

        plt.title(data.columns[ii])

        plt.show()

        

numericalData.head()
#numerical data correlogram

plt.style.use('seaborn-colorblind')

sns.pairplot(numericalData, kind="scatter", diag_kind = 'kde', plot_kws = {'alpha': 0.33, 's': 40, 'edgecolor': 'k'}, height = 3)

plt.show()
#categorical data correlogram

plt.style.use('seaborn-colorblind')



sns.pairplot(categoricalData, kind="scatter", diag_kind = 'kde', plot_kws = {'alpha': 0.33, 's': 40, 'edgecolor': 'k'}, height = 6)

plt.show()
#use z-score to get rid of skewed data 

outliers = numericalData[(np.abs(stats.zscore(numericalData)) < 2).all(axis=1)]



#drop possibly non-existring properties - PRICE BELOW 10K

processedNum = numericalData.drop(outliers.loc[outliers['PRICE'] < 10000].index)



#standardize data - scaling is done for future euclidean distance measure 

dataNorm = scaler.fit_transform(processedNum)

dataNorm = pd.DataFrame(dataNorm)



#histograms(processedNum)

dataNorm.head()
#Handle categorical attributes

categoricalData = dropped.loc[processedNum.index, categorical]



#air condition flag - 0 or 1

labels = categoricalData["AC"].astype('category').cat.categories.tolist()

replace_map_comp = {"AC" : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}



#house style category

#labels = categoricalData["STRUCT_D"].astype('category').cat.categories.tolist()

#replace_map_comp = {"STRUCT_D" : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}

categoricalData.replace(replace_map_comp, inplace=True)



#room style category

bins = np.arange(0, 16, 2)

categoricalData['ROOMS_CAT'] = np.digitize(categoricalData.ROOMS, bins)



#concat two preprocessed frames

finalData = pd.concat([processedNum, categoricalData], axis = 1)



#drop houses which number of rooms is less then number of bedrooms or bathrooms

roomsAnomaly = list(finalData[finalData["ROOMS"] < finalData["BEDRM"]].index) 

roomsAnomaly = roomsAnomaly + list(finalData[finalData["ROOMS"] <= finalData["BATHRM"]].index)

roomsAnomaly = roomsAnomaly + list(finalData[finalData["ROOMS"] >= 29].index)

finalData.drop(roomsAnomaly)



finalData = finalData.reset_index(drop=True)

categoricalData = categoricalData.reset_index(drop=True)

numericalData = numericalData.reset_index(drop=True)



numericalData.tail()
randomToView = [270, 130, 372, 126, 170, 444, 314, 122, 34, 48, 107, 178, 265, 306, 339, 319, 395, 385, 18, 495, 258, 96, 431, 479, 237, 247, 334, 261, 391, 245]
#display(dropped.iloc[randomToView])



## PROTOTYPE SELECTION - for now fixed, the frontend of kaggle does not allow user input

prototypes_idxs = list(map(int, input('Enter numbers: ').split()))



prototypes = dataNorm.iloc[[idx for idx in prototypes_idxs]]



#calculate average prototype

averageProt = pd.DataFrame(data= prototypes.sum()/len(prototypes))



#display 

finalData.iloc[[idx for idx in prototypes_idxs]]
#Calculating distance between each observation and prototype in the dataset



dist = scipy.spatial.distance.cdist(dataNorm.iloc[:,0:], prototypes.iloc[:,0:], metric='euclidean')

euclidDist = pd.DataFrame(dist)

euclidDist.columns = list(prototypes.index)



#print(dataNorm)



#find the nearest prototype for each data sample - forming voronoi tesselations

min_values = euclidDist.idxmin(axis=1)

sampleCenters = pd.DataFrame(min_values)



#example

sampleCenters.tail()
#find a standard deviation of each data cloud - it will be used in the further calculations of eMF

grouped = sampleCenters.groupby(0)



std = pd.DataFrame(columns=['sumOfStdSquared'])

stdDevGroup = pd.DataFrame()



for name, group in grouped:

        

    #find indexes of data samples in a data cloud

    groupDF = pd.DataFrame(group).index

        

    #find standard deviation for each attribute 

    stdDevGroup[name] = pd.DataFrame(dataNorm.iloc[groupDF]).std().pow(2)

    std.loc[name] = stdDevGroup[name].sum() 



stdDevGroup = np.transpose(stdDevGroup)

    

#example

stdDevGroup
#Calculating empirical membership function



def calculateEMF(featuresIdxs, weight):   

    

    EMF = pd.DataFrame(columns=[idx for idx in prototypes_idxs])

    

    chosenNorm = dataNorm.loc[:, featuresIdxs]    

    chosenProto = prototypes.loc[:, featuresIdxs]

    

    for index_value, row_data in chosenProto.iterrows():  

        

        #substract each sample by a prototype vector

        dist = chosenNorm.apply(lambda row: row - row_data, axis = 1).pow(2)

        

        dist *= weight

                                    

        #get normed vector space

        normDistVec = dist.apply(lambda row: row.sum(), axis = 1)

        

        #divide normalized vectors by a difference of the scalar product of a cloud and prototype vector

        #which gives a squared sum of standard deviations within a cloud



        EMFstd = std.loc[index_value]['sumOfStdSquared']           ##standard deviation of the current data cloud

        quotient = normDistVec.apply(lambda row: row / EMFstd)



        #finally calculate eMF for each sample for a particular prototype

        EMF.loc[:,index_value] = quotient.apply(lambda row: 1 / (1 + row))

        

    return EMF



#example of empirical membership functions for each sample for each prototype



time1 = time.time()

exampleEMF = calculateEMF([0, 1, 2, 3, 4, 5], 1) 

exampleEMF.tail()



time2 = time.time()

print(time2 - time1)
#display recommendations defined by a threshold 

def displayRecommendations(EMF, threshold):

    

    #find indexes defined by the threshold

    recommendationsIdxs = EMF[(EMF.loc[:,[idx for idx in prototypes_idxs]] > threshold).any(axis=1)].index

    recommended = categoricalData.iloc[recommendationsIdxs]



    catDict = categoricalData.iloc[prototypes_idxs].to_dict(orient = "list")

    finalRecomendIdxs = recommended[recommended[list(catDict)].isin(catDict).all(axis=1)].index



    finalRecomendIdxs = list(set(recommendationsIdxs) - set(prototypes_idxs))



    display(finalData.iloc[recommendationsIdxs])

   

    return len(finalRecomendIdxs) + 1

    

displayRecommendations(exampleEMF, 0.85)
def findAverageRelevant():

    

    print('Enter ID\'s of relevant recommendations: ')    

    relevant_idxs = list(map(int, input().split()))



    relevant = dataNorm.iloc[[idx for idx in prototypes_idxs]]



    #calculate average relevant recommendation

    averageRelevant = relevant.sum()/len(prototypes)

    

    return averageRelevant
#calculate continuoues Membership function per each numerical feature

sequence = np.arange(0, 1, 0.001)



def calcEMFPF(prototypes):

    

    (lenProt, wProt) = prototypes.shape

    membershipPF = np.zeros((wProt, len(sequence), lenProt + 1))



    for i in range(wProt):



        weight = 1



        for j in range(lenProt + 1):

            

            membershipPF[i, :, j] = stdDevGroup.iloc[0, i] /(np.power(weight*(sequence - prototypes.iloc[0, i]), 2)  + stdDevGroup.iloc[0, i])

            weight += 1.5

            

    return membershipPF, lenProt

##Evolving empirical fuzzy rule-based algorithm



threshold = 0.85



averRelevants = []

totalWeights = []



def evolveEMF(numOfIter):

    

    time1 = time.time()

        

    wAv = averageProt.shape[0]

    averageR = findAverageRelevant()

    averRelevants.append(averageR)

    (EMF, _) = calcEMFPF(averageProt.T)

    intersectionPoints = []

            

    for i in range(wAv):     

                    

        proper = (sequence * (max(dataNorm.iloc[:, i]) - min(dataNorm.iloc[:, i])) + min(dataNorm.iloc[:, i]))

        idx = np.argwhere(np.diff(np.sign(EMF[i, :, 0] - threshold))).flatten()

        intersectionPoints.append(proper[idx][0])

  

    dfIntersect = pd.DataFrame(intersectionPoints)

    dfRelevants = pd.DataFrame(averRelevants)

         

    distance = abs(dfIntersect - averageProt)

    if numOfIter > 1:

        average = dfRelevants.iloc[0:(numOfIter - 1), :].apply(lambda row: abs(row - list(averageProt.T)), axis = 1)

    else: average = dfRelevants.iloc[0, :]

            

    weight = list(10.*(distance.T / (((numOfIter - 1) * average) + abs(averRelevants[numOfIter - 1] - list(averageProt.T)))).iloc[0, :])

        

    for index, i in enumerate(weight):  

        

        if(i < 1): weight[index] = 1

                

    totalWeights.append(weight)

    print(weight)

    

    evolvedEMF = calculateEMF([0, 1, 2, 3, 4, 5, 6], weight) 

    print(displayRecommendations(evolvedEMF, threshold))

    

    time2 = time.time()



    print(time2 - time1)

    

def evaluate(iterations):

    

    print(displayRecommendations(exampleEMF, threshold))

    

    for i in range(1, iterations + 1):

    

        evolveEMF(i)

        

#evaluate(3)  

#print(totalWeights)
#display membership functions per numerical feature

threshold = 0.85



(membershipPF, lenProt) = calcEMFPF(prototypes)



for ii in range(0, len(numerical)):

    

    for z in range(lenProt):



        xAxis = (sequence * (max(processedNum.iloc[:, ii]) - min(processedNum.iloc[:, ii])) + min(processedNum.iloc[:, ii]))

        

        name = "Prototype"

        if(z == lenProt): name = "Weighted"

        

        plt.plot(xAxis, membershipPF[ii, :, z], linewidth= 3, label= name)        

        plt.xlabel(processedNum.columns[ii], fontsize=12) 

        plt.axhline(threshold, color='r')

        

        ymax = max(membershipPF[ii, :, z])

        

        idx = np.argwhere(np.diff(np.sign(membershipPF[ii, :, z] - threshold))).flatten()

        plt.vlines(xAxis[idx[0]], ymax = threshold, ymin = 0, color = 'r', linestyle = "--")

        if(len(idx) > 1): plt.vlines(xAxis[idx[1]], ymax = threshold, ymin = 0, color = 'r', linestyle = "--")



        

    plt.ylabel(r'$\epsilon$' + "MF", fontsize=12)

    plt.ylim(ymax = 1.3, ymin = 0)

    plt.xlim(xmin = 0)

               

    plt.axhline(y = 1, linestyle = "--", linewidth = 0.5)

    ax = plt.gca()

    ax.set_facecolor('xkcd:white')

    ax.spines['left'].set_color('black')

    ax.spines['bottom'].set_color('black') 

    #xlims = ax.get_xlim()

    #ax.set_xticks(xlims)

    

    plt.grid(color='gray', linestyle='-', linewidth=0.3)

            

    plt.legend(loc='upper left', frameon=True)

    plt.show()    
#display correlation of two attributes and their empirical membership function



def display3DPlot(attribute1, attribute2):

    

    for prototype in range(0, len(prototypes)):

        

        y = finalData.iloc[:, attribute1]

        x = finalData.iloc[:, attribute2]



        z = calculateEMF([attribute1, attribute2], [1, 8]).iloc[:, prototype]

        fig = plt.figure(figsize=(7,5))

        ax = fig.gca(projection='3d')

        ax.set_zlabel("evolved" + r'$\epsilon$' + "MF", fontsize=12, rotation = 90)

        ax.zaxis.set_rotate_label(False)

        ax.set_xlabel(x.name)

        ax.set_ylabel(y.name, rotation = 105)

        ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5, s = 5);

        ax.view_init(elev=45, azim=75)  

        

        z = calculateEMF([attribute1, attribute2], [1, 1]).iloc[:, prototype]

        fig = plt.figure(figsize=(7,5))

        ax = fig.gca(projection='3d')

        ax.set_zlabel(r'$\epsilon$' + "MF", fontsize=12, rotation = 90)

        ax.zaxis.set_rotate_label(False)

        ax.set_xlabel(x.name)

        ax.set_ylabel(y.name, rotation = 105)

        ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5, s = 5);

        ax.view_init(elev=45, azim=75)   

        

display3DPlot(1, 5)
def calcRecall(allRelevant, sugRelevant):

    

    recall = int(sugRelevant)/ int(allRelevant)

    

    return recall



def calcPrecision(sugRelevant, suggested):

    

    precision = int(sugRelevant)/int(suggested)



    return precision
results = pd.read_csv("../input/results3/Results.csv")



def evaluate(numOfIter, df):

         

    precision = pd.DataFrame([df.iloc[:, i].div(df.iloc[:, i + 3]) for i in range(1, 4)])

    recall = pd.DataFrame([df.iloc[:, j].div(df.iloc[:, 0]) for j in range(1, 4)])

    

    averagePrec = precision.mean(axis = 1)

    averageRec = recall.mean(axis = 1)

          

    plt.plot(range(1, numOfIter + 1), averagePrec, linewidth= 3)

    plt.xlabel("Number of iterations of evolving $\epsilon$FRB algorithm. ", fontsize=12)

    plt.xticks(np.arange(0, numOfIter, 1.0))

    #plt.ylim(1)

    plt.xlim(0, 4)

    plt.ylabel("Precision", fontsize=12) 

    plt.show()

    

    plt.plot(range(1, numOfIter + 1), averageRec, linewidth= 3) 

    plt.xlabel("Number of iterations of evolving $\epsilon$FRB algorithm. ", fontsize=12) 

    plt.xticks(np.arange(0, numOfIter, 1.0))

    plt.xlim(0, 4)

    plt.ylabel("Recall", fontsize=12) 

    plt.show()

 

evaluate(3, results)
def plotChangeOfWeights():

    

    plt.plot([1, 1.35, 2.5], linewidth= 3)

    plt.xlabel("Number of iterations of evolving $\epsilon$FRB algorithm. ", fontsize=12)

    plt.xticks(np.arange(0, 3, 1.0))

    plt.ylabel("Price attribute weight", fontsize=12) 

    plt.show() 

    

plotChangeOfWeights()
from scipy import polyval, polyfit



def timeEMF():

    

    ticks = np.array([1000, 5000, 10000, 50000, 100000, 150000])

    times = np.array([0.26, 1.23, 2.84, 13.01, 14.8, 16])

    

    ticksCont = np.array([i for i in range (0, 150000)])

    [a, b] = np.polyfit(np.log(ticks), times, 1)

    timesLinear = 1.6*(0.9*a*np.log(ticksCont) + b)

    #print(timesLinear)



    plt.plot(ticks, times, 'ro')

    plt.plot(ticksCont, timesLinear)

    plt.ylabel("Time in secons [s]")

    plt.xlabel("Number of data samples")

    plt.ylim(0, 20)

    plt.show()

    

    

timeEMF()
def scatterPrecRec():

    

    nr = np.array([88, 64, 53, 30, 80, 25, 71, 87, 45, 106])

    nrs = np.array([9, 2, 5, 6, 7, 15, 3, 6, 4, 1])

    ns = np.array([17, 3, 10, 12, 8, 21, 3, 9, 7, 1])

    

    precision = nrs/ns

    recall = nrs/nr

    

    print(precision.mean())

    

    plt.scatter(precision, recall)

    plt.xlim(0,1)

    plt.xlabel("Precision")

    plt.axvline(precision.mean())

    

    plt.ylim(0, 1)

    plt.ylabel("Recall")

    plt.axhline(recall.mean())

    

    plt.plot(precision.mean(), recall.mean(), '--bo')

    plt.annotate('The average Precision/Recall', xy=(precision.mean(), recall.mean()), xytext=(0.15, 0.5),

            arrowprops=dict(facecolor='red', shrink=1),

            )

    plt.show()

    

scatterPrecRec()


