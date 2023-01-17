import subprocess

import random

import pandas as pd

import seaborn as sns

import os

import numpy as np



perfCmd = ['sudo', 'perf', 'stat', '-x,']

fileDir = "randomFiles"

baseDir = '/'.join(['../input', 'randomfiles', fileDir])



numOfFiles = 10



# some functions and initialization done in my own machine like generating

# random files



def generateRandomFiles():

    for i in range(numOfFiles):

        randomFile = [str(random.randint(0, 10)) for i in range(2**i)]

        with open(fileDir + "/f%d" % i, 'w+') as fp:

            fp.write('\n'.join(randomFile))



def recordSort(trialNum):

    for i in range(numOfFiles):

        targetFile = fileDir+"/f%d" % i

        for j in range(trialNum):

            resultDir = '/'.join([fileDir, "result%d" % i,])

            if not os.path.exists(resultDir):

                os.mkdir(resultDir)

            resultFile = '/'.join([fileDir, "result%d" % i, "trial%d.csv" % j])

            newCmd = perfCmd.copy()

            newCmd.extend(['-o', resultFile, 'sort', targetFile,])

            subprocess.run(newCmd, stdout=subprocess.PIPE)

            # input(newCmd)



def plotProperCSV(nthTrial, trialNum):

    resultDir = '/'.join([baseDir, "result%d" % nthTrial,])

    data = pd.DataFrame(columns=['task-clock', 'context-switches', 'cpu-migrations', 'page-faults', 'cycles', 'instructions', 'branches', 'branch-misses'])

    for i in range(trialNum):

        data = getValue(resultDir+'/trial%d.csv'%i, data)

    powerNP = np.empty(trialNum)

    powerNP.fill(nthTrial)

    data = data.assign(power=pd.Series(powerNP).values)

    return data



def getValue(fn, data):

    keys = ['task-clock', 'context-switches', 'cpu-migrations', 'page-faults', 'cycles', 'instructions', 'branches', 'branch-misses']

    result = pd.read_csv(fn, names=['v','b','c','d','e','f','g'])

    result = result.drop([0])

    values = [float(i) for i in result.v.tolist()]

    return data.append(dict(zip(keys, values)), ignore_index=True)
# checking a random num, and see if the features fits a normal distribution

import matplotlib.pyplot as plt

fig, axarr = plt.subplots(6, 1, figsize=(12, 20))

p = plotProperCSV(5, 100)

sns.distplot(p['task-clock'], ax=axarr[0])

sns.distplot(p['page-faults'], ax=axarr[1])

sns.distplot(p['cycles'], ax=axarr[2])

sns.distplot(p['instructions'], ax=axarr[3])

sns.distplot(p['branches'], ax=axarr[4])

sns.distplot(p['branch-misses'], ax=axarr[5])
# put all data into one Pandas DataFrame

allData = pd.DataFrame(columns=['task-clock', 'context-switches', 'cpu-migrations', 'page-faults', 'cycles', 'instructions', 'branches', 'branch-misses', 'power'])

for i in range(10):

    allData = allData.append(plotProperCSV(i, 100), ignore_index=True)

allData.head()
# plot the violin plots that shows different plots at the same time and also

# shows the trend of each plot



fig, axarr = plt.subplots(6, 1, figsize=(12, 30))

sns.violinplot(x='power', y='task-clock', data=allData, ax=axarr[0])

sns.violinplot(x='power', y='page-faults', data=allData, ax=axarr[1])

sns.violinplot(x='power', y='cycles', data=allData, ax=axarr[2])

sns.violinplot(x='power', y='instructions', data=allData, ax=axarr[3])

sns.violinplot(x='power', y='branches', data=allData, ax=axarr[4])

sns.violinplot(x='power', y='branch-misses', data=allData, ax=axarr[5])