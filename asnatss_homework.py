# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from math import floor
df = pd.read_csv("../input/train.csv")

resultCsv = pd.read_csv("../input/solutionex.csv")

#считаем веса для каждого дня 

# w = (d + 1 -i)/d

countWeek = 1099 // 7

sumWeight = 0

weightArray = np.zeros(shape=(countWeek, 1))

for x in range(0, countWeek - 30 ):

    curWeight = pow((countWeek - x + 1)/countWeek, 1.17)

    sumWeight += curWeight

    weightArray[x] = curWeight

for x in range(0, countWeek):

    weightArray[x] /= sumWeight

weightArray = weightArray[::-1]

print(weightArray[156])
#предсказываем день

def predictDayWeeks(array):

    gistNumber = np.zeros(shape=(7, 1))

    sizeCurArray = len(array)

    for number in range(1, len(array)):

        index = ((array[number] - 1)%7)

        numberWeek = array[number] // 7

        if numberWeek >= countWeek:

            numberWeek = countWeek - 1

        gistNumber[index] += weightArray[numberWeek]

    maxIndex = -1

    maxResult = 0

    for x in range(0, len(gistNumber)):

        if gistNumber[x] > maxResult:

            maxResult = gistNumber[x]

            maxIndex = x

    return maxIndex
#максимальный размер

sizeArray = resultCsv.shape[0]
#предсказываем для каждого пользователя

for x in range(0, sizeArray):

    curRow = df.iloc[x]

    array = [int(z) for z in curRow[1].split()]

    resultValue = predictDayWeeks(array)

    resultValue += 1

    resultCsv["nextvisit"][x] = str(' ' + str(int(resultValue)))
resultCsv.to_csv('solution.csv', index=False, sep =',')