# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import threading

from threading import Thread
# Simple example

def threadFunc(ThreadNumb, ThreadName):

    print(str(ThreadNumb) + ThreadName)



t1Name = ': FirstThread'

t2Name = ': SecondThread'

t3Name = ': ThirdThread'



# init threads

t1 = threading.Thread(target=threadFunc, args=(0, t1Name))

t2 = threading.Thread(target=threadFunc, args=(1, t2Name))

t3 = threading.Thread(target=threadFunc, args=(2, t3Name))



# start threads

t1.start()

t2.start()

t3.start()



# join threads to the main thread

t1.join()

t2.join()

t3.join()
# Advanced example



# This is a piece of code from my personal project, it is given for review

# and can not be simply run on Kaggle due to lack of necessary data.



# It calculates the number of each combination of the two services in each order document.



# HeatmapData - input data to calculate the numbers for heatmap plot.

# HeatmapTable - result data for heatmap plot.



def threadFunc(ThreadNumb, HeatmapData, HeatmapTableOriginal):

    HeatmapTableCopy = HeatmapTableOriginal[HeatmapTableOriginal.columns]

    DocsPart = pd.read_pickle('DocsPart_' + str(ThreadNumb) + '.pkl')



    print('Thread: ' + str(ThreadNumb) + ' ' + str(datetime.datetime.now()))

    for Doc in DocsPart:

        DocServices = HeatmapData.loc[HeatmapData.OrderDoc == Doc].Service



        for i in DocServices:

            for j in DocServices:

                if i != j:

                    HeatmapTableCopy.loc[i][j] = HeatmapTableCopy.loc[i][j] + 1

    print('Thread: ' + str(ThreadNumb) + ' ' + str(datetime.datetime.now()))

    

    HeatmapTableCopy.to_pickle('HeatmapTablePart' + str(ThreadNumb) + '.pkl')



def main(HeatmapData, HeatmapTable): 

    ThreadNumb = 6

    DocNumb    = len(Docs.index)

    StepSize   = int(DocNumb / ThreadNumb) + 1



    StartPos = 0



    # divide data

    for i in range(ThreadNumb):

        DocsPart = Docs[StartPos:StartPos + StepSize]

        StartPos = StartPos + StepSize

        DocsPart.to_pickle('DocsPart_' + str(i) + '.pkl')



    Threads = np.empty(ThreadNumb, dtype=threading.Thread) 



    # start threads

    for i in range(ThreadNumb):

        Threads[i] = threading.Thread(target=threadFunc, args=(i, HeatmapData, HeatmapTable))

        Threads[i].start()

        print('Started thread: ' + str(i))



    # join threads to the main thread    

    for i in range(ThreadNumb):

        Threads[i].join()

        print('Join thread: ' + str(i))



    # consolidate calculations

    for i in range(ThreadNumb):

        HeatmapTablePart = pd.read_pickle('HeatmapTablePart' + str(ThreadNumb) + '.pkl')

        HeatmapTable = HeatmapTable + HeatmapTablePart



    # show

    HeatmapTable