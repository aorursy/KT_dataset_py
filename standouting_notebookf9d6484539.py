# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import operator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def loadTrainData():

    l=[]

    with open('train.csv') as file:

         lines=csv.reader(file)

         for line in lines:

             l.append(line) #42001*785

    l.remove(l[0])

    l=array(l)

    label=l[:,0]

    data=l[:,1:]

    return nomalizing(toInt(data)),toInt(label)  #label 1*42000  data 42000*784

    #return data,label

  