# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dfTrain=pd.read_csv("../input/train.csv")

print(dfTrain)



dfTrain.onpromotion.value_counts()
dfTrain.item_nbr.value_counts()
dfTest=pd.read_csv("../input/test.csv")

print(dfTest)
testItems= pd.unique(dfTest.item_nbr)

print(testItems)

trainItems= pd.unique(dfTrain.item_nbr)

print(trainItems)

itemsTrainTest = set(testItems).intersection(set(trainItems))

print(itemsTrainTest)

import pylab as plt

from matplotlib_venn import venn2, venn3_circles
v= venn2(subsets=(len(trainItems),len(itemsTrainTest),len(testItems)), set_labels = ('Train Dataset','Test Dataset'))

v.get_patch_by_id('10').set_color('darkred')

v.get_patch_by_id('01').set_color('yellow')

v.get_patch_by_id('11').set_color('darkblue')
