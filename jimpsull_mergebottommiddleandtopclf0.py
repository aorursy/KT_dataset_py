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
def assembleClf(num):
    
    prefixString='../input/svmclf' + str(num)
    botdf = pd.read_csv(prefixString + 'bot/clf' + str(num) + 'SvmBot.csv')
    middf = pd.read_csv(prefixString + 'mid/clf' + str(num) + 'SvmMid.csv')
    topdf = pd.read_csv(prefixString + 'top/clf' + str(num) + 'SvmTop.csv')
    
    clfdf=pd.concat([botdf, middf, topdf], sort=True)
    return clfdf

clf0=assembleClf(0)
print(clf0.shape)

clf1=assembleClf(1)
print(clf1.shape)

clf2=assembleClf(2)
print(clf2.shape)

clf3=assembleClf(3)
print(clf3.shape)

clf4=assembleClf(4)
print(clf4.shape)

svmdf = 0.2*(clf0 + clf1 + clf2 + clf3 + clf4)
svmdf['object_id']=clf0['object_id']
svmdf.head()
svmdf.describe()
svmdf.to_csv('svmDf.csv', index=False)