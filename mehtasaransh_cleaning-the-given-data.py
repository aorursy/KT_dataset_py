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
import re

import string

import os
trainPath = "../input/train.csv"

testPath = "../input/test.csv"
df = pd.read_csv(trainPath)

df.head()
totalContentCleaned = []

punctDict = {}

for punct in string.punctuation:

    punctDict[punct] = None

transString = str.maketrans(punctDict)

# since we intent to remove any punctuation with ''

for sen in df['comment_text']:

    

    #cleanedString = re.sub('[^a-zA-Z]+', '', sen)

    

    p = sen.translate(transString)

    totalContentCleaned.append(p)
totalContentCleaned[:5]
df['comment_text'] = totalContentCleaned

# we can save the file to csv if we want in local machine

#df.to_csv(os.path.join(os.path.abspath('data'), 'train_cleaned.csv'), index = False)
df2 = pd.read_csv(testPath)
df2.head()
totalContentCleaned = []

for sen in df2['comment_text']:

    

    #cleanedString = re.sub('[^a-zA-Z]+', '', sen)

    sen = str(sen)

    p = sen.translate(transString)

    totalContentCleaned.append(p)

df2['comment_text'] = totalContentCleaned
df2.head()

#df2.to_csv(os.path.join(os.path.abspath('data'), 'test_cleaned.csv'), index = False)