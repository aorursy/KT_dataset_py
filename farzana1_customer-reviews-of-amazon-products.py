# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.                            



import numpy as np

import pandas as pd

#import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords
# Reading data from csv file

df = pd.read_csv('../input/amazon_cells_labelled.csv')

df.head()
# show  how many columns and rows are 

df.shape
# Taking two columns comments and sentiment

data = pd.DataFrame()

for row_index,row in df.iterrows():

    comment=""

    sentiment=0

    for index,value in row.iteritems():

        if(type(value)==str):

            if not value.isdigit():

                comment+=value+" "

            else:

                sentiment = int(value)

                break

    data = data.append({"comment":comment,"sentiment":sentiment},ignore_index=True)

print(data)
import re

file = open('../input/amazon_cells_labelled.csv').read()

print(re.sub(r"[^A-Za-z0-9]+", ' ', file))



#print(re.search(pattern, file))
import re

file = open('../input/amazon_cells_labelled.csv').read()

for k in file.split("\n"):

    #print(re.sub(r"|[0\n]",' ',file))

    #print(re.sub(r"|[.0-1\n]",' ',file))

    print(re.sub(r"[^A-Za-z0-9]+", ' ', k))



#print(re.search(pattern, file))
import re

f = open("../input/amazon_cells_labelled.csv")

text = f.read()

f.close()

text = re.sub(r",+\n","\n",text)

text = re.sub(r",0\n","|0\n",text)

text = re.sub(r",1\n","|1\n",text)

text = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_'~()\"]","",text)

#print(text)

f = open("clean.csv","w")

f.write(text)

f.close()