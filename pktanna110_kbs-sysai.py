# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import nltk

# Any results you write to the current directory are saved as output.
test=[['tea is cold','make it hot'],['tea too sweet','make new tea'],['tea is sugerless','add suger']]

print(test[1][0])
print('enter a problem to get solution')

j=input()
x=[]

for i in range(len(test)):

    x.append(test[i][0])

    y=nltk.word_tokenize(test[i][0])

    p=0

    n=0

    for word in j.split():

        if word.lower() in y:

            p=p+1

        else:

            n=n+1

    if(p>0):

        print(test[i][0]+"  ==  "+test[i][1])

    #print (str(n)+' '+str(p))