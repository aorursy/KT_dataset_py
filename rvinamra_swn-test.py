# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import sentiwordnet as swn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
non_elite = pd.read_csv('../input/la_non_elite.csv')
print ('Non-elite data shape: ', non_elite.shape)
non_elite.head(10)
elite = pd.read_csv('../input/la_elite.csv')
print ('Elite data shape: ', elite.shape)
elite.head(10)
from nltk.corpus import wordnet as wn
sv = ['like', 'dislike', 'love', 'hate', 'hear', 'imagine', 'impress', 
     'smell', 'see', 'seem', 'sound', 'stay', 'suppose', 'taste', 'think'
     'understand', 'wish', 'feel', 'lack', 'look', 'mind', 'know', 'need',
     'believe', 'owe', 'deny', 'prefer', 'have', 'doubt', 'resemble', 'remember'
     'appear', 'satisfy', 'possess', 'please', 'want', 'concern', 'weigh',
     'grow', 'fit', 'exist', 'include', 'involve', 'belong', 'consist', 'contain',
     'depend', 'own', 'agree', 'disagree', 'mean', 'deserve', 'release', 'measure',
     'recognise', 'remain', 'turn', 'matter', 'is', 'are', 'am', 'was', 'be', 'being',
     'were', 'been']
for num in range(100):
    nev = eval(non_elite['Non-elite_Verb'][num])
    for i in nev:
        print(i)
    classification = list()
    for i in nev:
        if i in sv:
            classification.append('SV')
        elif i not in sv:
            y = swn.senti_synsets(i, 'v')
            y0 = list(y)[0:3]
            count = 0
            for j in y0:
                count+= 1 - j.obj_score()
            print(count/3)
            if count/3 >= 0.6:
                classification.append('SA')
            elif count/3 < 0.6 and count/3 >= 0.2:
                classification.append('IAV')
            else:
                classification.append('DAV')
            print('break')
    print(classification)
for num in range(100):
    ev = eval(elite['Elite_Verb'][num])
    for i in ev:
        print(i)
    classification = list()
    for i in ev:
        if i in sv:
            classification.append('SV')
        elif i not in sv:
            y = swn.senti_synsets(i, 'v')
            y0 = list(y)[0:3]
            count = 0
            for j in y0:
                count+= 1 - j.obj_score()
            print(count/3)
            if count/3 >= 0.6:
                classification.append('SA')
            elif count/3 < 0.6 and count/3 >= 0.2:
                classification.append('IAV')
            else:
                classification.append('DAV')
            print('break')
    print(classification)