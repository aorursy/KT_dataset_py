# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv
import numpy as np
import random
from scipy import stats

data=[]
with open ('/kaggle/input/drug-classification/drug200.csv') as csv_file:
           csv_reader=csv.reader(csv_file,delimiter=',')
           for row in csv_reader:
               data.append(row)

data=data[1:]

age=[D[0] for D in data]    
sex=[D[1] for D in data]    
BP=[D[2] for D in data]     
chol=[D[3] for D in data]   
na2k=[D[4] for D in data]   
drug=[D[5] for D in data]   

score=0
counter=0
for D in data:
    
    randSample=D
    

    drugLabels=['Y','A','B','C','X']
    ageRanges=[[15,74],[19,50],[51,74],[16,72],[15,74]]
    bpLevels=['LOW','NORMAL','HIGH']
    bpLabels=[['Y','C','X'],['Y','X'],['Y','A','B']]
    cLevels=['NORMAL','HIGH']
    cholLabels=[['Y','A','B','X',],['Y','A','B','C','X']]
    NA2KLabels=[['Y'],['A','B','C','X']]
    totalResults=[]

    index=0
    ageLabels=[]
    for A in ageRanges:
        if A[0]<int(randSample[0])<A[1]:
            ageLabels.append(drugLabels[index])
        index+=1
    totalResults.append(ageLabels)

    index=0
    for B in bpLevels:
        if randSample[2]==B:
            totalResults.append(bpLabels[index])
        index+=1

    index=0
    for C in cLevels:
        if randSample[3]==C:
            totalResults.append(cholLabels[index])
        index+=1

    if 15<float(randSample[4] ):
        totalResults.append(NA2KLabels[0])
    else:
        totalResults.append(NA2KLabels[1])
    drugNames=['drugA','drugB','drugC','drugX','DrugY']
    tally=[0,0,0,0,0]
    for T in totalResults:
        for t in T:
            if t=='A':
                tally[0]+=1
            elif t=='B':
                tally[1]+=1
            elif t=='C':
                tally[2]+=1
            elif t=='X':
                tally[3]+=1
            elif t=='Y':
                tally[4]+=1

    if drugNames[np.argmax(tally)]==randSample[-1]:
        score+=1
    counter+=1

print(score/counter)
