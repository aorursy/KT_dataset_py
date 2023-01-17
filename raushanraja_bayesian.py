# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
d={

    'age':pd.Series(['youth','youth','middle','senior','senior','senior','middle','youth','youth','senior','youth','middle','middle','senior']),

    'income':pd.Series(['high','high','high','medium','low','low','low','medium','low','medium','medium','medium','high','medium']),              

    'student':pd.Series(['no','no','no','no','yes','yes','yes','no','yes','yes','yes','no','yes','no']),

    'credit rating':pd.Series(['fair','excellent','fair','fair','fair','excellent','excellent','fair','fair','fair','excellent','excellent','fair','excellent']),

    'buys computer':pd.Series(['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no'])

}

table=pd.DataFrame(d)

age=table['age']

income=table['income']

student=table['student']

credit=table['credit rating']

buysComputer=table['buys computer']

table.to_csv('email.csv', index = False)
def countDict(data):

    distinct=list(data.unique())   

    data=list(data)

    dicts=dict()

    for i in distinct:

        dicts[i]=data.count(i)

    return dicts
ageCount=countDict(age)

incomeCount=countDict(income)

studentCount=countDict(student)

creditCount=countDict(credit)

buysComputerCount=countDict(buysComputer)

# print(ageCount['youth'])

ageCount
yeslist=set(table.index[ table['buys computer'] == 'yes'].tolist())

yesnumber=len(yeslist)



nolist=set((table.index[ table['buys computer'] == 'no'].tolist()))

nonumber=len(nolist)



ProbNo=nonumber/(yesnumber+nonumber)

ProbYes=yesnumber/(yesnumber+nonumber)

print("Probability of buys computer Yes:",ProbYes)

print("Probability of buys computer NO:",ProbNo)

def findProbXi(data,param,dicts):

    findyes=set(table.index[ table[data] == param].tolist())

    yes=len(findyes.intersection(yeslist))

    no=(dicts[param])-yes

    return yes/yesnumber,no/nonumber



    

    
ageWhenYes,ageWhenNo=findProbXi('age','youth',ageCount)

incomeWhenYes,incomeWhenNo=findProbXi('income','medium',incomeCount)

studentWhenYes,studentWhenNo=findProbXi('student','yes',studentCount)

creditWhenYes,creditWhenNo=findProbXi('credit rating','fair',creditCount)
probYesWhenX=ageWhenYes*incomeWhenYes*studentWhenYes*creditWhenYes*ProbYes

print("c1=Buys Computer : Yes")

print("P(C1/X): %.5f" %probYesWhenX,"\n\n")

probNoWhenX=ageWhenNo*incomeWhenNo*studentWhenNo*creditWhenNo*ProbNo

print("c2=Buys Computer : NO")

print("P(C2/X): %.5f"%probNoWhenX)
