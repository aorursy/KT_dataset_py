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
import numpy as np

class Fuzzy_Operations():

    def union(self,a,b):
        d = dict()
        for x in a.keys():
            d[x] = max(a[x],b[x]) 
        return d

    def intersection(self,a,b): 
        d = dict()
        for x in a.keys():
            d[x] = min(a[x],b[x]) 
        return d

    def complement(self,a): 
        d = dict()
        for x in a.keys():
            d[x] = round(1 - a[x],4) 
        return d

    def containment(self,a,b): 
        for k in a.keys():
            if a[k] > b[k]:
                return str(a)+" is not a subset of "+str(b) 
        return str(a)+" is a subset of "+str(b)

def checkForEquality(a,b):
    for k in a.keys(): 
        if a[k] != b[k]:
            return False
    return True

def normalizeFuzzySets(a,b): 
    ak = a.keys()
    bk = b.keys() 
    for x in ak:
        if x not in bk: 
            b[x] = 0
    for x in bk:
        if x not in ak: 
            a[x] = 0
    return a,b

def performOperations(a,b): 
    f = Fuzzy_Operations()
    a,b = normalizeFuzzySets(a,b)
    print(f.containment(a,b)) 
    print(f.containment(b,a)) 
    print("Complement(A)",f.complement(a)) 
    print("Complement(B)",f.complement(b)) 
    print("Intersection(A,B)",f.intersection(a,b)) 
    print("Union(A,B)",f.union(a,b))


def verifyDeMorgansLaw(a,b):
    f = Fuzzy_Operations()
    lhs1 = f.complement(f.union(a,b))
    rhs1 = f.intersection(f.complement(a),f.complement(b)) 
    print("DeMorgans Law 1 verification is "+str(checkForEquality(lhs1,rhs1)))
    lhs2 = f.complement(f.intersection(a,b))
    rhs2 = f.union(f.complement(a),f.complement(b)) 
    print("DeMorgans Law 2 verification is "+str(checkForEquality(lhs2,rhs2)))
samples = list() 
samples.append(dict({"A":dict({1:0.2,2:0.3,3:0.8,4:1}),"B":dict({1:0.3,2:0.2,3:0.5,4:0.8})}))
samples.append(dict({"A":dict({2:0.3,3:0.5}),"B":dict({1:0.2,2:0.3,3:0.6,4:0.8})}))

for i,sample in enumerate(samples): 
    print("Sample "+str(i+1)) 
    performOperations(sample['A'],sample['B']) 
    verifyDeMorgansLaw(sample['A'],sample['B']) 
    print()
import numpy as np
from skfuzzy.membership import trapmf,gaussmf,gbellmf,smf 
import matplotlib.pyplot as plt

class Age():
    def __init__(self):
        self.ages = np.arange(1,101,1) 
        self.createFuzzySets() 
        plt.title("Age")
        plt.show()

    def createFuzzySets(self): 
        self.constructYoung() 
        self.constructMiddle() 
        self.constructOld()

    def constructYoung(self):
        mf = trapmf(self.ages,[0,0,30,40])
        plt.plot(self.ages,mf)

    def constructMiddle(self):
        mf = gaussmf(self.ages,40,10) 
        plt.plot(self.ages,mf)

    def constructOld(self):
        mf = trapmf(self.ages,[40,70,100,100]) 
        plt.plot(self.ages,mf)
class RaceCars():
    def __init__(self):
        self.speeds = np.arange(0,400,1) 
        self.constructSet() 
        plt.title("Speeds for Race Cars") 
        plt.show()

    def constructSet(self):
        mf = smf(self.speeds,100,300) 
        plt.plot(self.speeds,mf)
class ACTemp():
    def __init__(self):
        self.temps = np.arange(10,40,1) 
        self.constructSet() 
        plt.title("Temperature for AC") 
        plt.show()

    def constructSet(self):
        mf = gbellmf(self.temps,15,2,24) 
        plt.plot(self.temps,mf)
Age()
RaceCars() 
ACTemp()
