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
import copy
import itertools
from collections import defaultdict
from operator import itemgetter
file=open('/kaggle/input/msn-dataset/msnbc990928.seq','r')
dataset=[]
for line in file:
    x=line.split()
    dataset.append(x)
dataset
#Checking if a subsequence is a subsequence of the ain sequence

def isSubsequence(mainSequence,subSequence):
    subSequenceClone=list(subSequence)
    return isSubsequenceRecursive(mainSequence,subSequenceClone)

def isSubsequenceRecursive(mainSequence,subSequenceClone,start=0):
    if(not subSequenceClone):
        return True
    firstElem=set(subSequenceClone.pop(0))
    for i in range(start,len(mainSequence)):
        if(set(mainSequence[i]).issuperset(firstElem)):
            return isSubsequenceRecursive(mainSequence,subSequenceClone,i+1)
    return False
sample=[['1'],['2','3'],['4'],['1','5']]
isSubsequence(sample,[['1'],['4'],['5']])
isSubsequence(sample,[['1'],['2','3'],['5']])
isSubsequence(sample,[['1'],['2','4']])
#Computing length

def sequenceLength(sequence):
    return sum(len(i) for i in sequence)
sequenceLength([['1'],['2','3'],['5'],['2','3','4']])
#Computing support

def countSupport(dataset,candidateSequence):
    return sum(1 for seq in dataset if isSubsequence(seq,candidateSequence))
dataset
countSupport(dataset,[['2']])
countSupport(dataset,[['1'],['2','3']])
countSupport(dataset,[['1','1']])
countSupport(dataset,[['1'],['1']])
#Apriori candidate generation

def generateCandidatesForPair(cand1,cand2):
    cand1Clone=copy.deepcopy(cand1)
    cand2Clone=copy.deepcopy(cand2)
    if(len(cand1[0])==1):
        cand1Clone.pop(0)
    else:
        cand1Clone[0]=cand1Clone[0][1:]
    if (len(cand2[-1])==1):
        cand2Clone.pop(-1)
    else:
        cand2Clone[-1]=cand2Clone[-1][:-1]
    
    if not cand1Clone==cand2Clone:
        return []
    else:
        newCandidate=copy.deepcopy(cand1)
        if (len(cand2[-1])==1):
            newCandidate.append(cand2[-1])
        else:
            newCandidate[-1].extend(cand2[-1][-1])
        return newCandidate
candidateA=[['1'],['2','3'],['4']]
candidateB=[['2','3'],['4','5']]
generateCandidatesForPair(candidateA,candidateB)
candidateA=[['1'],['2','3'],['4']]
candidateB=[['2','3'],['4'],['5']]
generateCandidatesForPair(candidateA,candidateB)
candidateA=[['1'],['2','3'],['4']]
candidateB=[['1'],['2','3'],['5']]
generateCandidatesForPair(candidateA,candidateB)
#Candidate generation

def generateCandidates(lastLevelCandidates):
    k=sequenceLength(lastLevelCandidates[0])+1
    if (k==2):
        flatShortCandidates=[item for sublist2 in lastLevelCandidates for sublist1 in sublist2 for item in sublist1]
        result=[[[a,b]] for a in flatShortCandidates for b in flatShortCandidates if b>a]
        result.extend([[[a],[b]] for a in flatShortCandidates for b in flatShortCandidates])
        return result
    else:
        candidates=[]
        for i in range(0,len(lastLevelCandidates)):
            for j in range(0,len(lastLevelCandidates)):
                newCand=generateCandidatesForPair(lastLevelCandidates[i],lastLevelCandidates[j])
                if (not newCand==[]):
                    candidates.append(newCand)
        candidates.sort()
        return candidates
lastLevelFrequentPatterns =[[['1','2']],[['2','3']],[['1'],['2']],[['1'],['3']],[['2'],['3']],[['3'],['2']],[['3'],['3']]]
newCandidates=generateCandidates(lastLevelFrequentPatterns)
newCandidates
#Computing direct subsequences

def generateDirectSubsequences(sequence):
    result=[]
    for i,itemset in enumerate(sequence):
        if(len(itemset)==1):
            sequenceClone=copy.deepcopy(sequence)
            sequenceClone.pop(i)
            result.append(sequenceClone)
        else:
            for j in range(len(itemset)):
                sequenceClone=copy.deepcopy(sequence)
                sequenceClone[i].pop(j)
                result.append(sequenceClone)
    return result
#Candidate pruning

def pruneCandidates(candidatesLastLevel,candidatesGenerated):
    return [cand for cand in candidatesGenerated if all(x in candidatesLastLevel for x in generateDirectSubsequences(cand))]
candidatesPruned=pruneCandidates(lastLevelFrequentPatterns,newCandidates)
candidatesPruned
minSupport=2
candidatesCounts=[(i,countSupport(dataset,i)) for i in candidatesPruned]
resultLvl=[(i,count) for (i,count) in candidatesCounts if(count>=minSupport)]
resultLvl
#Apriori algorithm

def apriori(dataset,minSupport,verbose=False):
    global numberOfCountingOperations
    numberOfCountingOperations=0
    Overall=[]
    itemsInDataset=sorted(set([item for sublist1 in dataset for sublist2 in sublist1 for item in sublist2]))
    singleItemSequences=[[[item]] for item in itemsInDataset]
    singleItemCounts=[(i,countSupport(dataset,i)) for i in singleItemSequences if countSupport(dataset,i)>=minSupport]
    Overall.append(singleItemCounts)
    print("Result, lvl 1: "+str(Overall[0]))
    k=1
    while(True):
        if not Overall[k-1]:
            break
        candidatesLastLevel=[x[0] for x in Overall[k-1]]
        candidatesGenerated=generateCandidates(candidatesLastLevel)
        candidatesPruned=[cand for cand in candidatesGenerated if all(x in candidatesLastLevel for x in generateDirectSubsequences(cand))]
        candidatesCounts=[(i,countSupport(dataset,i)) for i in candidatesPruned]
        resultLvl=[(i,count) for(i,count) in candidatesCounts if(count>=minSupport)]
        if verbose:
            print("Candidates generated, lvl "+str(k+1)+": "+str(candidatesGenerated))
            print("Candidates pruned, lvl "+str(k+1)+": "+str(candidatesPruned))
            print("Result, lvl "+str(k+1)+": "+str(resultLvl))
        Overall.append(resultLvl)
        k=k+1
    Overall=Overall[:-1]
    Overall=[item for sublist in Overall for item in sublist]
    return Overall
apriori(dataset,2,verbose=False)
