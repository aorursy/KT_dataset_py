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
import pandas as pd

import numpy as np

import random

iris=pd.read_csv("../input/Iris.csv")



def distance(p1,p2):

    dist=np.sqrt(np.sum(np.power(p2-p1,2)))

    return dist



def majority_vote(votes):

    vote_count={}

    for vote in votes:

        if vote in vote_count:

            vote_count[vote]+=1

        else:

            vote_count[vote]=1

    winner=[]

    max_count=max(vote_count.values())

    for vote,count in vote_count.items():

        if count==max_count:

            winner.append(vote)

    return random.choice(winner)



def find_nearest(p,points,k=5):

    distances=np.zeros(points.shape[0])

    for i in range(len(distances)):

        distances[i]=distance(p,points[i])

    ind=np.argsort(distances)

    return ind[:k]



def knn_classifer(p,points,k,outcomes):

    ind=find_nearest(p,points,k)

    return majority_vote(outcomes[ind])



points=np.array([iris["SepalLengthCm"][0],iris["SepalWidthCm"][0],iris["PetalLengthCm"][0],iris["PetalWidthCm"][0]])

for i in range(1,len(iris["Species"])):

    ad=np.array([iris["SepalLengthCm"][i],iris["SepalWidthCm"][i],iris["PetalLengthCm"][i],iris["PetalWidthCm"][i]])

    points=np.vstack([points,ad])

outcomes=np.array(iris["Species"])



p=np.array([5.1,3.5,1.4,0.2])

f=knn_classifer(p,points,5,outcomes)

print(p," belongs to class ",f)