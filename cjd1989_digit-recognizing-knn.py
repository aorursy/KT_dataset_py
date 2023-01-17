import numpy as np
import pandas as pd

data=pd.read_csv('../input/train.csv')
#type(data.ix[5])
#labels=data['label'].values


def calcDist(point1,point2): #calculates the Euclidean distance between two points represented as Numpy arrays
    diff=point2-point1
    squared=diff**2
    sum1=squared.sum()
    root=(sum1)**0.5
    return root
    
def distanceCalc(df,query):
    for num in range(0,2):
    #for num in range(0,len(df)):
        #print(df[num:num+1].values)
        arr=df[num:num+1].values
        print(type(arr))
        print(len(arr))
        print(arr[1:len(arr)])
    

    
        
distanceCalc(data,5)