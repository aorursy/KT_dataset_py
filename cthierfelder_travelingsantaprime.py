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
#import xgboost as xgb

import time
import pandas as pd
import numpy as np
from sklearn.utils import graph_shortest_path
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('/kaggle/input/traveling-santa-2018-prime-paths/cities.csv')
data
def findFirstPrimes(n):
    primes = [2]
    for i in range(3,n,2):
        isPrime = True
        for p in primes:
            if i%p == 0:
                isPrime = False
                break
        if isPrime == True:
            primes.append(i)
    return primes
numberOfRecords = data.shape[0]
relevantPrimes = findFirstPrimes(numberOfRecords)
print('number of cities: ',numberOfRecords)
print('number of relevant primes: ',len(relevantPrimes))
X = data['X']
Y = data['Y']
plt.figure(figsize=(15,9))
plt.plot(X,Y, 'ro',markersize=0.01)
Xprimes = [data.loc[p]['X'] for p in relevantPrimes]
Yprimes = [data.loc[p]['Y'] for p in relevantPrimes]
plt.plot(Xprimes,Yprimes, 'bo',markersize=0.1)
print(data['X'].max())
print(data['Y'].max())
def distance(city1,city2):
    id1, x1, y1 = data.loc[city1]
    id2, x2, y2 = data.loc[city2]
    vec = [(x1-x2),(y1-y2)]
    d = np.linalg.norm(vec)
    #print('DISTANCE CALL', city1,city2,d)
    return d

def calcTotalDistanceFast(myPath,includePrimeCondition=True,dMin=0,dMax=10000):
    myPath = np.array(myPath,dtype=None)
    #print('before: ',np.where(myPath==0),len(myPath))
    myPath = np.delete(myPath,np.where(myPath==0))
    #print('after:  ',np.where(myPath==0),len(myPath))
    myPath = np.insert(myPath, 0, 0, axis=0)
    myPath = np.append(myPath, 0)
    #print('after2: ',np.where(myPath==0),len(myPath))
    total_distance = 0
    longDistanceOrigList = []
    
    pathXYs=data.iloc[myPath,:]
    dX2s=np.square(np.diff(np.array(pathXYs.X)))
    dY2s=np.square(np.diff(np.array(pathXYs.Y)))

    relevantPrimes1=np.array(relevantPrimes)
    
    for step, cityId in enumerate(myPath[:-1]):
        dist = np.sqrt(dX2s[step]+dY2s[step])
        distFac = 1.0
        if includePrimeCondition:
            if (step+1)%10 == 0:
                if not np.isin([cityId],relevantPrimes1,assume_unique=True)[0]:
                    distFac = 1.1
        if step%30000 == 0:
            pass
            #print(step,total_distance)
        d=dist*distFac
        if dMin<d and d<dMax:
            longDistanceOrigList.append(step)
        total_distance += d
    return total_distance, myPath, longDistanceOrigList
#dumb path
myPath  = np.arange(numberOfRecords)
myPath  = np.append(myPath,0)
startTime = time.time()
totDist = calcTotalDistanceFast(myPath)
endTime = time.time()
print(totDist[0], 'running in ',endTime-startTime,'sec')
startTime = time.time()
totDist = calcTotalDistanceFast(myPath,False)
endTime = time.time()
print(totDist[0], 'running in ',endTime-startTime,'sec')  
#Initial Nearest Neighbour

DISTANCETHRESHOLD=10
myPath=[0]

currentCityId = 0

for step in range(197769):
    if step%10000==0 or step > 197760:
        print('step:',step,len(myPath),data[~data.CityId.isin(myPath)].shape[0])
    currentCityX = data.loc[currentCityId,'X']
    currentCityY = data.loc[currentCityId,'Y']
    for mutiplyer in [1,5,10,50,100,500,1000]:
        distThresh = mutiplyer*DISTANCETHRESHOLD
        currentCityNeigbours = data[~data.CityId.isin(myPath) & (currentCityX-distThresh<data.X) & (data.X<currentCityX+distThresh) & (currentCityY-distThresh<data.Y) & (data.Y<currentCityY+distThresh)]
        nbrOfNeighbours = currentCityNeigbours.shape[0]
        if nbrOfNeighbours>0:
            break
            
    neighbourVectors = np.array(list(zip(currentCityNeigbours['X'].to_list(),currentCityNeigbours['Y'].to_list())))
    cityVector       = [[currentCityX,currentCityY]]
    distances        = distance_matrix(cityVector,neighbourVectors)[0]
    nearestNeighbourIndex = np.argsort(distances)[0]
    nearestNeighbourId    = int(currentCityNeigbours.iloc[nearestNeighbourIndex]['CityId'])
    myPath.append(nearestNeighbourId)
    
    currentCityId = nearestNeighbourId

totDist=calcTotalDistanceFast(myPath)
print(totDist[0])
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB01_simpleNN.csv', index=False)
print(len(myPath),myPath[0],myPath[-1])

dataPATH = pd.read_csv('submission_SantaPrime_chthA01_simpleNN.csv')
myPath = dataPATH['Path'].to_list()
X = data['X']
Y = data['Y']
xx=data.loc[myPath].X.to_list()
yy=data.loc[myPath].Y.to_list()
plt.figure(figsize=(15,9))
plt.plot(X,Y, 'ro',markersize=0.05)
plt.plot(xx,yy, color='blue',linewidth=0.28)
#Pair-switch optimizer excluding PRIME condition
dataPATH = pd.read_csv('submission_SantaPrime_chthA01_simpleNN.csv')
myPath = dataPATH['Path'].to_list()
print('without PRIME condition',calcTotalDistanceFast(myPath,False)[0])
print('with    PRIME condition',calcTotalDistanceFast(myPath,True)[0])

pathLenth = calcTotalDistanceFast(myPath,False)
for j in range(4):
    for k in range(1,197768-4,1):
        if k %10000==0:
            pathLenth = calcTotalDistanceFast(myPath,True)[0]
            print(j,k,pathLenth,len(myPath))
        k1=k
        k2=k+1
        k3=k+2
        k4=k+3
        
        if k==1:
            d12 = distance(myPath[k1],myPath[k2])
            d23 = distance(myPath[k2],myPath[k3])
            d34 = distance(myPath[k3],myPath[k4])
            d13 = distance(myPath[k1],myPath[k3])
            d24 = distance(myPath[k2],myPath[k4])
        else:
            d12 = d23
            d23 = d34
            d34 = distance(myPath[k3],myPath[k4])
            d13 = d24
            d24 = distance(myPath[k2],myPath[k4])

        d1 = d12 + d23 + d34
        d2 = d13 + d23 + d24

        if d2 < d1:
            myPath[k2:k4] = myPath[k2:k4][::-1]
        else:
            pass

print('without PRIME condition',calcTotalDistanceFast(myPath,False)[0])
print('with    PRIME condition',calcTotalDistanceFast(myPath,True)[0])

totDist=calcTotalDistanceFast(myPath)
print(totDist[0])
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB02_PairSwitch.csv', index=False)
print(len(myPath),myPath[0],myPath[-1])
def removeCrossLoop(myPath,distMin,distMax,DISTANCEBUFFER=300,OFFSET=0):
    totalGain = 0
    currentPathStats              = calcTotalDistanceFast(myPath,False,distMin,distMax)
    currentPathLength             = currentPathStats[0]
    relevantConnectionStartPoints = currentPathStats[-1]
    #print('distCalc:',currentPathLength,'  Paths: ',len(relevantConnectionStartPoints),'  larger than: ',distMin)
    distThresh = DISTANCEBUFFER
    for point, kB in enumerate(relevantConnectionStartPoints):
        if kB < len(myPath)-1:
            if distMin < distance(myPath[kB],myPath[kB+1]):
                nbrOfPoints=len(relevantConnectionStartPoints)
                x12 = [data.loc[myPath[kB],'X'],data.loc[myPath[kB+1],'X']]
                y12 = [data.loc[myPath[kB],'Y'],data.loc[myPath[kB+1],'Y']]

                relevantCitiyIds = data[(min(x12)-distThresh<data.X) & (data.X<max(x12)+distThresh) & (min(y12)-distThresh<data.Y) & (data.Y<max(y12)+distThresh)]
                relevantPathPoints = np.where(np.isin(np.array(myPath),np.array(relevantCitiyIds.CityId.to_list())))[0]
                #print('kB=',kB,'    length=',distance(myPath[kB],myPath[kB+1]),'    relevantPathPoints: ',len(relevantPathPoints))
                for step,kA in enumerate(relevantPathPoints):
                    prog=round(step*100.0/len(relevantPathPoints),2)

                    if kA < len(myPath)-1 and kA!=kB:
                        d1=distance(myPath[kA],myPath[kA+1]) + distance(myPath[kB]  ,myPath[kB+1])
                        d2=distance(myPath[kA],myPath[kB]  ) + distance(myPath[kA+1],myPath[kB+1])

                        if kA<kB:
                            k2=kB+1
                            k1=kA+1
                        else:
                            k1=kB+1
                            k2=kA+1

                        if d2<d1-OFFSET:
                            gain = d2-d1
                            totalGain = totalGain + gain
                            #print(point,'/',nbrOfPoints,'-',prog,'%','Candidate ',kA,kB,'  :',gain,totalGain,end='\n')
                            myPath[k1:k2] = myPath[k1:k2][::-1]
                        else:
                            pass
        else:
            print(point, 'kB=',kB,' distance too short')
            
    currentPathStats              = calcTotalDistanceFast(myPath,False,distMin,distMax)
    currentPathLength             = currentPathStats[0]
    relevantConnectionStartPoints = currentPathStats[-1]
    print('distCalc:',currentPathLength,'  Paths remaining: ',len(relevantConnectionStartPoints),'  larger than: ',distMin)
    #print('---done---')
    return myPath
dataPATH = pd.read_csv('submission_SantaPrime_chthA02_PairSwitch.csv')
myPath = dataPATH['Path'].to_list()
totDist=calcTotalDistanceFast(myPath,True,4000,10000)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))

myPath=removeCrossLoop(myPath,4000,10000,DISTANCEBUFFER=0,OFFSET=100)
myPath=removeCrossLoop(myPath,1000,10000,DISTANCEBUFFER=200,OFFSET=100)
myPath=removeCrossLoop(myPath,1000,10000,DISTANCEBUFFER=200,OFFSET=100)
myPath=removeCrossLoop(myPath,1000,10000,DISTANCEBUFFER=200,OFFSET=100)
myPath=removeCrossLoop(myPath,1000,10000,DISTANCEBUFFER=200,OFFSET=100)
myPath=removeCrossLoop(myPath,1000,10000,DISTANCEBUFFER=200,OFFSET=100)
myPath=removeCrossLoop(myPath,1000,10000,DISTANCEBUFFER=200,OFFSET=100)

totDist=calcTotalDistanceFast(myPath,True,1000,10000)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop01.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop01.csv')
myPath = dataPATH['Path'].to_list()

myPath=removeCrossLoop(myPath,900,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,900,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,800,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,800,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,700,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,700,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,600,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,600,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,600,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,500,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,500,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,500,10000,DISTANCEBUFFER=200,OFFSET=50)

totDist=calcTotalDistanceFast(myPath,True,500,10000)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop02.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop02.csv')
myPath = dataPATH['Path'].to_list()

myPath=removeCrossLoop(myPath,400,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,400,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,400,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,400,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,300,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,300,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,300,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,300,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,300,10000,DISTANCEBUFFER=200,OFFSET=50)
myPath=removeCrossLoop(myPath,300,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,300,10000,DISTANCEBUFFER=200,OFFSET=20)

totDist=calcTotalDistanceFast(myPath,True,300,10000)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop03.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop03.csv')
myPath = dataPATH['Path'].to_list()

myPath=removeCrossLoop(myPath,250,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,250,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,250,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=200,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,200,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=20)
myPath=removeCrossLoop(myPath,150,10000,DISTANCEBUFFER=100,OFFSET=10)

totDist=calcTotalDistanceFast(myPath,True,150,10000)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop04.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop04.csv')
myPath = dataPATH['Path'].to_list()

for k in range(13):
    myPath=removeCrossLoop(myPath,125,10000,DISTANCEBUFFER=100,OFFSET=20)

for k in range(20):
    myPath=removeCrossLoop(myPath,100,10000,DISTANCEBUFFER=100,OFFSET=0)

totDist=calcTotalDistanceFast(myPath,True,100,10000)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop05.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop05.csv')
myPath = dataPATH['Path'].to_list()

print(sorted([round(distance(myPath[k],myPath[k+1]),2) for k in calcTotalDistanceFast(myPath,includePrimeCondition=True,dMin=150,dMax=10000)[-1]]))
myPath=removeCrossLoop(myPath,150,300,DISTANCEBUFFER=200,OFFSET=0)

print(sorted([round(distance(myPath[k],myPath[k+1]),2) for k in calcTotalDistanceFast(myPath,includePrimeCondition=True,dMin=125,dMax=150)[-1]]))
myPath=removeCrossLoop(myPath,125,150,DISTANCEBUFFER=200,OFFSET=0)

print(sorted([round(distance(myPath[k],myPath[k+1]),2) for k in calcTotalDistanceFast(myPath,includePrimeCondition=True,dMin=100,dMax=125)[-1]]))
myPath=removeCrossLoop(myPath,100,125,DISTANCEBUFFER=200,OFFSET=0)
myPath=removeCrossLoop(myPath,100,125,DISTANCEBUFFER=100,OFFSET=0)
myPath=removeCrossLoop(myPath,100,125,DISTANCEBUFFER=100,OFFSET=0)

for k in range(11):
    myPath=removeCrossLoop(myPath,90,100,DISTANCEBUFFER=100,OFFSET=0)

for k in range(20):
    myPath=removeCrossLoop(myPath,80,90,DISTANCEBUFFER=100,OFFSET=0)

for k in range(17):
    myPath=removeCrossLoop(myPath,70,80,DISTANCEBUFFER=100,OFFSET=0)

for k in range(21):
    myPath=removeCrossLoop(myPath,60,70,DISTANCEBUFFER=100,OFFSET=0)

    
totDist=calcTotalDistanceFast(myPath,True,60,70)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop06.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop06.csv')
myPath = dataPATH['Path'].to_list()

for k in range(30):
    myPath=removeCrossLoop(myPath,50,60,DISTANCEBUFFER=50,OFFSET=0)

totDist=calcTotalDistanceFast(myPath,True,50,60)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop07.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop07.csv')
myPath = dataPATH['Path'].to_list()

for k in range(30):
    myPath=removeCrossLoop(myPath,40,50,DISTANCEBUFFER=20,OFFSET=0)

myPath=removeCrossLoop(myPath,40,50,DISTANCEBUFFER=50,OFFSET=0)
    
totDist=calcTotalDistanceFast(myPath,True,40,50)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop08.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop08.csv')
myPath = dataPATH['Path'].to_list()

for k in range(60):
    myPath=removeCrossLoop(myPath,30,40,DISTANCEBUFFER=10,OFFSET=0)

for k in range(5):
    myPath=removeCrossLoop(myPath,30,40,DISTANCEBUFFER=20,OFFSET=0)

for k in range(60):
    myPath=removeCrossLoop(myPath,20,30,DISTANCEBUFFER=10,OFFSET=0)
    
totDist=calcTotalDistanceFast(myPath,True,20,30)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop09.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop09.csv')
myPath = dataPATH['Path'].to_list()

for k in range(30):
    myPath=removeCrossLoop(myPath,10,20,DISTANCEBUFFER=5,OFFSET=0)

myPath=removeCrossLoop(myPath,10,20,DISTANCEBUFFER=10,OFFSET=0)

totDist=calcTotalDistanceFast(myPath,True,10,20)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop10.csv', index=False)
#Pair-switch optimizer excluding PRIME condition
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop10.csv')
myPath = dataPATH['Path'].to_list()
print('without PRIME condition',calcTotalDistanceFast(myPath,False)[0])
print('with    PRIME condition',calcTotalDistanceFast(myPath,True)[0])

pathLenth = calcTotalDistanceFast(myPath,False)
for j in range(4):
    for k in range(1,197768-4,1):
        if k %10000==0:
            pathLenth = calcTotalDistanceFast(myPath,True)[0]
            print(j,k,pathLenth,len(myPath))
        k1=k
        k2=k+1
        k3=k+2
        k4=k+3
        
        if k==1:
            d12 = distance(myPath[k1],myPath[k2])
            d23 = distance(myPath[k2],myPath[k3])
            d34 = distance(myPath[k3],myPath[k4])
            d13 = distance(myPath[k1],myPath[k3])
            d24 = distance(myPath[k2],myPath[k4])
        else:
            d12 = d23
            d23 = d34
            d34 = distance(myPath[k3],myPath[k4])
            d13 = d24
            d24 = distance(myPath[k2],myPath[k4])

        d1 = d12 + d23 + d34
        d2 = d13 + d23 + d24

        if d2 < d1:
            myPath[k2:k4] = myPath[k2:k4][::-1]
        else:
            pass

print('without PRIME condition',calcTotalDistanceFast(myPath,False)[0])
print('with    PRIME condition',calcTotalDistanceFast(myPath,True)[0])

totDist=calcTotalDistanceFast(myPath)
print(totDist[0])
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthA03_Loop10pair.csv', index=False)
print(len(myPath),myPath[0],myPath[-1])
def removeCrossLoopIndividual(myPath,distMin,distMax,DISTANCEBUFFER=300,OFFSET=0):
    totalGain = 0
    distThresh = DISTANCEBUFFER
    for kB in range(len(myPath)-1):
        if kB%10000==0:
            print(round(totalGain,0),end=' ')
        x12 = [data.loc[myPath[kB],'X'],data.loc[myPath[kB+1],'X']]
        y12 = [data.loc[myPath[kB],'Y'],data.loc[myPath[kB+1],'Y']]

        relevantCitiyIds = data[(min(x12)-distThresh<data.X) & (data.X<max(x12)+distThresh) & (min(y12)-distThresh<data.Y) & (data.Y<max(y12)+distThresh)]
        relevantPathPoints = np.where(np.isin(np.array(myPath),np.array(relevantCitiyIds.CityId.to_list())))[0]
        #print('kB=',kB,'    length=',distance(myPath[kB],myPath[kB+1]),'    relevantPathPoints: ',len(relevantPathPoints))
        for step,kA in enumerate(relevantPathPoints):

            if kA < len(myPath)-1 and kA!=kB:
                d1=distance(myPath[kA],myPath[kA+1]) + distance(myPath[kB]  ,myPath[kB+1])
                d2=distance(myPath[kA],myPath[kB]  ) + distance(myPath[kA+1],myPath[kB+1])

                if kA<kB:
                    k2=kB+1
                    k1=kA+1
                else:
                    k1=kB+1
                    k2=kA+1

                if d2<d1-OFFSET:
                    gain = d2-d1
                    totalGain = totalGain + gain
                    #print(point,'/',nbrOfPoints,'-',prog,'%','Candidate ',kA,kB,'  :',gain,totalGain,end='\n')
                    myPath[k1:k2] = myPath[k1:k2][::-1]
                else:
                    pass
            
    currentPathStats              = calcTotalDistanceFast(myPath,False)
    currentPathLength             = currentPathStats[0]
    print('\n distCalc:',currentPathLength)
    #print('---done---')
    return myPath
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop10pair.csv')
myPath = np.array(dataPATH['Path'].to_list())

myPath=removeCrossLoopIndividual(myPath,0,10000,0,0)
myPath=removeCrossLoopIndividual(myPath,0,10000,0,0)
myPath=removeCrossLoopIndividual(myPath,0,10000,0,0)
myPath=removeCrossLoopIndividual(myPath,0,10000,5,0)
myPath=removeCrossLoopIndividual(myPath,0,10000,5,0)
myPath=removeCrossLoopIndividual(myPath,0,10000,5,0)
myPath=removeCrossLoopIndividual(myPath,0,10000,10,0)

totDist=calcTotalDistanceFast(myPath,True,20,30)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop11.csv', index=False)
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop11.csv')
myPath = np.array(dataPATH['Path'].to_list())
totDist=calcTotalDistanceFast(myPath,True,100,150)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))

myPath=removeCrossLoop(myPath,10,20,10,0)
myPath=removeCrossLoop(myPath,10,20,20,0)
myPath=removeCrossLoop(myPath,10,20,30,0)

totDist=calcTotalDistanceFast(myPath,True,200,10000)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop12.csv', index=False)
totDist=calcTotalDistanceFast(myPath,True,200,10000)
print('with    PRIME condition: ',totDist[0],len(totDist[-1]))
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission_SantaPrime_chthB03_Loop12.csv', index=False)
pd.DataFrame(totDist[1],columns=['Path']).to_csv('submission.csv', index=False)



dataPATH = pd.read_csv('submission_SantaPrime_chthA01_simpleNN.csv')
myPath = np.array(dataPATH['Path'].to_list())
X1=data.loc[myPath].X.to_list()
Y1=data.loc[myPath].Y.to_list()

dataPATH = pd.read_csv('submission_SantaPrime_chthA02_PairSwitch.csv')
myPath = np.array(dataPATH['Path'].to_list())
X2=data.loc[myPath].X.to_list()
Y2=data.loc[myPath].Y.to_list()

dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop01.csv')
myPath = np.array(dataPATH['Path'].to_list())
X3=data.loc[myPath].X.to_list()
Y3=data.loc[myPath].Y.to_list()

dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop02.csv')
myPath = np.array(dataPATH['Path'].to_list())
X4=data.loc[myPath].X.to_list()
Y4=data.loc[myPath].Y.to_list()

dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop03.csv')
myPath = np.array(dataPATH['Path'].to_list())
X5=data.loc[myPath].X.to_list()
Y5=data.loc[myPath].Y.to_list()

dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop04.csv')
myPath = np.array(dataPATH['Path'].to_list())
X6=data.loc[myPath].X.to_list()
Y6=data.loc[myPath].Y.to_list()

dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop05.csv')
myPath = np.array(dataPATH['Path'].to_list())
X7=data.loc[myPath].X.to_list()
Y7=data.loc[myPath].Y.to_list()

dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop11.csv')
myPath = np.array(dataPATH['Path'].to_list())
X8=data.loc[myPath].X.to_list()
Y8=data.loc[myPath].Y.to_list()

plt.figure(figsize=(14,20))
plt.subplot(421)
plt.plot(X1,Y1,linewidth=0.32)
plt.subplot(422)
plt.plot(X2,Y2,linewidth=0.32)
plt.subplot(423)
plt.plot(X3,Y3,linewidth=0.32)
plt.subplot(424)
plt.plot(X4,Y4,linewidth=0.32)
plt.subplot(425)
plt.plot(X5,Y5,linewidth=0.32)
plt.subplot(426)
plt.plot(X6,Y6,linewidth=0.32)
plt.subplot(427)
plt.plot(X7,Y7,linewidth=0.32)
plt.subplot(428)
plt.plot(X8,Y8,linewidth=0.32)
#dataPATH = pd.read_csv('submission_SantaPrime_chthA01_simpleNN.csv')
#dataPATH = pd.read_csv('submission_SantaPrime_chthA02_PairSwitch.csv')
#dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop01.csv')
#dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop02.csv')
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop11.csv')

myPath = np.array(dataPATH['Path'].to_list())
X = data['X']
Y = data['Y']
N1, N2= [0,197770]
XPATH = data.loc[dataPATH['Path'][N1:N2]]['X'].to_list()
YPATH = data.loc[dataPATH['Path'][N1:N2]]['Y'].to_list()
    
    
plt.figure(figsize=(15,9))
plt.plot(XPATH,YPATH,linewidth=0.82)

d=calcTotalDistanceFast(myPath)
originOfLongs = d[-1]
#for k in originOfLongs:
#    id1, x1, y1 = data.loc[myPath[k]]
#    id2, x2, y2 = data.loc[myPath[k+1]]
#    plt.plot([x1,x2],[y1,y2],linewidth=0.82,color='red')

k=197768
id1, x1, y1 = data.loc[myPath[k]]
id2, x2, y2 = data.loc[myPath[k+1]]
print(x1,x2,y1,y2)
plt.plot([x1,x2],[y1,y2],'bo',markersize=0.82,color='red')
import matplotlib
from matplotlib import cm
cmap=cm.gist_rainbow

dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop12.csv')
myPath1 = np.array(dataPATH['Path'].to_list())
dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop11.csv')
myPath2 = np.array(dataPATH['Path'].to_list())

n = len(myPath1)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,7))
for i in range(201):
    ind = myPath1[n//200*i:min(n, n//200*(i+1)+1)]
    ax1.plot(data.loc[ind].X.to_list(), data.loc[ind].Y.to_list(), color=cmap(i/200.0), linewidth=1)
    ax1.autoscale(tight=True)

for i in range(201):
    ind = myPath2[n//200*i:min(n, n//200*(i+1)+1)]
    ax2.plot(data.loc[ind].X.to_list(), data.loc[ind].Y.to_list(), color=cmap(i/200.0), linewidth=1)
    ax2.autoscale(tight=True)

import matplotlib
from matplotlib import cm
cmap=cm.gist_rainbow

dataPATH = pd.read_csv('submission_SantaPrime_chthA03_Loop12.csv')
myPath1 = np.array(dataPATH['Path'].to_list())

n = len(myPath1)
fig, ax1 = plt.subplots(figsize=(10,7))
for i in range(201):
    ind = myPath1[n//200*i:min(n, n//200*(i+1)+1)]
    ax1.plot(data.loc[ind].X.to_list(), data.loc[ind].Y.to_list(), color='grey', linewidth=1)
    ax1.autoscale(tight=True)
    if i == 170:
        ax1.plot(data.loc[ind].X.to_list(), data.loc[ind].Y.to_list(), color=cmap(i/200.0), linewidth=1)
    
