# imports and test of python
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mpl_toolkits.mplot3d import Axes3D
from statistics import mean
from scipy import spatial

print("Python works")
#Klasse für Hilfsfunktionen
class Pointclouds():
    def __init__(self):
        pass

    def createPointcloud(self, roomDimension, numberOfPoints):
        lenX, lenY, lenZ = roomDimension
        
        points = np.column_stack((random.uniform(0.0, lenX, numberOfPoints),random.uniform(0.0, lenY, numberOfPoints),random.uniform(0.0, lenZ, numberOfPoints)))
        
        return points

    def addError(self, points, sigma, distribution = 'normal'):
        # adding some error to the referenc Pointcloud to simulate a second measurement with a less accurate scanner
        if distribution == 'normal':
            otherpoints = points + random.normal(0.0, sigma, np.shape(points))
        elif distribution == 'uniform':
            otherpoints = points + random.uniform(-sigma, sigma, np.shape(points))
        return otherpoints

    def addUnevenError(self, points, sigmaX,sigmaY,sigmaZ):
        n = np.shape(points)[0]
        noise = np.column_stack((random.normal(0.0, sigmaX, n),random.normal(0.0, sigmaY, n),random.normal(0.0, sigmaZ, n)))
        otherpoints = points + noise
        return otherpoints

    def addErrorWithOffset(self, points, sigma, offset):
        otherpoints = points + random.normal(offset, sigma, np.shape(points))
        return otherpoints

    def calcErrorDiff(self, PointcloudA, PointcloudB): #return RMSEdiffRelative
        # Calculate RMSE with both methods
        # direkt method
        ## rigid body transform is not needed, because in this simulations the coordinate systems are the same for both scans
        errorVectors = PointcloudA - PointcloudB                #distances of the coresponding points
        #calc length of error vector
        errorDistances = np.linalg.norm(errorVectors, axis=1)              

        RMSE_direct = np.sqrt(mean(pow(errorDistances,2)))                      

        # indirekt method
        distancesA = spatial.distance.pdist(PointcloudA)
        distancesB = spatial.distance.pdist(PointcloudB)
        distErrors =  distancesB - distancesA
        RMSE_indirect =np.sqrt(mean(pow(distErrors,2)))

        RMSEdiffRelative = (RMSE_indirect - RMSE_direct )/RMSE_direct  # the RMSE_indirect is (RMSEdiffRelative * 100) % bigger than the RMSE_direct
        return [RMSEdiffRelative , RMSE_direct, RMSE_indirect]                                        # or return [RMSEdiffRelative , RMSE_direct, RMSE_indirect] 

def rmse(individualErrors):
    return np.sqrt(mean(individualErrors ** 2))

print("Hilfsklasse erstellt")
    

pts = Pointclouds() #objekt for helpfunctions
numberOfPoints = 1000 #may be different for different szenarios
################ Validation Test
 #   - 10m3Raum
#       - normalverteilter Fehler 50mm  np.random.normal()
#       - das gleiche nochmal, kommt das gleiche Ergebnis heraus

numberOfPoints = 200
error = 50    #mm   standardabweichung des normalverteilten Fehlers
#errors = [50 for _ in range(1, 102, 10)]
rounds = [x for x in range(1, 101, 1)]  # 100 trys
diffs = []
for round in rounds: 
    points = pts.createPointcloud((10_000.0, 10_000.0, 10_000.0), numberOfPoints)
    otherPoints = pts.addError(points, error, distribution = 'normal')
    diffs.append(pts.calcErrorDiff(points, otherPoints))
diffs = np.array(diffs)

plt.figure()
plt.scatter(rounds, diffs[:, 0])
plt.title("100 times with normal distribution")
plt.xlabel('rounds')
plt.ylabel('diff of the to methods')
plt.show()
print("averageRMSE_direct:   ", np.mean(diffs[:, 1]))
print("averageRMSE_indirect: ", np.mean(diffs[:, 2]))
numberOfValues = 100_000
standardabweichung = 50

rmse(np.zeros(numberOfValues) + random.normal(0.0, standardabweichung, numberOfValues))
numberOfValues = 100_000
standardabweichung = 50
valuesA = np.zeros(numberOfValues) + random.normal(0.0, standardabweichung, numberOfValues)
valuesB = np.zeros(numberOfValues) + random.normal(0.0, standardabweichung, numberOfValues)
rmse(valuesA + valuesB)    # hier ist es egal, ob (valuesA + valuesB) oder (valuesA - valuesB)
################ First Test
#   - 10m3Raum
#   - normalverteilter Fehler mit variierender Standardabweichung von 1mm bis 100 mm

numberOfPoints = 200 
errors = [x for x in range(1, 101, 1)] #mm   standardabweichung des normalverteilten Fehlers
diffs = []
for round in rounds: 
    points = pts.createPointcloud((10_000.0, 10_000.0, 10_000.0), numberOfPoints)
    otherPoints = pts.addError(points, error, distribution = 'normal')
    diffs.append(pts.calcErrorDiff(points, otherPoints))
diffs = np.array(diffs)

plt.figure()
plt.scatter(errors, diffs[:, 0])
plt.title("100 times with different sigma")
plt.xlabel('sigma in mm')
plt.ylabel('diff of the to methods')
plt.show()
################ Second Test
#   - Punkte nur auf XY-Ebene 10m^2
#   - normalverteilter Fehler mit Standardabweichung 50mm

numberOfPoints = 200
error = 50    #mm   standardabweichung des normalverteilten Fehlers
rounds = [x for x in range(1, 101, 1)]  # 100 rounds
diffs = []
for round in rounds: 
    points = pts.createPointcloud((10_000.0, 10_000.0, 0), numberOfPoints)
    otherPoints = pts.addError(points, error, distribution = 'normal')
    diffs.append(pts.calcErrorDiff(points, otherPoints))
diffs = np.array(diffs)

plt.figure()
plt.scatter(rounds, diffs[:, 0])
plt.title("100 times with points on XY-Plane")
plt.xlabel('rounds')
plt.ylabel('diff of the to methods')
plt.show()
################ Third Test
#   - 10m^3 Raum
#   - normalverteilter Fehler mit Standardabweichung 50mm
#   - Fehler in z-Richtung "faktor" mal größer als der Fehler in x und y Richtung.

numberOfPoints = 500
error = 50    #mm   standardabweichung des normalverteilten Fehlers
faktors = [x for x in np.arange(0.1,5,0.1)]
diffs = []
for faktor in faktors: 
    points = pts.createPointcloud((10_000.0, 10_000.0, 0), numberOfPoints)
    otherPoints = pts.addUnevenError(points, error, error, error*faktor)
    diffs.append(pts.calcErrorDiff(points, otherPoints))
diffs = np.array(diffs)

plt.figure()
plt.scatter(faktors, diffs[:, 0])
plt.title("uneven Error in z direction")
plt.xlabel('faktor')
plt.ylabel('diff of the to methods')
plt.show()
print("RMSE_direct   with faktor 5: ", diffs[-1, 1])
print("RMSE_indirect with faktor 5: ", diffs[-1, 2])
############### Fourth Test

#   - Punkte nur in XY ebene Verteilt 10m2
#     - Fehler nur in x Richtung
#     - Fehler nur in y Richtung
#     - Fehler nur in z Richtung

numberOfPoints = 1000
plt.figure()
error = 50
diffs = []
faktors = range(1, 10, 1)

for faktor in faktors: 
    points = pts.createPointcloud((10_000.0, 10_000.0, 0.0), numberOfPoints)
    otherPoints = pts.addUnevenError(points, error*faktor, error, error)
    diffs.append(pts.calcErrorDiff(points, otherPoints))
plt.scatter(faktors, np.array(diffs)[:,0], label = '#X', color = 'navy')

diffs = []
for faktor in faktors: 
    points = pts.createPointcloud((10_000.0, 10_000.0, 0.0), numberOfPoints)
    otherPoints = pts.addUnevenError(points, error, error*faktor, error)
    diffs.append(pts.calcErrorDiff(points, otherPoints))
plt.scatter(faktors, np.array(diffs)[:,0], label = '#Y', color = 'red')

diffs = []
for faktor in faktors: 
    points = pts.createPointcloud((10_000.0, 10_000.0, 0.0), numberOfPoints)
    otherPoints = pts.addUnevenError(points, error, error, error*faktor)
    diffs.append(pts.calcErrorDiff(points, otherPoints))
plt.scatter(faktors, np.array(diffs)[:,0], label = '#Z', color = 'darkcyan')


plt.scatter(faktors, np.array(diffs)[:,0])
plt.title("1000 Points on XY Plane")
plt.xlabel('error in # axes is _ times bigger than other')
plt.ylabel(r'distRMSE is y% smaller than RMSE')
plt.legend(loc = "lower left")
plt.show()