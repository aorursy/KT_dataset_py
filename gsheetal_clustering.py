import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

import math
import random
import time
from tkinter import *

Iris_dataset=pd.read_csv('..../Iris.csv')
Flower_type = Iris_dataset['Species'].values.tolist()
Iris_dataset.dropna()

FlowerDb=Iris_dataset.drop(['Species'], axis=1)
FlowerDb

F_arrange=FlowerDb.values.tolist()
F_arrange

def euclidean(centroid, FlowerDb):
    Eucli_Sum=[]
    if len(centroid) == len(FlowerDb)+1:
        centroid=centroid[1:]
    for i in range (0, len(FlowerDb)):
        Eucli_Sum.append((centroid[i]- FlowerDb[i])**2)
    euclidean= math.sqrt(sum(Eucli_Sum))
    return euclidean

def jaccard(centroid, FlowerDb):
    Jacc_Min=[]
    Jacc_Max = []
    if len(centroid) == 5:
        centroid = centroid[1:]
    for i in range(len(FlowerDb)):
        Jacc_Min.append(min(centroid[i],FlowerDb[i]))
        Jacc_Max.append(max(centroid[i],FlowerDb[i]))
    return 1-(sum(Jacc_Min)/sum(Jacc_Max))

def cosine(centroid, FlowerDb):
    if len(centroid) == 5:
        centroid = centroid[1:]
    a1 = centroid
    a1 = np.array(a1)
    a1 = a1.reshape(1,-1)
    b1 = np.array(FlowerDb)
    b1 = b1.reshape(1,-1)
    ans = cosine_similarity(a1,b1)
    return 1-ans[0][0]

def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    seq = fileHandler.readlines()
    fileHandler.close()
    del seq[0] 
    Db = []
    for m_int in seq:
        instance = lineToTuple(m_int)
        Db.append(instance)
    return Db

def stringsToNumbers(FlowerList):
    for i in range(len(FlowerList)):
        if (isValidNumberString(FlowerList[i])):
            FlowerList[i] = float(FlowerList[i])

def distance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumOfSquares = 0
    for i in range(0, len(instance1)-1):
        sumOfSquares += (instance1[i+1] - instance2[i])**2
    return sumOfSquares

def meanInstance(name, instanceList):
    numinst = len(instanceList)
    if (numinst == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes)
    for instance in instanceList:
        for i in range(0, numAttributes):
            means[i+1] += instance[i]
    for i in range(0, numAttributes):
        means[i+1] /= float(numinst)
    return tuple(means)

def assign(instance, center_pts, distance):
    minDistance = distance(center_pts[0],instance)
    minDistanceIndex = 0
    for i in range(1, len(center_pts)):
        dist_formula = distance(center_pts[i], instance )
        if (dist_formula < minDistance):
            minDistance = dist_formula
            minDistanceIndex = i
    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    FlowerList = []
    for i in range(numSubLists):
        FlowerList.append([])
    return FlowerList

def assignAll(inst, center_pts, Flower_type, distance):
    groups = createEmptyListOfLists(len(center_pts))
    classgroups = createEmptyListOfLists(len(center_pts))
    i=0
    for instance in inst:
        groupIndex = assign(instance, center_pts, distance)
        groups[groupIndex].append(instance)
        classgroups[groupIndex].append(Flower_type[i])
        i=i+1
    return groups, classgroups

def computeCentroids(groups):
    center_pts = []
    for i in range(len(groups)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, groups[i])
        center_pts.append(centroid)
    return center_pts

def kmeans(inst, k, distance,Flower_type, animation=False, initCentroids=None):
    response = {}
    if (initCentroids == None or len(initCentroids) < k):
        random.seed(time.time())
        center_pts = random.sample(inst, k)
    else:
        center_pts = initCentroids
    cent_pt = []
    if animation:
        dly = 1.0 
        groups = createEmptyListOfLists(k)
        groups[0] = inst
    itr = 0
    while (center_pts != cent_pt):
        itr += 1
        groups, required_Flower_type = assignAll(inst, center_pts, Flower_type, distance)
        cent_pt = center_pts
        center_pts = computeCentroids(groups)
        withinss = computeWithinss(groups, center_pts, distance)
    print('Iterations are ', itr)
    response["groups"] = groups
    response["center_pts"] = center_pts
    response["withinss"] = withinss
    response["Flower_type"] = required_Flower_type
    return response

def computeWithinss(groups, center_pts, distance):
    response = 0
    for i in range(len(center_pts)):
        centroid = center_pts[i]
        group = groups[i]
        for instance in group:
            response += distance(centroid, instance)
    return response

def repeatedKMeans(inst, k, n):
    bestgrouping = {}
    bestgrouping["withinss"] = float("inf")
    for i in range(1, n+1):
        print ("k-means trial %d," % i )
        trialgrouping = kmeans(inst, k)
        print ("withinss: %.1f" % trialgrouping["withinss"])
        if trialgrouping["withinss"] < bestgrouping["withinss"]:
            bestgrouping = trialgrouping
            minWithinssTrial = i
    print("Minimum withinss:", minWithinssTrial)
    return bestgrouping

def printTable(inst):
    for instance in inst:
        if instance != None:
            m_int = instance[0] + "\t"
            for i in range(1, len(instance)):
                m_int += "%.2f " % instance[i]
            print(m_int)

def extractAttribute(inst, index):
    response = []
    for instance in inst:
        response.append(instance[index])
    return response

def connectPoints(clust, inst1, inst2, color):
    width = clust.winfo_reqwidth()
    height = clust.winfo_reqheight()
    margin = clust.FlowerDb["margin"]
    minX = clust.FlowerDb["minX"]
    minY = clust.FlowerDb["minY"]
    maxX = clust.FlowerDb["maxX"]
    maxY = clust.FlowerDb["maxY"]
    scaleX = float(width - 2*margin) / (maxX - minX)
    scaleY = float(height - 2*margin) / (maxY - minY)
    for p1 in inst1:
        for p2 in inst2:
            a1 = margin + (p1[1]-minX)*scaleX
            b1 = height - margin - (p1[2]-minY)*scaleY
            x2 = margin + (p2[1]-minX)*scaleX
            y2 = height - margin - (p2[2]-minY)*scaleY
            clust.create_line(a1, b1, x2, y2, fill=color)
    clust.update()

def mergegroups(groups):
    response = []
    for group in groups:
        response.extend(group)
    return response

def prepareWindow(inst):
    width = 500
    height = 500
    margin = 50
    root = Tk()
    clust = clust(root, width=width, height=height, background="white")
    clust.pack()
    clust.FlowerDb = {}
    clust.FlowerDb["margin"] = margin
    setBounds2D(clust, inst)
    paintAxes(clust)
    clust.update()
    return clust

def setBounds2D(clust, inst):
    attributeX = extractAttribute(inst, 1)
    attributeY = extractAttribute(inst, 2)
    clust.FlowerDb["minX"] = min(attributeX)
    clust.FlowerDb["minY"] = min(attributeY)
    clust.FlowerDb["maxX"] = max(attributeX)
    clust.FlowerDb["maxY"] = max(attributeY)

def paintAxes(clust):
    width = clust.winfo_reqwidth()
    height = clust.winfo_reqheight()
    margin = clust.FlowerDb["margin"]
    minX = clust.FlowerDb["minX"]
    minY = clust.FlowerDb["minY"]
    maxX = clust.FlowerDb["maxX"]
    maxY = clust.FlowerDb["maxY"]
    clust.create_line(margin/2, height-margin/2, width-5, height-margin/2,
                       width=2, arrow=LAST)
    clust.create_text(margin, height-margin/4,
                       text=str(minX), font="Sans 11")
    clust.create_text(width-margin, height-margin/4,
                       text=str(maxX), font="Sans 11")
    clust.create_line(margin/2, height-margin/2, margin/2, 5,
                       width=2, arrow=LAST)
    clust.create_text(margin/4, height-margin,
                       text=str(minY), font="Sans 11", anchor=W)
    clust.create_text(margin/4, margin,
                       text=str(maxY), font="Sans 11", anchor=W)
    clust.update()


def showDataset2D(inst):
    clust = prepareWindow(inst)
    paintDataset2D(clust, inst)

def paintDataset2D(clust, inst):
    clust.delete(ALL)
    paintAxes(clust)
    drawPoints(clust, inst, "blue", "circle")
    clust.update()

def showgroups2D(groupingDictionary):
    groups = groupingDictionary["groups"]
    center_pts = groupingDictionary["center_pts"]
    withinss = groupingDictionary["withinss"]
    clust = prepareWindow(mergegroups(groups))
    paintgroups2D(clust, groups, center_pts,
                    "Withinss: %.1f" % withinss)

def paintgroups2D(clust, groups, center_pts, title=""):
    clust.delete(ALL)
    paintAxes(clust)
    colors = ["blue", "red", "green", "brown", "purple", "orange"]
    for groupIndex in range(len(groups)):
        color = colors[groupIndex%len(colors)]
        inst = groups[groupIndex]
        centroid = center_pts[groupIndex]
        drawPoints(clust, inst, color, "circle")
        if (centroid != None):
            drawPoints(clust, [centroid], color, "square")
        connectPoints(clust, [centroid], inst, color)
    width = clust.winfo_reqwidth()
    clust.create_text(width/2, 20, text=title, font="Sans 14")
    clust.update()
    
grouping = kmeans(F_arrange, 3, cosine,Flower_type, True)
printTable(grouping["center_pts"])
import statistics 
from statistics import mode 
def sse(grouping, center_pts):
    i=0
    SSE=0
    for clust in grouping:
        for FlowerDb in clust:
            SSE=SSE+distance(center_pts[i], FlowerDb)
        i=i+1
    return SSE
def accuracy(grouping):
    correct_pred = 0
    for clust in grouping['Flower_type']:
        for label in clust:
            if label == mode(clust):
                correct_pred+=1
    return correct_pred/150

print('SSE value for Cosine Distance is ',sse(grouping['groups'],grouping['center_pts']))
print("Accuracy of Cosine Distance is ",accuracy(grouping))

grouping = kmeans(F_arrange, 3, jaccard,Flower_type, True)
printTable(grouping["center_pts"])

print('SSE value for Jaccard Distance is ', sse(grouping['groups'],grouping['center_pts']))
print("Accuracy of Jaccard Distance is ", accuracy(grouping))

grouping = kmeans(F_arrange, 3, euclidean,Flower_type, True)
printTable(grouping["center_pts"])

print('SSE value for Euclidean Distance is ', sse(grouping['groups'],grouping['center_pts']))
print("Accuracy of Euclidean Distance is ", accuracy(grouping))

def kmeans_sse(inst, k, distance,Flower_type, animation=False, initCentroids=None):
    response = {}
    if (initCentroids == None or len(initCentroids) < k):
        random.seed(time.time())
        center_pts = random.sample(inst, k)
    else:
        center_pts = initCentroids
    cent_pt = [] 
    new_sse= 99999 
    prev_sse=1000000 
    if animation:
        dly = 1.0 
        groups = createEmptyListOfLists(k)
        groups[0] = inst
    itr = 0
    while (new_sse < prev_sse):
        itr += 1
        groups, required_Flower_type = assignAll(inst, center_pts, Flower_type, distance)
        prev_sse = new_sse
        center_pts = computeCentroids(groups)
        new_sse= sse(groups, center_pts)
        withinss = computeWithinss(groups, center_pts, distance)
    print('Iterations are ', itr)
    print('Old SSE value', prev_sse)
    print('New SSE value', new_sse)
    response["groups"] = groups
    response["center_pts"] = center_pts
    response["withinss"] = withinss
    response["Flower_type"] = required_Flower_type
    return response
before= datetime.now()
grouping = kmeans_sse(F_arrange, 3, cosine,Flower_type, True)
after= datetime.now()

printTable(grouping["center_pts"])
print("Time Delay ", after- before)
print('SSE value for Iterated Cosine Distance is ', sse(grouping['groups'],grouping['center_pts']))
print("Accuracy of Iterated Cosine Distance is ",accuracy(grouping))
before= datetime.now()

grouping = kmeans_sse(F_arrange, 3, jaccard,Flower_type, True)
after= datetime.now()
printTable(grouping["center_pts"])
print("Time Delay ", after- before)
print('SSE value for Iterated Jaccard Distance is ',sse(grouping['groups'],grouping['center_pts']))
print("Accuracy of Iterated Jaccard Distance is ", accuracy(grouping))
before= datetime.now()

grouping = kmeans_sse(F_arrange, 3, euclidean,Flower_type, True)
after= datetime.now()
printTable(grouping["center_pts"])
print("Time Delay", after- before)
print("SEE Value of Iterated Euclidean Distance ",sse(grouping['groups'],grouping['center_pts']))
print("Accuracy Value of Iterated Euclidean Distance ", accuracy(grouping))

def kmeans_100(inst, k, distance,Flower_type, animation=False, initCentroids=None):
    response = {}
    if (initCentroids == None or len(initCentroids) < k):
        random.seed(time.time())
        center_pts = random.sample(inst, k)
    else:
        center_pts = initCentroids
    cent_pt = []
    if animation:
        dly = 1.0 
        groups = createEmptyListOfLists(k)
        groups[0] = inst
    itr = 0
    while (itr<100):
        itr +=1
        groups, required_Flower_type = assignAll(inst, center_pts, Flower_type, distance)
        cent_pt = center_pts
        center_pts = computeCentroids(groups)
        withinss = computeWithinss(groups, center_pts, distance)
    print('Iterations are ', itr)
    response["groups"] = groups
    response["center_pts"] = center_pts
    response["withinss"] = withinss
    response["Flower_type"] = required_Flower_type
    return response
before= datetime.now()
grouping = kmeans_100(F_arrange, 3, cosine,Flower_type, True)
after= datetime.now()
printTable(grouping["center_pts"])
print("Time Delay ", after- before)
print("SSE Value of MSV(100)Iterated Cosine Distance ",sse(grouping['groups'],grouping['center_pts']))
print("Accuracy Value of MSV(100)Iterated Cosine Distance ", accuracy(grouping))
before= datetime.now()

grouping = kmeans_100(F_arrange, 3, jaccard,Flower_type, True)
after= datetime.now()
printTable(grouping["center_pts"])
print("Time Delay ", after- before)
print("SSE Value of MSV(100)Iterated Jaccard Distance", sse(grouping['groups'],grouping['center_pts']))
print("Accuracy Value of MSV(100)Iterated Jaccard Distance", accuracy(grouping))
before= datetime.now()

grouping = kmeans_100(F_arrange, 3, euclidean,Flower_type, True)
after= datetime.now()
printTable(grouping["center_pts"])
print("Time Delay ", after- before)
print("SEE Value of MSV(100)Iterated Euclidean Distance ",sse(grouping['groups'],grouping['center_pts']))
print("Accuracy Value of MSV(100)Iterated Euclidean Distance ",accuracy(grouping))
