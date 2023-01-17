# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import time

import heapq as hq

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 100000



import os

print(os.listdir("../input/roadcracks")) #Liste fichiers



"""Importation de l'image et conversion en array"""

from PIL import Image

img = Image.open( "../input/roadcracks/fissure2.PNG")

img.load()

data = np.asarray( img, dtype="int32" )



"""Affichage Image"""

plt.imshow(img)

plt.show()



#Conversion en tableau 2D (moyenne des valeurs RGB)

data = np.sum(data, axis = 2) / 3 

plt.gray()

#Image en nuance

#sns.heatmap(data = data)
def displayImage(array):

    im = Image.fromarray(array)

    plt.gray()

    plt.imshow(im)
#Pooling

import skimage.measure

data = skimage.measure.block_reduce(data, (2,2), np.max)

print(data.shape)
fig, ax = plt.subplots()

ax.imshow(data)



perspective = [

(0, 200, 214, 200),

(0, 150, 214, 150),

(0, 110, 214, 110),

(0, 75, 214, 75),

(0, 40, 214, 40),

    

(5, 0, 0, 213),

(50, 0, 43, 213),

(100, 0, 100, 213),

(145, 0, 150, 213),

(190, 0, 200, 213),

]



for l1, c1, l2, c2 in perspective:

    ax.plot([l1, l2], [c1, c2], linewidth = 1.5, c = 'r')



plt.show()

fig, ax = plt.subplots()

ax.imshow(data)

tailleCase = 5

    

intensities = []

positions = []



ligPlot = []

colPlot = []



for lig in range(data.shape[0] // tailleCase):

    for col in range(data.shape[1] // tailleCase):

        posLig, posCol = lig*tailleCase, col*tailleCase

        mini = data[posLig:posLig+tailleCase,posCol:posCol+tailleCase].min()

        reponse = np.where(data[posLig:posLig+tailleCase,posCol:posCol+tailleCase] == mini)

        

        rep = list(zip(reponse[0], reponse[1]))

        for dLig, dCol in rep:

            ligPlot.append(posLig + dLig)

            colPlot.append(posCol + dCol)

            

            positions.append((posLig + dLig, posCol + dCol))

            intensities.append(mini)



ax.scatter(colPlot, ligPlot, s = 1, color = 'r')

plt.show()
print("Before point selection", len(positions))

threshold = np.mean(intensities) - np.std(intensities)



for index in range(len(intensities) -1, -1, -1):

    if intensities[index] >= threshold: 

        del intensities[index]

        del positions[index]



print("Afterpoint selection", len(positions))
start = time.time() 



ligPlot = [i[0] for i in positions]

colPlot = [i[1] for i in positions]



fig, ax = plt.subplots()

ax.imshow(data)

ax.scatter(colPlot, ligPlot, s = 1, color = 'r')

plt.show()



print(time.time() - start)
colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate', 'deepskyblue', 'lime', 'orange']

#print("Positions number:", len(positions))



pathIntensities = []

paths = []

ligPlot, colPlot = [], []



for indexPos, (startLig, startCol) in enumerate(positions):

    ax.scatter(startCol, startLig, s = 10, color = 'r')

    vu = 0

    copyPositions = [(lig, col) for lig, col in positions]

    

    print("Process:", indexPos+1, "/", len(positions), "-", indexPos/len(positions)*100, "%")

    

    queue = [(data[startLig][startCol], startLig, startCol)]

    distance = [ [float("inf") for i in range(data.shape[1])] for a in range(data.shape[0])]

    predecessor = [ [(-1,-1) for i in range(data.shape[1])] for a in range(data.shape[0])]



    while queue and vu < 20 : #!= len(positions) :

        intensity, lig, col = hq.heappop(queue)

        if intensity > distance[lig][col]: continue

        

        if (lig,col) in copyPositions:

            vu += 1

            copyPositions.remove((lig,col))

        

        for dLig, dCol in [(1,0),(-1,0),(0,1),(0,-1)]:

            newLig, newCol = lig + dLig, col + dCol

            if not(0 <= newLig < data.shape[0] and 0 <= newCol < data.shape[1]): continue



            newIntensity = intensity + data[newLig][newCol]

            if distance[newLig][newCol] > newIntensity:

                distance[newLig][newCol] = newIntensity

                predecessor[newLig][newCol] = (lig, col)

                hq.heappush(queue, (newIntensity, newLig, newCol))

    

    for endLig, endCol in positions[indexPos+1:]:

        l, c = endLig, endCol

        

        if distance[l][c] == float("inf"): continue

        

        curLig = []

        curCol = []

        

        while (l,c) != (startLig, startCol):

            curLig.append(l)

            curCol.append(c)

            

            nL, nC = predecessor[l][c]

            l,c = nL, nC 

        

        ligPlot.extend(curLig)

        colPlot.extend(curCol)

        

        paths.append((curLig, curCol))

        pathIntensities.append(distance[endLig][endCol])



        

fig, ax = plt.subplots()

ax.imshow(data)

ax.scatter(colPlot, ligPlot, s = 0.2, c = 'r')

plt.show()
print("Before path selection", len(paths))



threshold = np.mean(pathIntensities) #- np.std(pathIntensities)



for index in range(len(paths) -1, -1, -1):

    if len(paths[index]) < 5 and pathIntensities[index] > threshold: 

        del pathIntensities[index]

        del paths[index]



print("Afterpoint selection", len(paths))
ligPlot, colPlot = [], []

for lig,col in paths:

    ligPlot.extend(lig)

    colPlot.extend(col)



fig, ax = plt.subplots()

ax.imshow(data)

ax.scatter(colPlot, ligPlot, s = 0.01, color = 'r')



plt.show()
"""A PARTIR DE CE BLOC IL N Y A QUE DES TESTS"""



"""Détection de bords avec contraste"""



data2 = np.zeros((data.shape[0]-2, data.shape[1] - 2))



distance = lambda x,y,a,b: np.sqrt(np.square(x-a) + np.square(y-b))



data2.shape



for i in range(1, data.shape[0]-1):

    for j in range(1, data.shape[1]-1):

        data2[i-1][j-1] = distance(data[i][j-1], data[i][j+1], data[i-1][j], data[i+1][j])





displayImage(data2)
data3 = np.zeros(data2.shape)

mediane = np.median(data2)

for i in range(data2.shape[0]):

    for j in range(data2.shape[1]):

        if data2[i][j] > np.median(data2):

            data3[i][j] = 255

        else:

            data3[i][j] = 0



                               
import numpy as np

import matplotlib.pyplot as plt



from skimage import measure



# Find contours at a constant value of 0.8

contours = measure.find_contours(data, data.min()+30)



# Display the image and plot all contours found

fig, ax = plt.subplots()

ax.imshow(data)

for n, contour in enumerate(contours):

    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)



#ax.plot([0.,200.41],[100.,200.], linewidth = 2)

#ax.axis('image')

#ax.set_xticks([])

#ax.set_yticks([])

print(valeur)

plt.show()
"""Importation de l'image et conversion en array"""

from PIL import Image

img = Image.open( "../input/imagefissures/section7-image.png" )

img.load()

data = np.asarray(img, dtype="int32" )
a = []

hq.heappush(a, (10,2,2))

print(a)

hq.heappush(a, (3,0,0))

print(a)

hq.heappush(a, (2,100,10))

print(a)

hq.heappop(a)

print(a)
def f(x,y):

    print(y)

    return 10*x+y

b = np.fromfunction(f,(5,4),dtype=int)

print(b)
array = np.array([[float(10*j+i) for i in range(100)] for j in range(100)])

print(type(array))

im = Image.fromarray(array)

plt.gray()

plt.imshow(im)

print(array)



test = np.array([[[1,20,30],[1000,10001,100000],[3,18,24]], 

                 [[100,800,900],[110,300,564],[120,126,127]], 

                 [[250,289,279],[260,512,514],[270,362,367]]])



print(test.shape)

print(test)

test = np.sum(test, axis = 2)

print(test.shape)

print(test)
test = np.array([[a+i+0.1 for a in range(100)] for i in range(100)])

test[50][99] = 0

#im = Image.fromarray(test)

#plt.gray()

#plt.imshow(im)

fig, ax = plt.subplots()

ax.imshow(test)

tailleCase = 100



for lig in range(test.shape[0] // tailleCase):

    for col in range(test.shape[1] // tailleCase):

        posLig, posCol = lig*tailleCase, col*tailleCase

        mini = test[posLig:posLig+tailleCase,posCol:posCol+tailleCase].min()

        reponse = np.where(test[posLig:posLig+tailleCase,posCol:posCol+tailleCase] == mini)

        print(posLig, posCol)

        print(mini, reponse)

        rep = np.array(list(zip(reponse[0], reponse[1])))

        for x,y in rep:

            ax.scatter(posCol + y, posLig + x+10, s = 100, color = 'r')

plt.show()
fig, ax = plt.subplots()

ax.imshow(test)

ax.scatter([0,10,20,30,40], [0,10,20,30,40], s = 0.1, color = 'r')

plt.show()
test = [1,2,3,4]

print(np.mean(test))
fig, ax = plt.subplots()

ax.imshow(data)

ax.plot([0,100, 50], [200,300, 60], linewidth = 5, color = 'r')



plt.show()