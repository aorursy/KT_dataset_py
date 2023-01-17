import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from PIL import Image

import os

import time

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans
PCA_n_components = 25

RF_n_estimators = 25
path = "../input"

shapeInfo = []

shapeAdditionalInfo = []

labels = []

maxX = 0

maxY = 0

maxId = 0

HEIGHT=20

WIDTH=50

for d in os.listdir(path):

    personPath = "%s/%s" % (path,d)

    phrasesPath = "%s/phrases" % personPath

    if os.path.isdir(personPath):

        if(len(d) == 3):

            person = d

            for d in os.listdir(phrasesPath):

                sentenceId = d

                shapesPath = "%s/%s/shapes.csv" % (phrasesPath,sentenceId)

                if(os.path.isfile(shapesPath)):

                    #print(shapesPath)

                    shapes = pd.read_csv(shapesPath)

                    shapes = shapes.sort_values(["sentenceId2","id","y","x"])

                    currentNumpyShape = None

                    n = 0

                    lastSentenceId2 = None

                    lastId = None

                    maxIdX = 0

                    maxIdY = 0

                    currentShapeList = []

                    for r in shapes.itertuples():

                        sentenceId2 = r.sentenceId2

                        x = r.x

                        y = r.y

                        id = r.id

                        maxIdX = max(maxIdX,x)

                        maxIdY = max(maxIdY,y)

                        maxX = max(maxX,x)

                        maxY = max(maxY,y)

                        maxId = max(maxId,id)

                        if sentenceId2 != lastSentenceId2:

                            n = 0

                            if currentNumpyShape is not None:

                                for cnS in currentNumpyShape[id]:

                                    cnS[0] = cnS[0]/maxIdX

                                    cnS[1] = cnS[1]/maxIdY

                            for csL in currentShapeList:

                                ox = csL["x"]

                                oy = csL["y"]

                                csL["x"] = int((ox/maxIdX)*(WIDTH-1))

                                csL["y"] = int((oy/maxIdY)*(HEIGHT-1))

                                if csL["y"] >= HEIGHT:

                                    raise Exception("#1 WTF %d %d %d" % (oy,maxIdY,csL["y"]))

                                if csL["x"] >= WIDTH:

                                    raise Exception("#2 WTF %d %d" % (maxIdX,csL["x"]))

                                shapeLists.append(csL)

                            currentShapeList = []

                            maxIdX = 0

                            maxIdY = 0

                            currentNumpyShape = np.zeros((40,68-48,2),dtype=np.dtype('float'))

                            shapeLists = []

                            shapeAdditionalInfo.append(shapeLists)

                            shapeInfo.append(currentNumpyShape)

                            labels.append(sentenceId)

                        if lastId != id:

                            n = 0

                        X = {"id":r.id,"x":r.x,"y":r.y}

                        currentShapeList.append(X)

                        currentNumpyShape[id][n] = [x,y]

                        lastSentenceId2 = sentenceId2

                        lastId = id

                        n += 1

                    shapeAdditionalInfo.append(shapeLists)

                    for cnS in currentNumpyShape[id]:

                        cnS[0] = cnS[0]/maxIdX

                        cnS[1] = cnS[1]/maxIdY

                    for csL in currentShapeList:

                        ox = csL["x"]

                        oy = csL["y"]

                        csL["x"] = int((ox/maxIdX)*(WIDTH-1))

                        csL["y"] = int((oy/maxIdY)*(HEIGHT-1))

                    shapeInfo.append(currentNumpyShape)

                    labels.append(sentenceId)



labels = np.array(labels)



print("maxX: %d maxY: %d maxId: %d" % (maxX,maxY,maxId))
lst = []

lstIdx = []

lstId = []

n = 0

scaler = StandardScaler()

for sai in shapeAdditionalInfo:

    for i in range(1,30):

        sI = np.zeros((HEIGHT,WIDTH),dtype=np.dtype('uint8'))

        found = False

        for s in sai:

            if s["id"] == i:

                sI[s["y"]][s["x"]] = 1

                found = True

        if found:

            lst.append(sI.flatten().tolist())

            lstIdx.append(n)

            lstId.append(i)

    n += 1

score = -20000000

cols = 4

bestK = 0

bestKmeans = None

data = np.array(lst)

scaler.fit(data)

data = scaler.transform(data)

for K in range(3,20):

    kmeans = KMeans(n_clusters=K, random_state=0)

    kmeans.fit(data)

    tmpscore = kmeans.score(data)

    print("tmpscore for K [%d]: %f" % (K,tmpscore))

    if(tmpscore > score):

        score = tmpscore

        bestK = K

        bestKmeans = kmeans

classes = bestK

kmeans = bestKmeans

#classes = 2

#kmeans = KMeans(n_clusters=2, random_state=0)

#kmeans.fit(data)

print("Best KMeans score: %f with %d classes" % (kmeans.score(data),bestK))

klabels = kmeans.labels_.tolist()

print("first 40 entries:")

print(klabels[0:40])

plt.figure(figsize=(10,2*classes))

plt.subplots_adjust(wspace=0.2,hspace=2)

for l in range(0,classes):

    idx = -1

    for i in range(0,cols):

        idx = klabels[idx+1:].index(l)+idx+1

        sI = np.zeros((HEIGHT,WIDTH),dtype=np.dtype('uint8'))

        for s in shapeAdditionalInfo[lstIdx[idx]]:

            if s["id"] == lstId[idx]:

                sI[s["y"]][s["x"]] = 1

        img = Image.fromarray(sI*255, 'L')

        plt.subplot(classes,cols,1+l*cols+i)

        plt.imshow(img)

        plt.title("class: %d offset: %d" % (l,idx))
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}

id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

label_ids = np.array([label_to_id_dict[x] for x in labels])

id_to_label_dict
sI = np.zeros((HEIGHT,WIDTH),dtype=np.dtype('uint8'))

for s in shapeAdditionalInfo[3]:

    if s["id"]==1:

        sI[s["y"]][s["x"]] = 1

img = Image.fromarray(sI*255, 'L')

img
start = time.time()

pca = PCA(n_components=PCA_n_components)

shapeInfoFlattten = [ sI.flatten() for sI in shapeInfo ]

print("len(label_ids): %d, len(labels): %d, len(shapeInfoFlattten): %d" % (len(label_ids),len(labels),len(shapeInfoFlattten)))

pca_result = pca.fit_transform(shapeInfoFlattten)

end = time.time()

print("PCA took %d s" % (end-start))
start = time.time()



X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids,stratify = label_ids, test_size=0.25, random_state=42)



end = time.time()

print("train_test_split took %d s" % (end - start))
start = time.time()



forest = RandomForestClassifier(n_estimators=RF_n_estimators,n_jobs=4,random_state=42)

forest = forest.fit(X_train, y_train)



end = time.time()

print("RandomForestClassifier took %d s" % (end - start))
start = time.time()



test_predictions = forest.predict(X_test)



end = time.time()

print("test_predictions took %d s" % (end - start))
start = time.time()



precision = accuracy_score(test_predictions, y_test) * 100



end = time.time()

print("RandomForest accuracy_score took %d s" % (end - start))

print("Accuracy with RandomForest: {0:.6f}".format(precision))