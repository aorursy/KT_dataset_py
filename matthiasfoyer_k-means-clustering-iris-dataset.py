from math import pow, sqrt



class Point:

    def __init__(self,SL,SW,PL,PW):

        self.PL = PL

        self.PW = PW

        self.SL = SL

        self.SW = SW

        

    def dist(self,p):

        return(sqrt(pow(self.PL-p.PL,2)+pow(self.PW-p.PW,2)+pow(self.SL-p.SL,2)+pow(self.SW-p.SW,2)))
import statistics



class Cluster:

    def __init__(self):

        self.collection = []

        

    def append(self,point):

        self.collection.append(point)

        if len(self.collection) is 1:

            self.center = point

        

    def updateCenter(self):

        tmp_PL = []

        tmp_PW = []

        tmp_SL = []

        tmp_SW = []

        for i in self.collection:

            tmp_PL.append(i.PL)

            tmp_PW.append(i.PW)

            tmp_SL.append(i.SL)

            tmp_SW.append(i.SW)

        self.center = Point(statistics.mean(tmp_PL),statistics.mean(tmp_PW),statistics.mean(tmp_SL),statistics.mean(tmp_SW))
import pandas as pd

import seaborn as sns

import random



def clusterize(df,k):



    # Initialisation des clusters

    

    # On peut aussi utiliser des clusters aléatoires

    # randomRows = random.sample(range(0, len(df.index)), k)

    

    rows = [5,55,132]

    clusterList = []

    for i in rows:

        c = Cluster()

        c.append(Point(df.iloc[i,0],df.iloc[i,1],df.iloc[i,2],df.iloc[i,3]))

        clusterList.append(c)



    while True:

        

        # Enregistrement des valeurs des centres en entrée

        c1 = []

        for i in range(k):

            c1.append(clusterList[i].center)



        # Assignation des instances

        for i in range(len(df.index)):

            d = []

            iPoint = Point(df.iloc[i,0],df.iloc[i,1],df.iloc[i,2],df.iloc[i,3])

            for j in range(k):

                d.append(clusterList[j].center.dist(iPoint))

            minCluster = d.index(min(d))

            clusterList[minCluster].append(iPoint)



        # Mise à jour des centres

        for i in range(k):

            clusterList[i].updateCenter



        # Enregistrement des valeurs des centres en sortie

        c2 = []

        for i in range(k):

            c2.append(clusterList[i].center)

            

        # Vérification de la condition d'arrêt

        if (c1 == c2):

            break

        

        # Suppression des assignations

        for i in range(k):

            clusterList[i].collection = []

            

    petalWidth, petalLength, sepalWidth, sepalLength, cluster = [], [], [], [], []

        

    for i in range(k):

        for j in range(len(clusterList[i].collection)):

            petalWidth.append(clusterList[i].collection[j].PW)

            petalLength.append(clusterList[i].collection[j].PL)

            sepalWidth.append(clusterList[i].collection[j].SW)

            sepalLength.append(clusterList[i].collection[j].SL)

            cluster.append(i+1)



    results = pd.DataFrame(list(zip(petalWidth, petalLength, sepalWidth, sepalLength, cluster)),

                      columns =['petalWidth', 'petalLength', 'sepalWidth', 'sepalLength', 'cluster'])



    sns.pairplot(results, hue='cluster')

    

    return clusterList
IRIS = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
clusterize(IRIS,3)
sns.pairplot(IRIS, hue='species')