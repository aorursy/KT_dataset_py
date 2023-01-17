"""Class for a own implementation of a self-organizing map (SOM).

Description
-----------
...

Names of the students
---------------------
1. Cem Philipp Freimoser
2. Dominik Jülg

"""
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from veryprettytable import VeryPrettyTable

class Node:
    def __init__(self, amountOfWeights):
        self.weights = np.random.rand(1, amountOfWeights)
        self.associatedLables = {}
        pass

    def adaptWeights(self, learnRate, neighbourhoodFunction, currentDatapoint):
        self.weights = self.weights + learnRate * neighbourhoodFunction * (currentDatapoint - self.weights)

    def netActivity(self, currentDatapoint):
        activity = 0
        for everyValue in currentDatapoint:
            for everyWeight in self.weights[0]:
                activity += everyValue * everyWeight
        return activity

    def classAssoation(self):
        return self.election()

    def associateTo(self, label):
        if not label in self.associatedLables:
            self.associatedLables[label] = 1
        else:
            self.associatedLables[label] += 1

    def election(self):
        if bool(self.associatedLables):
            return max(self.associatedLables, key=lambda k: self.associatedLables[k])
        else:
            return float('NaN')


# -------------------------------------------------------------------------------------------------------------------------------
class SomReport:

    def __init__(self, som, x, y):
        self.som = som
        self.x = x
        self.y = y
        self.confusionMatixes = {}
        pass

    def printReport(self, constituency=1):
        self.calculateMatixes(constituency)
        x = VeryPrettyTable()
        x.field_names = ["Class", "TP", "FP", "FN", "TN", "ACCURACY", "PRECISION", "ERRORRATE"]
        meanTP = 0
        meanFP = 0
        meanFN = 0
        meanTN = 0
        meanAccuracy = 0
        meanPrecision = 0
        meanErrorRate = 0
        for uniqClass in np.sort(np.unique(self.y)):
            row1, row2 = self.confusionMatixes[uniqClass]
            tp, fp = row1
            fn, tn = row2
            accuracy = self.accuracy(tp, fp, fn, tn)
            precision = self.precision(tp, fp)
            errorRate = self.errorRate(tp, fp, fn, tn)
            meanTP += tp
            meanFP += fp
            meanFN += fn
            meanTN += tn
            meanAccuracy += accuracy
            meanPrecision += precision
            meanErrorRate += errorRate
            x.add_row([uniqClass, tp, fp, fn, tn, accuracy, precision, errorRate])
        div = len(np.unique(self.y))
        meanTP = round(meanTP / div, 2)
        meanFP = round(meanFP / div, 2)
        meanFN = round(meanFN / div, 2)
        meanTN = round(meanTN / div, 2)
        meanAccuracy = round(meanAccuracy / div, 2)
        meanPrecision = round(meanPrecision / div, 2)
        meanErrorRate = round(meanErrorRate / div, 2)
        x.add_row(["mean ", meanTP, meanFP, meanFN, meanTN, meanAccuracy, meanPrecision, meanErrorRate])
        print(x)

    def calculateMatixes(self, constituency=1):
        for uniqClass in np.unique(self.y):
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for index in range(len(self.x)):
                xVector = self.x[index]
                yVector = self.y.iloc[index]
                prediction = self.som.predictClass(xVector, constituency)
                # true positive
                if prediction == yVector and yVector == uniqClass:
                    tp += 1
                # true negative
                elif prediction != yVector and yVector != uniqClass:
                    tn += 1
                # false positive
                elif prediction == uniqClass and yVector != uniqClass:
                    fp += 1
                    # fale negative
                elif prediction != uniqClass and yVector == uniqClass:
                    fn += 1
            row1 = [tp, fp]
            row2 = [fn, tn]
            self.confusionMatixes[uniqClass] = [row1, row2]
        return self.confusionMatixes

    def accuracy(self, tp, fp, fn, tn):
        return round((tp + tn) / (tp + fp + fn + tn), 2)

    def precision(self, tp, fp):
        if tp == 0:
            return 0
        return round((tp) / (tp + fp), 2)

    def errorRate(self, tp, fp, fn, tn):
        return round((fp + fn) / (tp + fp + fn + tn), 2)


# -------------------------------------------------------------------------------------------------------------------------------
class SOM:
    """Self-organizing map implementation"""

    def __init__(self, nRows, nColumns):

        self.maxX = nRows
        self.maxY = nColumns
        pass

    def findBMU(self, datapoint):
        xCord = 0
        yCord = 0
        currentMin = float('inf')
        for x in range(self.maxX):
            for y in range(self.maxY):
                vec1 = datapoint
                vec2 = self.output_layer[x][y].weights
                distance = np.linalg.norm(vec1 - vec2)
                if currentMin > distance:
                    xCord = x
                    yCord = y
                    currentMin = distance
        return xCord, yCord, self.output_layer[xCord][yCord]

    def decayRadius(self, currentIteration):
        return self.initRadius * np.exp(-currentIteration / self.timeConstant)

    def decayLearningRate(self, currentIteration):
        return self.initLearingRate * (1 - (currentIteration / self.maxIterations))

    def selectNeuronsInRadius(self, radius, bmuX, bmuY):
        neighbourNeurons = []
        for x in range(self.maxX):
            for y in range(self.maxY):
                neuron = self.output_layer[x][y]
                distance = np.linalg.norm(np.array([bmuX - x, bmuY - y]))
                if distance <= radius:
                    neighbourNeurons.append((neuron, x, y))
        return np.array(neighbourNeurons)

    def neighbourhoodFunction(self, node, bmu, iteration):
        return np.exp(- np.linalg.norm(node - bmu) ** 2 / (2 * self.sigma(iteration) ** 2))

    def sigma(self, iteration):
        return self.initSigma * (1 - (iteration / self.maxIterations))

    def print_progress(self, currentTimeStep):
        if currentTimeStep / (self.maxIterations - 1) >= self.progressBar[0] / 10:
            print(self.progressBar[0] * 10, "%", sep='', end=' ', flush=True)
            self.progressBar.pop(0)
            if currentTimeStep == self.maxIterations - 1:
                print()

    def analyseFeatureSize(self):
        # Herausfinden der Größe des Gewichtsvektors für Neuronen
        a, b = self.trainingData.shape
        indexOfSelectedDataPoint = random.randint(-1, b)
        return self.trainingData[indexOfSelectedDataPoint].shape[0]

    def prepareTraining(self, x, y=None, iterations=10, learningRate=0.1, radius=None, sigma=1):
        if radius == None:
            radius = self.maxX if self.maxX > self.maxY else self.maxY

        self.timeConstant = iterations / np.log(radius)
        self.maxIterations = iterations
        self.initRadius = radius
        self.initLearingRate = learningRate
        self.initSigma = sigma
        self.progressBar = list(range(11))
        self.trainingData = x
        self.labels = y

        temp = [Node(self.analyseFeatureSize()) for x in range(self.maxX * self.maxY)]
        self.output_layer = np.asarray(temp).reshape(self.maxX, self.maxY)

    def selectDataPoint(self):
        # Ziehen mit zurücklegen
        indexOfSelectedDataPoint = random.randint(-1, self.trainingData.shape[1])
        datapoint = self.trainingData[indexOfSelectedDataPoint]
        label = self.labels.iloc[indexOfSelectedDataPoint] if self.labels is not None else "no y provided"
        return datapoint, label

    def fit(self, x, y=None, iterations=10, learningRate=0.1, radius=None, sigma=1):
        self.prepareTraining(x, y, iterations, learningRate, radius, sigma)

        for currentIteration in range(iterations):
            # Competition
            datapoint, label = self.selectDataPoint()
            bmuX, bmuY, bmu = self.findBMU(datapoint)
            # Used for classification
            bmu.associateTo(label)
            # Cooperation
            r = self.decayRadius(currentIteration)
            l = self.decayLearningRate(currentIteration)
            for selectedNeuron in self.selectNeuronsInRadius(r, bmuX, bmuY):
                neuron, x, y = selectedNeuron
                h = self.neighbourhoodFunction(np.array([x, y]), np.array([bmuX, bmuY]), currentIteration)
                # Adaptation
                neuron.adaptWeights(l, h, datapoint)
            self.print_progress(currentIteration)

    def scaleRange(self, array):
        # Hilfsfunktion um Grayscale Bild zu erzeugen
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
        return scaler.fit_transform(array)

    def getMapOfActivity(self, datapoint):
        # Wir erstellen ein Array, dass genau so groß ist,
        # wie die SOM. In den Feldern tragen wir die
        # Netzaktivität ein
        activityMap = np.empty([self.maxX, self.maxY])
        for x in range(self.maxX):
            for y in range(self.maxY):
                activityMap[x][y] = self.output_layer[x][y].netActivity(datapoint)
        return self.scaleRange(activityMap).astype(np.uint8)

    def cluster(self, datapoint):
        # Simple Klassifizierung
        # Liefert als Ergebnis die Koordinaten der BMU
        # als Klasse
        x, y, bmu = self.findBMU(datapoint)
        return x, y

    def getMapOfLabels(self):
        # Wir erstellen ein Array, dass genau so groß ist,
        # wie die SOM. In den Feldern tragen wir die
        # assoziierte Klasse ein
        labelMap = np.empty([self.maxX, self.maxY])
        for x in range(self.maxX):
            for y in range(self.maxY):
                assoicatedClass = self.output_layer[x][y].classAssoation()
                # Da nicht jeder Node eine BMU war, kann nicht jede Node
                # mit einer Klasse assoziiert werden. Daher verschieben wir
                # die Klassen (da nummerisch) um 1 nach rechts. Damit 0 eine
                # legitime default Klasse ist
                if np.isnan(assoicatedClass):
                    assoicatedClass = 0
                else:
                    assoicatedClass += 1

                labelMap[x][y] = assoicatedClass
        return labelMap.astype(np.uint8)


    def askNeibourhood(self, constituency, x, y):
        # Ziel ist es den ersten Nachbarn zu finden,
        # welcher eine Meinung hat
        voters = self.selectNeuronsInRadius(constituency, x, y)
       
        electionResults = {}
        for voter, x, y in voters:
            vote = voter.election()
            if vote not in electionResults and not np.isnan(vote):
                electionResults[vote] = 1
            elif not np.isnan(vote):
                electionResults[vote] += 1
        #key = max(electionResults, key=lambda k: electionResults[k])
        if not bool(electionResults):
            
            # Erhöhe den Umkreis
            return self.askNeibourhood(constituency + 1, x, y)
        return max(electionResults, key=lambda k: electionResults[k])

    def voteOrAskNeigborToVote(self, node, constituency, x, y):
        vote = node.election()
        # Es könnte sein, das diese Node nie eine BMU war
        # Dann fragen wir einfach die Nachbaren was die so
        # meinen.
        if np.isnan(vote):
            return self.askNeibourhood(constituency, x, y)
        return vote

    def voteCount(self, datapoint, constituency):
        # Wir suchen zu einen Datenpunkt die BMU.
        # Anschließend schauen wir mit welchen Klassen,
        # diese BMU assoziiert wurde. Weiterhin kann ein Radius
        # definiert werden, welcher auch befragt wird
        # ähnlich wie bei KNN
        x, y, bmu = self.findBMU(datapoint)
        voters = self.selectNeuronsInRadius(constituency, x, y)
        electionResults = {}
        for (node, x, y) in voters:
            vote = self.voteOrAskNeigborToVote(node, constituency, x, y)
            if vote not in electionResults:
                electionResults[vote] = 1
            else:
                electionResults[vote] += 1
        return electionResults

    def predict(self, datapoint, constituency=1):
        electionResults = self.voteCount(datapoint, constituency)
        sumOfVotes = 0

        for (key, value) in electionResults.items():
            sumOfVotes += value

        results = {}
        for key, value in electionResults.items():
            results[key] = value / sumOfVotes

        return results

    def predictClass(self, datapoint, constituency=1):
        electionResults = self.voteCount(datapoint, constituency)
        return max(electionResults, key=lambda k: electionResults[k])
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
# read data
df = pd.read_csv("../input/20nm_lucas_train.csv", index_col=0, low_memory=False)

# drop NaN values
df.dropna(how="any", inplace=True)
# create list including all hyperspectral bands
hypbands = [str(x) for x in np.arange(400., 2500., 20)]
features = df.loc[:, '400.0':'2480.0']
SolutionCol = df['Label']
classes = {0: "Sand", 1: "LoamySand", 2: "SandyLoam", 3: "Loam", 4: "SiltLoam", 5: "Silt", 6: "SandyClayLoam",
           7: "ClayLoam", 8: "SiltyClayLoam", 9: "SandyClay", 10: "SiltyClay", 11: "Clay"}
from sklearn.decomposition import PCA
pca = PCA(n_components=40)
pca.fit(features)
features = pca.transform(features)
from sklearn.model_selection import train_test_split
x, x_test, y, y_test = train_test_split(features, SolutionCol, test_size=0.1, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=1)
numberOfRows = 10
numberOfColumns = 10
som = SOM(numberOfRows, numberOfColumns)
som.fit(x = x_train, y= y_train,iterations = 8000)
## 1. Testen Sie die SOM
from matplotlib.pyplot import imshow
from PIL import Image 
def printImagesFrom(datapoint):
    greyScaleMatrix = som.getMapOfActivity(datapoint)
    #print(greyScaleMatrix)
    %matplotlib inline
    imshow(greyScaleMatrix, cmap='gray')
def selectRandomDatapoint():
    a,b = x_val.shape
    indexOfSelectedDataPoint = random.randint(-1, b)
    x = x_val[indexOfSelectedDataPoint]
    return x

def selectRandomDatapointWithLabel():
    a,b = x_val.shape
    indexOfSelectedDataPoint = random.randint(-1, b)
    x = x_val[indexOfSelectedDataPoint]
    y = y_val.iloc[indexOfSelectedDataPoint]
    return x, y
printImagesFrom(selectRandomDatapoint())
for i in range(10):
    datapoint, label = selectRandomDatapointWithLabel()
    cluster = som.cluster(datapoint)
    print("SOM cluster:", cluster, "Real Label", label)
for i in range(10):
    datapoint, label = selectRandomDatapointWithLabel()
    l = som.predictClass(datapoint)
    print("SOM predicted:", l, "Real Label", label)
datapoint, label = selectRandomDatapointWithLabel()
som.predict(datapoint)
report = SomReport(som, x_val, y_val)
report.printReport()
image = som.getMapOfLabels()
%matplotlib inline
t = 1
cmap = {0:[0,0,0,t], 1:[0,0,0.4,t], 2:[0,0.6,0,t],3:[0.3,0,0,t],4:[0.3,0,0.6,t],5:[0.8,0,0,t] ,6:[0.5,0.2,0,t],7:[0.7,0.7,0.6,t], 8:[0.8,0.2,0.5,t], 9:[0.8,1.0,0,t],10:[0.9,0.6,0,t], 11:[1,1,0.5,t] , 12:[1,1,0.5,t]}
arrayShow = np.array([[cmap[i] for i in j] for j in image])    
plt.imshow(arrayShow)