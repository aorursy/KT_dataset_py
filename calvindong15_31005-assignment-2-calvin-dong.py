# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # split the dataframe into training and test datasets

from sklearn.metrics import accuracy_score # Determine how many instances the model has guessed correctly

from sklearn.metrics import classification_report # Get performance measures

from timeit import default_timer as timer # Time how long it takes to perform python operations





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv', encoding="latin1")

# Picking best columns to be used in training

bostonCrimes = df[["DISTRICT","MONTH", "DAY_OF_WEEK", "HOUR", "UCR_PART"]]

bostonCrimes = bostonCrimes.dropna(subset = ["DISTRICT","MONTH", "DAY_OF_WEEK", "HOUR", "UCR_PART"])



# Binning values

monthBins = [0,4,8,12]

hourBins = [0,6,12,18,24]

bostonCrimes["MONTH"] = pd.cut(bostonCrimes["MONTH"], monthBins)

bostonCrimes["HOUR"] = pd.cut(bostonCrimes["HOUR"], hourBins, include_lowest = True)

print(bostonCrimes)

dfTest = pd.read_csv('../input/gini-test-dataset/test dataset.csv')

weather = dfTest.drop(columns = ["Day"])

print(weather)
class Tree:

    def __init__(self, data, target):

        self.rootNode = None

        self.data = data

        self.target = target

        self.maxTreeDepth = self.data.shape[1] - 1 # Max tree depth will be the amount of attributes that are present in the dataset

        self.predictedData = None

    

    def Fit(self, data, target):

        self.rootNode = TreeNode(data, target, self.maxTreeDepth)

        self.rootNode.Make()

        

    def Predict(self, data, target):

        dataOrig = data

        data = data.drop(columns = target)

        predictedList = pd.DataFrame()

        decisionMade = False

        nextNode = False

        for i in range(len(data)):

            decisionMade = False

            nextNode = False

            instance = data.iloc[[i]] # Get instance to predict

            currentNode = self.rootNode

            while not decisionMade:

                if currentNode.leafNode:

                    instance.insert(0, f"Predicted_{target}",currentNode.prediction , True) # Add prediction decision to instance

                    frames = [predictedList, instance] 

                    predictedList = pd.concat(frames) # Add instance to dataframe to be returned to user

                    decisionMade = True

                    nextNode = True

                else:

                    for children in currentNode.children: 

                        if (instance[currentNode.splitAttribute].unique() == children.decision): # If no decision made, find next node in tree to go to 

                            currentNode = children

                            nextNode = True

                            break

                                 

                if not nextNode: # Exception for when the instance can not find a path through the decision tree

                    instance.insert(0, f"Predicted_{target}", "Nan", True) # Add prediction decision to instance

                    frames = [predictedList, instance] 

                    predictedList = pd.concat(frames) # Add instance to dataframe to be returned to user

                    decisionMade = True

                nextNode = False

        

        self.predictedData = predictedList

        print("\naccuracy if the model is: ", end='')

        print(accuracy_score(dataOrig[self.target], predictedList[f"Predicted_{target}"]))

        print(classification_report(dataOrig[self.target], predictedList[f"Predicted_{target}"]))

        return predictedList

    

        

    def PrintTree(self):

        unprocessed = [] # A stack to be used for depth first traversal of the tree

        unprocessed.append(self.rootNode)

        rootExceptions = []

        treePrinted = False

        leafPrinted = False

        processedLeafs = []

        

        print("Printing decision tree")

        print (f"Targets are {self.data[self.target].unique()}")

        while not (len(unprocessed) == 0):

            processedString = ""

            tempString = ""

            leafPrinted = False

            temp = unprocessed[-1]

            unprocessed.pop()

            if not temp.leafNode:

                for children in temp.children:

                    unprocessed.append(children)

                        

            else:

                currentNode = temp

                while not leafPrinted: # Travel back up the tree to root node so that branch can be printed

                    if currentNode.leafNode:

                        tempString = f"{currentNode.decision}: {currentNode.prediction}"

                        currentNode = currentNode.parent

                    elif currentNode.parent == None:

                        tempString = f"{currentNode.splitAttribute} "

                        leafPrinted = True

                    else:

                        tempString = f"{currentNode.decision}" + ", " + currentNode.splitAttribute + " is "

                        currentNode = currentNode.parent

                    processedString = tempString + processedString 

                

                print(processedString)
class TreeNode:     

    def __init__(self, data, target, nodeDepth):

        self.children = []

        self.parent = None

        self.leafNode = False

        self.nodeDepth = nodeDepth # Not a user defined node depth, node depth is used here to determine if all attributes have been split in a branch

        self.splitAttribute = None

        self.decision = None

        self.prediction = None

        self.data = data

        self.target = target

        self.targetValues = self.data[self.target].unique()

        self.size = self.data.shape[0]

        #self.minNodeSize = 2

    

    def Make(self):

        lowestGini = 1

        if (len(self.data[self.target].unique()) == 1 or self.nodeDepth == 0):

            self.CreateNewLeafNode()

            return



        for attribute in self.data:

            if (attribute == self.target):

                continue

            else:

                returnedWeightedGini = self.CalculateGiniIndex(attribute)

                #print(f"Weighted gini {attribute} returned a weighted gini of {returnedWeightedGini}")

                if (returnedWeightedGini < lowestGini):

                    lowestGini = returnedWeightedGini

                    self.splitAttribute = attribute

        # print(f"Split attribute will be {self.splitAttribute}, as it returned a weighted gini of {lowestGini}")

        

        # The new split has now been calculated and new nodes formed

        for classes in self.data[self.splitAttribute].unique():

            self.children.append(TreeNode(self.data.loc[self.data[self.splitAttribute] == classes], self.target, self.nodeDepth - 1))

            self.children[-1].decision = classes # Set class for next node to split on

            self.children[-1].parent = self # make reference to itself as the parent node of the child

            self.children[-1].Make()

                

        # This was code testing performance using minimum node sizes as a stopping criterion        

        """createLeaf = False

        for uniqueAttVar in self.data[self.splitAttribute].unique():

            newData = self.data.loc[self.data[self.splitAttribute] == uniqueAttVar]

            if (newData.shape[0] < self.minNodeSize):

                createLeaf = True

             

            

        if not createLeaf:

            for uniqueAttVar in self.data[self.splitAttribute].unique():

                self.children.append(TreeNode(self.data.loc[self.data[self.splitAttribute] == uniqueAttVar], self.target, self.nodeDepth - 1))

                self.children[-1].decision = uniqueAttVar # Set decision for next node to split on

                self.children[-1].Make()

        else:

            self.CreateNewLeafNode()

            return"""

    

    def CalculateGiniIndex(self, attribute):

        weightedGini = 0

        for classes in self.data[attribute].unique():

            giniIndex = 1 # gini index forula is 1 - sum(squared probabilties for each class)

            sortedCla = self.data.loc[self.data[attribute] == classes] # Get all instances for a unique value in every attribute

            numOfInst = sortedCla.shape[0]

            if (numOfInst == 0): # Skip to avoid division by 0

                continue 

            for targets in self.targetValues:

                numOfTarInst = sortedCla.loc[sortedCla[self.target] == targets].shape[0] # Find how many instances of each class occurs when looking at a certain target

                giniIndex = giniIndex - pow(numOfTarInst/numOfInst, 2)

            

            #print(f'Gini index of {uniqueAttVar} is {giniIndex}')

            weightedGini = weightedGini + ((numOfInst/self.size)*giniIndex) # Keep adding calculations for weighted gini for every unique attribute value found

        #print(f'Weighted gini for {attribute} is {weightedGini}')

        return weightedGini  

    

    def CreateNewLeafNode(self):

        self.leafNode = True

        highestCount = 0

        for uniqueClass in self.data[self.target].unique():

            numOf = self.data.loc[self.data[self.target] == uniqueClass].shape[0]

            if (numOf > highestCount):

                self.prediction = uniqueClass

        #print(f"Lead node here for {self.splitAttribute} and prediction is {self.prediction} and decision is {self.decision}\n")

        return
if __name__ == "__main__":

    trainingData, testingData = train_test_split(bostonCrimes, train_size = 20000, test_size = 2000) # Boston Crimes dataset contains 317218 instances after processing, 

                                                                                           # making a prediction on this can takes upwards of thirty minutes if all instances are used

    start = timer() 

    tree = Tree(weather, "Decision")# Making tree for weather dataset

    tree.Fit(weather, "Decision")

    tree.Predict(weather, "Decision")

    tree.PrintTree()

    end = timer()

    print(f"Time to complete: {end - start}")

    

    start = timer()

    treeBost = Tree(trainingData, "UCR_PART") # Making tree for Boston crimes dataset

    treeBost.Fit(trainingData, "UCR_PART")

    predictedData = treeBost.Predict(testingData, "UCR_PART")

    treeBost.PrintTree()

    end = timer()

    print(f"Time to complete: {end - start}")

    