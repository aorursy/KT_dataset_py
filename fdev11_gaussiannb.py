# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy

import scipy

import scipy.special

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import distance

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#from scipy.special import softmax

from sklearn.naive_bayes import GaussianNB

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def read_arff_data(filename):

    f = open(filename)

    data_line = False

    data = []

    for l in f:

        l = l.strip() # get rid of newline at the end of each input line

        if data_line:

            content = [float(x) for x in l.split(',')]

            if len(content) == 3:

                data.append(content)

        else:

            if l.startswith('@DATA'):

                data_line = True

    return data



arff_train = read_arff_data("../input/trainarfftxt/train.arff.txt")

arff_test = read_arff_data("../input/trainarfftxt/eval.arff.txt")



train_data = pd.read_csv('../input/circle2/train(1).csv')

np_train_X = train_data[['X','Y']].values

np_train_Y = train_data[['class']].values

np_train_Y = [v[0] for v in np_train_Y]



test_data = pd.read_csv('../input/circle2/test(1).csv')

np_test_X = test_data[['X','Y']].values

np_test_Y = test_data[['class']].values

np_test_Y = [v[0] for v in np_test_Y]



#print(np_train_X)

#print(np_train_Y)

#print(np_test_X)

#print(np_test_Y)



circle_train = 1

circle_test = 2
# Sources:

# - General: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# - GaussianNB: https://scikit-learn.org/stable/modules/naive_bayes.html



def possible_class_values(expected):

    want = expected

    classes = {}

    for i in range(0,len(want)):

        c = want[i]

        classes[c] = 1



    return classes.keys()



class Data:

    # Expected data format: dataTable,  

    def __init__(self, data, expected):

        self.data = data

        self.expected = expected

        pass

    

    def possible_class_values(self):

        return possible_class_values(self.expected)



def split_arff(data):

    AData = []

    AOutput = []

    # Split data and expected output into their own arrays.                            

    for row in data:

        # Split into data and expected value.

        X, x = [row[0], row[1]], row[2]

        AData.append(X)

        AOutput.append(x)

    

    return Data(AData, AOutput)



class GaussianClassifier:

    def __init__(self):

        self.clf = GaussianNB()

        pass

    

    def fit(self, d: Data):

        self.clf.fit(d.data, d.expected)

    

    def predict(self, data):

        return self.clf.predict(data)

    

    def plot_internal(self, color_map):

        print("Plotting gaussian internal.")



# Geometrischer Schwerpunkt-basiert.

class LinearClassifier:

    def __init__(self):

        self.distance = distance.euclidean

        self.distance = lambda a, b : math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        pass

    

    def fit(self, d: Data):

        data = d.data

        want = d.expected

        counts = {}

        sums = {}

        classes = d.possible_class_values()

        for c in classes:

            sums[c] = [0.0, 0.0]

            counts[c] = 0

        

        for i in range(0, len(data)):

            sums[want[i]] =  numpy.add(sums[want[i]], data[i])

            counts[want[i]] += 1 # Count how many for Schwerpunkt.

        

        # Berechne geometrischen SP.

        for c in classes:

            sums[c] = [coord / counts[c] for coord in sums[c]]

            

        self.spunkte = sums

        

    def predict(self, data):

        out = []

        

        for row in data:

            c = self.predict_single(row)

            out.append(c)

        

        return out

    

    def predict_single(self, point):

        distances = {}

        minDistanceClass = list(self.spunkte.keys())[0]

        minDistanceValue = self.distance(self.spunkte[minDistanceClass], point)

        

        # Calculate sp of nearest class.

        for c, sp in self.spunkte.items():

            dist = self.distance(sp, point)

            if dist < minDistanceValue:

                minDistanceClass = c

                minDistanceValue = dist

                

        return minDistanceClass

    

    def plot_internal(self, color_map):

        print("Plotting internal Linear.", len(self.spunkte.keys()))



        for c, sp in self.spunkte.items():

            print("sp:", sp)

            plt.plot(sp[0],sp[1],'ro', color='red')

            

        self.debug_point()

            

            

    def debug_point(self):

        p = [-0.820002, 1.905271]

        print("p-debug:", p)

        for c, sp in self.spunkte.items():

            print("is:", self.predict_single(p), "c:", c, "c-sp:", sp, "c-distance:", self.distance(sp, p))

        

        p = -0.820002, 1.905271

        #plt.plot(p[0],p[1],'ro', color='black')

        

            



class Classifier:

    def __init__(self, clf, train: Data, test: Data):

        self.train = train

        self.test = test

        

        self.clf = clf

        self.clf.fit(self.train)

        pass

    

    def predict(self, mydata):

        return self.clf.predict(mydata)

    

    def random_color():

        rgbl=[255,0,0]

        random.shuffle(rgbl)

        return tuple(rgbl)

    

    def get_dynamic_class_color_map(self, classes):

        class_color_map = {}

        for i in range(0,len(classes)):

            class_color_map[classes[i]] = 33

            

    def get_static_class_color_map(self, classes):

        return {-1: 'blue', 0: 'red', 1: 'orange'}

    

    def get_static_class_color_map_special(self):

        return {-1: 'maroon', 0: 'peru', 1: 'cyan'}

    

    def get_class_color_map(self, classes):

        return self.get_static_class_color_map(classes)

    

    def get_class_color_map_special(self):

        return self.get_static_class_color_map_special()

        

    def plot_train(self):

        self.plot(self.train)

        

    def plot_test(self):

        self.plot(self.test)

    

    def plot(self, d):

        data = d.data

        want = d.expected

        X11 = []

        X12 = []

        X21 = []

        X22 = []

        

        color_map = self.get_class_color_map(d.possible_class_values())

        

        X = {}

        Y = {}

        

        for c in d.possible_class_values():

            X[c] = []

            Y[c] = []

        

        for i in range(0,len(data)):

            X[want[i]].append(data[i][0])

            Y[want[i]].append(data[i][1])

            #if want[i] == 1:

            #    X11.append(data[i][0])

            #    X12.append(data[i][1])

            #if want[i] == -1:

            #    X21.append(data[i][0])

            #    X22.append(data[i][1])

            

        for k in X.keys():

            x = X[k]

            y = Y[k]

            plt.scatter(x, y, color=color_map.get(k, 'black'))

        

        #plt.scatter(X11, X12,color='red')

        #plt.scatter(X21, X22,color='blue')

    

    def plot_special(self, data, color_map):

        X = {}

        Y = {}

        

        classes = data.keys()



        for c in classes:

            X[c] = []

            Y[c] = []

        

        for c, lst in data.items():

            for i in range(0, len(data[c])):

                X[c].append(data[c][i][0])

                Y[c].append(data[c][i][1])

            #if want[i] == 1:

            #    X11.append(data[i][0])

            #    X12.append(data[i][1])

            #if want[i] == -1:

            #    X21.append(data[i][0])

            #    X22.append(data[i][1])

            

        for k in X.keys():

            print("color:", color_map.get(k, 'green'))

            x = X[k]

            y = Y[k]

            plt.scatter(x, y, color=color_map.get(k, 'green'))

    

        

    def doit(self):

        data = self.test.data

        got = self.predict(data)

        #print("got:", got)

        #print("want:", self.test.expected)

        

        self.clf.plot_internal(self.get_class_color_map(self.train.possible_class_values))

        

        want = self.test.expected



        scoreboard = {"good": 0, "bad": 0}

        good = []

        bad = {}

        for c in self.test.possible_class_values():

            bad[c] = []



        for i in range(0,len(data)):

            if got[i] == want[i]:

                scoreboard["good"] += 1

                good.append(data[i])

            else:

                scoreboard["bad"] += 1

                bad[want[i]].append(data[i])



        print('score - {:f}%'.format(scoreboard["good"] * 1.0 / len(data) * 100.0))



        print("bad:", bad)

        

        self.plot_special(bad, self.get_class_color_map_special())



#class2.doit()
class1 = Classifier(GaussianClassifier(), split_arff(arff_train), split_arff(arff_test))

class1.plot_test()

class1.doit()
class1 = Classifier(LinearClassifier(), split_arff(arff_train), split_arff(arff_test))

class1.plot_test()

class1.doit()
class2 = Classifier(GaussianClassifier(), Data(np_train_X, np_train_Y), Data(np_test_X, np_test_Y))

class2.plot_test()

class2.doit()
class2 = Classifier(LinearClassifier(), Data(np_train_X, np_train_Y), Data(np_test_X, np_test_Y))

class2.plot_test()

class2.doit()
# k-nearest-neighbor. Gridden.

# Linear. Schwerpunkte. Zu welchem SP geringste Distanz.