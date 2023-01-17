# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Imports

import pandas as pd

import numpy as np

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import SpatialDropout1D

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Activation

from tensorflow.keras import optimizers

from keras.utils import to_categorical

import matplotlib.mlab as mlab

import csv

import math

import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.text import Tokenizer

import time

import datetime

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

tf.__version__

print("importing Done ^_^")
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

import numpy as np



#source - https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/

class SOM(object):

    """

    2-D Self-Organizing Map with Gaussian Neighbourhood function

    and linearly decreasing learning rate.

    """



    #To check if the SOM has been trained

    _trained = False



    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):

        """

        Initializes all necessary components of the TensorFlow

        Graph.



        m X n are the dimensions of the SOM. 'n_iterations' should

        should be an integer denoting the number of iterations undergone

        while training.

        'dim' is the dimensionality of the training inputs.

        'alpha' is a number denoting the initial time(iteration no)-based

        learning rate. Default value is 0.3

        'sigma' is the the initial neighbourhood value, denoting

        the radius of influence of the BMU while training. By default, its

        taken to be half of max(m, n).

        """



        #Assign required variables first

        self._m = m

        self._n = n

        if alpha is None:

            alpha = 0.3

        else:

            alpha = float(alpha)

        if sigma is None:

            sigma = max(m, n) / 2.0

        else:

            sigma = float(sigma)

        self._n_iterations = abs(int(n_iterations))



        ##INITIALIZE GRAPH

        self._graph = tf.Graph()



        ##POPULATE GRAPH WITH NECESSARY COMPONENTS

        with self._graph.as_default():



            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE



            #Randomly initialized weightage vectors for all neurons,

            #stored together as a matrix Variable of size [m*n, dim]

            self._weightage_vects = tf.Variable(tf.random.normal(

                [m*n, dim]))



            #Matrix of size [m*n, 2] for SOM grid locations

            #of neurons

            self._location_vects = tf.constant(np.array(

                list(self._neuron_locations(m, n))))



            ##PLACEHOLDERS FOR TRAINING INPUTS

            #We need to assign them as attributes to self, since they

            #will be fed in during training



            #The training vector

            self._vect_input = tf.placeholder("float", [dim])

            #Iteration number

            self._iter_input = tf.placeholder("float")



            ##CONSTRUCT TRAINING OP PIECE BY PIECE

            #Only the final, 'root' training op needs to be assigned as

            #an attribute to self, since all the rest will be executed

            #automatically during training



            #To compute the Best Matching Unit given a vector

            #Basically calculates the Euclidean distance between every

            #neuron's weightage vector and the input, and returns the

            #index of the neuron which gives the least value

            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(

                tf.pow(tf.subtract(self._weightage_vects, tf.stack(

                    [self._vect_input for i in range(m*n)])), 2), 1)),

                                  0)



            #This will extract the location of the BMU based on the BMU's

            #index

            slice_input = tf.pad(tf.reshape(bmu_index, [1]),

                                 np.array([[0, 1]]))

            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,

                                          tf.constant(np.array([1, 2]))),

                                 [2])



            #To compute the alpha and sigma values based on iteration

            #number

            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,

                                                  self._n_iterations))

            _alpha_op = tf.multiply(alpha, learning_rate_op)

            _sigma_op = tf.multiply(sigma, learning_rate_op)



            #Construct the op that will generate a vector with learning

            #rates for all neurons, based on iteration number and location

            #wrt BMU.

            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(

                self._location_vects, tf.stack(

                    [bmu_loc for i in range(m*n)])), 2), 1)

            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(

                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))

            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)



            #Finally, the op that will use learning_rate_op to update

            #the weightage vectors of all neurons based on a particular

            #input

            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(

                learning_rate_op, np.array([i]), np.array([1])), [dim])

                                               for i in range(m*n)])

            weightage_delta = tf.multiply(

                learning_rate_multiplier,

                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),

                       self._weightage_vects))                                         

            new_weightages_op = tf.add(self._weightage_vects,

                                       weightage_delta)

            self._training_op = tf.assign(self._weightage_vects,

                                          new_weightages_op)                                       



            ##INITIALIZE SESSION

            self._sess = tf.Session()



            ##INITIALIZE VARIABLES

            init_op = tf.initialize_all_variables()

            self._sess.run(init_op)



    def _neuron_locations(self, m, n):

        """

        Yields one by one the 2-D locations of the individual neurons

        in the SOM.

        """

        #Nested iterations over both dimensions

        #to generate all 2-D locations in the map

        for i in range(m):

            for j in range(n):

                yield np.array([i, j])



    def train(self, input_vects):

        """

        Trains the SOM.

        'input_vects' should be an iterable of 1-D NumPy arrays with

        dimensionality as provided during initialization of this SOM.

        Current weightage vectors for all neurons(initially random) are

        taken as starting conditions for training.

        """



        #Training iterations

        for iter_no in range(self._n_iterations):

            #Train with each vector one by one

            for input_vect in input_vects:

                self._sess.run(self._training_op,

                               feed_dict={self._vect_input: input_vect,

                                          self._iter_input: iter_no})



        #Store a centroid grid for easy retrieval later on

        centroid_grid = [[] for i in range(self._m)]

        self._weightages = list(self._sess.run(self._weightage_vects))

        self._locations = list(self._sess.run(self._location_vects))

        for i, loc in enumerate(self._locations):

            centroid_grid[loc[0]].append(self._weightages[i])

        self._centroid_grid = centroid_grid



        self._trained = True



    def get_centroids(self):

        """

        Returns a list of 'm' lists, with each inner list containing

        the 'n' corresponding centroid locations as 1-D NumPy arrays.

        """

        if not self._trained:

            raise ValueError("SOM not trained yet")

        return self._centroid_grid



    def map_vects(self, input_vects):

        """

        Maps each input vector to the relevant neuron in the SOM

        grid.

        'input_vects' should be an iterable of 1-D NumPy arrays with

        dimensionality as provided during initialization of this SOM.

        Returns a list of 1-D NumPy arrays containing (row, column)

        info for each input vector(in the same order), corresponding

        to mapped neuron.

        """



        if not self._trained:

            raise ValueError("SOM not trained yet")



        to_return = []

        for vect in input_vects:

            min_index = min([i for i in range(len(self._weightages))],

                            key=lambda x: np.linalg.norm(vect-

                                                         self._weightages[x]))

            to_return.append(self._locations[min_index])



        return to_return
#subtask1

from matplotlib import pyplot as plt



#Training inputs for RGBcolors

def peformClustering(inputData):

    print("fire clustering")

    

    dicton =  groupTournamentsToTeams(dataSetDF)

    colors = np.array(list(dicton.values()))#[:100,[2,3,4,5,6,7,8,9]]

    color_names = np.array(list(dicton.keys()))

    colors = colors / colors.max()

        #Train a 20x30 SOM with 40 iterations

    som = SOM(1, 2, len(colors[0]), 5)

    som.train(colors)

        

    print("Teams amount is " + str(len(colors)))

    print("#Get output grid")

    image_grid = np.array(som.get_centroids())

        

    print("#Map colours to their closest neurons")

    colors = colors.astype(float)

    mapped = som.map_vects(colors)

    #print(image_grid)

    #print(image_grid.argmax(axis=1))

    #Plot

    plt.imshow(image_grid.argmax(axis=1))

    plt.title('Color SOM')

    #print(mapped)

    firstTeams = []

    secondTeams = []

    for i, m in enumerate(mapped):

        plt.text(m[1] * 100, m[0] * 10, color_names[i], ha='center', va='center',

                 bbox=dict(facecolor='white', alpha=0.5, lw=0))

        if m[1] < 0.5:

            firstTeams+=[color_names[i]]

        else:

            secondTeams+=[color_names[i]]

    plt.show()

    inputData = np.c_[ inputData, np.ones(inputData.shape[0])] 

    index = 0

    for match in inputData:

        teamOneName = match[2]

        teamTwoName = match[5]

        inputData[index] = inputData[index].astype(object)

        if firstTeams.count(teamOneName) > 0:

            match[10] = "league1"

        if secondTeams.count(teamOneName) > 0:

            match[10] = "league2"

        inputData[index] = match

        index += 1

    return inputData
#Ключ - название команды. Значение - массив турниров, на котором она сыграла

def groupTournamentsToTeams(dataSetC):

    dataSet = np.copy(dataSetC)

    dataSet = tokenizeTeamsAndTournament(dataSet)

    teamsDict = {}

    maxLen = 0

    for match in dataSet:

        try:

            if (teamsDict[match[2]].count(match[1])):

                teamsDict[match[2]] += [match[1]]

            teamsDict[match[2]].sort()

        except:

            teamsDict[match[2]] = [match[1]]

            teamsDict[match[2]].sort()

        if (len(teamsDict[match[2]]) > maxLen):

            maxLen = len(teamsDict[match[2]])

            

        try:

            if (teamsDict[match[5]].count(match[1])):

                teamsDict[match[5]] += [match[1]]

            teamsDict[match[5]].sort()

        except:

            teamsDict[match[5]] = [match[1]]

            teamsDict[match[5]].sort()

        if (len(teamsDict[match[5]]) > maxLen):

            maxLen = len(teamsDict[match[5]])

            

            

    for key, value in teamsDict.items():

        if len(value) < maxLen:

            for i in range(0, maxLen - len(value)):

                teamsDict[key] += [0]

    return teamsDict
def tokenizeTeamsAndTournament(dataSet):

    for match in dataSet:

        #match[0] = match[0].replace(" ", "")

        match[1] = match[1].replace(" ", "")

        #match[2] = match[2].replace(" ", "")

        match[3] = str(match[3]).replace(",", ".")

        #match[5] = match[5].replace(" ", "")

        match[6] = str(match[6]).replace(",", ".")

        match[8] = match[8].replace(" ", "")

    tokenizerMatch_id = Tokenizer(num_words = 20000, filters="")

    tokenizerTournament_id = Tokenizer(num_words = 20000,filters="")

    tokenizerTeams = Tokenizer(num_words = 20000,filters="")

    tokenizerMatch_id.fit_on_texts(dataSet[:,0].transpose())

    tokenizerTournament_id.fit_on_texts(dataSet[:,1].transpose())

    tokenizerTeams.fit_on_texts(np.concatenate((dataSet[:, 2], dataSet[:,5])).transpose())

    for match in dataSet:

        #match[0] = tokenizerMatch_id.word_index[match[0].lower()]

        match[1] = tokenizerTournament_id.word_index[match[1].lower()]

        #match[2] = tokenizerTeams.word_index[match[2].lower()]

        #match[5] = tokenizerTeams.word_index[match[5].lower()]

        #match[8] = time.mktime(datetime.datetime.strptime(match[8].lower(), "%d.%m.%Y%H:%M").timetuple())

    return dataSet

    

    
def createStringRepresentation(dataSet):

    matches = []

    for match in dataSet:

        print(match)

        tournament = match[1].replace(" ", "")

        team1 = match[2].replace(" ", "")

        team1Odd = match[3]

        team1Score = match[4]

        team2 = match[5].replace(" ", "")

        team2Odd = match[6]

        team2Score = match[7]

        resultRelatedToTeam1 = ""

        if match[9] > 0.5:

            team1Score = 1 if math.isnan(team1Score) else int(team1Score)

            team2Score = 0 if math.isnan(team2Score) else int(team2Score)

            resultRelatedToTeam1 = "win"

        else:

            team1Score = 0 if math.isnan(team1Score) else int(team1Score)

            team2Score = 1 if math.isnan(team2Score) else int(team2Score)

            resultRelatedToTeam1 = "lose"

        matchDescription = team1 + " " + str(team1Odd) + " " + team2 + " " + str(team2Odd) + " " + tournament + " " + str(team1Score) + " " + str(team2Score) + " " + resultRelatedToTeam1

        matches += [matchDescription]

    return matches

        
def plotHistogramsForLeagues(dataSet):

    def splitDatasetByLeague(dataSet):

        dataSetLeague1 = []

        dataSetLeague2 = []

        indexFirst = 0

        indexSecond = 0

        for match in dataSet:

                if(match[10] == "league1"):

                    dataSetLeague1 += [match]

                    indexFirst +=1

                elif(match[10] == "league2"):

                    dataSetLeague2 += [match]

                    indexSecond +=1

        return dataSetLeague1,dataSetLeague2



    def plotExtendedHistogramForLeague(dataSet, numberOfLeague):

        dictWins = {}

        for match in dataSet:

            if str(match[2]) not in dictWins:

                dictWins[str(match[2])] = {"matches": 0, "wins": 0}

            if str(match[5]) not in dictWins:

                dictWins[str(match[5])] = {"matches": 0, "wins": 0}



            dictWins[str(match[2])]["matches"] +=1

            dictWins[str(match[5])]["matches"] +=1



            if(match[9] == 1):

                dictWins[str(match[2])]["wins"]+=1

            elif(match[9] == 0):

                dictWins[str(match[5])]["wins"]+=1

        

        dictWinsDist = {}

        

        dictWinsDist["<3"] = 0

        dictWinsDist["3-6"] = 0

        dictWinsDist["7-9"] = 0

        dictWinsDist[">10"] = 0

        

        for key in dictWins:

            if(dictWins[key]["wins"] < 3):

                dictWinsDist["<3"]+=1

            elif(3<=dictWins[key]["wins"]<=6):

                dictWinsDist["3-6"]+=1

            elif(7<=dictWins[key]["wins"]<=9):

                dictWinsDist["7-9"]+=1

            elif(dictWins[key]["wins"]>=10):

                dictWinsDist[">10"]+=1



        plt.bar(list(dictWinsDist.keys()),dictWinsDist.values())

        plt.xlabel('Number of wins for teams', fontsize=16)

        plt.ylabel('Number of teams', fontsize=16)

        plt.title("Histogram {}".format(numberOfLeague))

        plt.show()

        

        dictDistribution = {}



        dictDistribution["<6m_<50%w"] = 0

        dictDistribution["<6m_>50%w"] = 0

        dictDistribution["7-12m_<50%w"] = 0

        dictDistribution["7-12m_>50%w"] = 0

        dictDistribution["13-18m_<50%w"] = 0

        dictDistribution["13-18m_>50%w"] = 0

        dictDistribution[">18m_<50%w"] = 0

        dictDistribution[">18m_>50%w"] = 0



        for key in dictWins:

            if((dictWins[key]["matches"] < 6) and (dictWins[key]["wins"]/dictWins[key]["matches"] * 100 < 50)):

                dictDistribution["<6m_<50%w"]+=1

            elif((dictWins[key]["matches"] < 6) and (dictWins[key]["wins"]/dictWins[key]["matches"] * 100 > 50)):

                dictDistribution["<6m_>50%w"]+=1

            elif((6<=dictWins[key]["matches"] <= 12) and (dictWins[key]["wins"]/dictWins[key]["matches"] * 100 < 50)):

                dictDistribution["7-12m_<50%w"]+=1

            elif((6<=dictWins[key]["matches"] <= 12) and (dictWins[key]["wins"]/dictWins[key]["matches"] * 100 > 50)):

                dictDistribution["7-12m_>50%w"]+=1

            elif((13<=dictWins[key]["matches"] <= 18) and (dictWins[key]["wins"]/dictWins[key]["matches"] * 100 < 50)):

                dictDistribution["13-18m_<50%w"]+=1

            elif((13<=dictWins[key]["matches"] <= 18) and (dictWins[key]["wins"]/dictWins[key]["matches"] * 100 > 50)):

                dictDistribution["13-18m_>50%w"]+=1

            elif((dictWins[key]["matches"]>18) and (dictWins[key]["wins"]/dictWins[key]["matches"] * 100 < 50)):

                dictDistribution[">18m_<50%w"]+=1

            elif((dictWins[key]["matches"]>18) and (dictWins[key]["wins"]/dictWins[key]["matches"] * 100 > 50)):

                dictDistribution[">18m_>50%w"]+=1 



        plt.figure(figsize=(20,5))

        keys = ["<6matches_<50%wins","<6matches_>50%wins","7-12matces_<50%wins","7-12matches_>50%wins","13-18matches_<50%wins","13-18matches_>50%wins",">18matches_<50%wins",">18matches_>50%wins"]

        pos = np.arange(len(keys))

        plt.bar(pos,dictDistribution.values(),color=['orange'],edgecolor='black')

        plt.xlabel('Percentage of wins regarding to number of matches', fontsize=16)

        plt.ylabel('Number of teams', fontsize=16)

        plt.title("Number of teams {} with more and less than 50% wins".format(numberOfLeague))

        plt.xticks(pos, keys)

        plt.show()

    

    dataSetLeague1, dataSetLeague2 = splitDatasetByLeague(dataSet)

    plotExtendedHistogramForLeague(dataSetLeague1, "for League 1")

    plotExtendedHistogramForLeague(dataSetLeague2, "for League 2")

    print("Hist building finished ^-^")
def groupDataSetByTeams(dataSet):

    groupedMatches = {}

    for matches in dataSet:

        try:

            groupedMatches[matches[2]] += [matches]

        except:

            groupedMatches[matches[2]] = [matches]

        try:

            groupedMatches[matches[5]] += [matches]

        except:

            groupedMatches[matches[5]] = [matches]

    print("Match arranging is finished ^_^")

    return groupedMatches

    
def countNewELO(team1ELO, team2ELO, isFirstWin):

    expectation1 = 1 / (1 + 10**((team2ELO - team1ELO) / 400))

    newElo1 = team1ELO + 20 * (isFirstWin - expectation1)

    

    expectation2 = 1 / (1 + 10**((team1ELO - team2ELO) / 400))

    newElo2 = team2ELO +  20 * ((1 - isFirstWin) - expectation2)



    return (newElo1, newElo2)
def countELOForDataSet(dataSet):

    for match in dataSet:

        match[2] = match[2].replace(" ", "")

        match[5] = match[5].replace(" ", "")

    dataSet = np.c_[dataSet, np.ones(dataSet.shape[0]), np.ones(dataSet.shape[0])] 

    teamsEloRanking = {}

    index = 0

    for match in dataSet:

        team1Elo = 0

        team2Elo = 0

        try:

            team1Elo = teamsEloRanking[match[2]]

        except:

            team1Elo = 1400

            teamsEloRanking[match[2]] = 1400 

        try:

            team2Elo = teamsEloRanking[match[5]]

        except:

            team2Elo = 1400

            teamsEloRanking[match[5]] = 1400 

        dataSet[index][11] = team1Elo

        dataSet[index][12] = team2Elo

        firstWin = match[9]

        newElo1, newElo2 = countNewELO(team1Elo, team2Elo, firstWin)

        teamsEloRanking[match[2]] = newElo1

        teamsEloRanking[match[5]] = newElo2

        index += 1

    print("ELO counting finished ^_^")

    return (dataSet, teamsEloRanking)
def plotGraphicsForELO(dataSet):

    arrayForGraphics = []

    dictionaryForRange = {"0-20":[0, 0], "20-40":[0, 0], "40-60":[0,0], "60-80":[0,0], "80-100":[0,0], "100-120":[0,0], ">120":[0,0]}

    for match in dataSet:

        eloDiff = match[11] - match[12]

        isFavoriteWin = 1

        if eloDiff > 0:

            if match[9] > 0.5:

                isFavoriteWin = 1

            else:

                isFavoriteWin = 0

        else:

            if match[9] > 0.5:

                isFavoriteWin = 0

            else:

                isFavoriteWin = 1

        eloDiff = abs(eloDiff)

        arrayForGraphics += [[match[3], match[6], isFavoriteWin, eloDiff]]

        if eloDiff < 20:

            dictionaryForRange["0-20"][0] += isFavoriteWin

            dictionaryForRange["0-20"][1] += 1

        elif eloDiff < 40:

            dictionaryForRange["20-40"][0] += isFavoriteWin

            dictionaryForRange["20-40"][1] += 1

        elif eloDiff < 60:

            dictionaryForRange["40-60"][0] += isFavoriteWin

            dictionaryForRange["40-60"][1] += 1

        elif eloDiff < 80:

            dictionaryForRange["60-80"][0] += isFavoriteWin

            dictionaryForRange["60-80"][1] += 1

        elif eloDiff < 100:

            dictionaryForRange["80-100"][0] += isFavoriteWin

            dictionaryForRange["80-100"][1] += 1

        elif eloDiff < 120:

            dictionaryForRange["100-120"][0] += isFavoriteWin

            dictionaryForRange["100-120"][1] += 1

        else:

            dictionaryForRange[">120"][0] += isFavoriteWin

            dictionaryForRange[">120"][1] += 1

        

    dataForHyst = []

    for items in dictionaryForRange.values():

        dataForHyst += [int(100 * items[0] / items[1])]

        

    xAxis = ("0-20", "20-40", "40-60", "60-80", "80-100", "100-120", ">120")

    num_bins = len(dictionaryForRange.keys())

    

    y_pos = np.arange(len(xAxis))

    

    plt.bar(y_pos, dataForHyst, align='center', alpha=0.5)

    plt.xticks(y_pos, xAxis)

    plt.xlabel('Разница ELO')

    plt.ylabel('Вероятность победы фаворита')

    plt.title(r'Зависимость побед фаворитов от разницы ELO ')

    plt.show()

    

    #indexToRemove = []

    #index = 0

    #for stat in arrayForGraphics:

    #    if float(stat[0]) < 1.0 or float(stat[1]) < 1.0:

    #        indexToRemove += [index]

    #    index += 1

    #removedAmount = 0    

    #for toRemove in indexToRemove:

    #    arrayForGraphics.pop(toRemove - removedAmount)

    #    removedAmount += 1

    #arrayForGraphics = np.array(arrayForGraphics)

    #arrayForGraphics = arrayForGraphics[np.argsort(arrayForGraphics[:, 3])]

    #plt.plot(arrayForGraphics[:1000,3], arrayForGraphics[:1000,0])

    #plt.plot([0, 120])

    #plt.show()

    #arrayForGraphics = np.array(arrayForGraphics)

    #plt.plot(arrayForGraphics[:1000,3], arrayForGraphics[:1000,1])

    #plt.plot([0, 120])

    #plt.show()

    print("Elo Visualisation Finished ^_^")

    
def createModel(vocab_size, tournamentSize):

    inputTeam1 = tf.keras.Input(shape=(1,))

    inputTeam2 = tf.keras.Input(shape=(1,))

    inputTournament = tf.keras.Input(shape=(1,))

    inputEloAndOdds = tf.keras.Input(shape=(2,))

    

    team1Branch = tf.keras.layers.Embedding(vocab_size,64)(inputTeam1)

    #team1Branch = tf.keras.layers.Flatten()(team1Branch)

    team1Branch = tf.keras.Model(inputs=inputTeam1, outputs=team1Branch)

    

    team2Branch = tf.keras.layers.Embedding(vocab_size, 64)(inputTeam2)

    #team2Branch = tf.keras.layers.Flatten()(team2Branch)

    team2Branch = tf.keras.Model(inputs=inputTeam2, outputs=team2Branch)

    

    tournamentBranch = tf.keras.layers.Embedding(tournamentSize, 64)(inputTournament)

    #tournamentBranch = tf.keras.layers.Flatten()(tournamentBranch)

    tournamentBranch = tf.keras.Model(inputs=inputTournament, outputs=tournamentBranch)

   

    eloAndOdds = tf.keras.layers.Dense(1, activation="softmax")(inputEloAndOdds)

    #eloAndOdds = tf.keras.layers.Dense(1, activation="softmax")(eloAndOdds)

    eloAndOdds = tf.keras.Model(inputs = inputEloAndOdds, outputs = eloAndOdds)

    

    combined = tf.keras.layers.Add()([team1Branch.output, team2Branch.output, tournamentBranch.output, eloAndOdds.output])



    result = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(combined)

    result = tf.keras.layers.Dense(2, activation="softmax")(result)

    

    model = tf.keras.Model(inputs=[team1Branch.input, team2Branch.input, tournamentBranch.input, eloAndOdds.input], outputs=result)

    

    return model
def tokenizeTeams(dataSet):

    for match in dataSet:

        match[1] = match[1].replace(" ", "")

        match[2] = match[2].replace(" ", "")

        match[5] = match[5].replace(" ", "")

        match[3] = str(match[3]).replace(",", ".")

        match[6] = str(match[6]).replace(",", ".")

        if float(match[3]) < 1.0:

            match[3] = 1.8

        if float(match[6]) < 1.0:

            match[6] = 1.8

    tokenizerTournament_id = Tokenizer(num_words = 20000,filters="")

    tokenizerTeams = Tokenizer(num_words = 20000,filters="")

    tokenizerTeams.fit_on_texts(np.concatenate((dataSet[:, 2], dataSet[:,5])).transpose())

    tokenizerTournament_id.fit_on_texts(dataSet[:, 1])

    for match in dataSet:

        match[1] = tokenizerTournament_id.word_index[match[1].lower()]

        match[2] = tokenizerTeams.word_index[match[2].lower()]

        match[5] = tokenizerTeams.word_index[match[5].lower()]

        match[8] = time.mktime(datetime.datetime.strptime(match[8].lower(), "%d.%m.%Y %H:%M").timetuple())

        match[8] = int(match[8])

    print("Team tokenizing finished ^_^")

    return (dataSet, tokenizerTournament_id, tokenizerTeams)


def writeResultsToFile(dataSetDFTest,prediction):

    print("Writing fired -_-")

    matches = []

    resultArray = []

    for (match, result) in zip(dataSetDFTest,prediction):

        name = match[0]

        matches.append(name)

        coef = result[0]

        if coef >= 0.5:

            resultArray.append(1)

        else:

            resultArray.append(0)

    resultFrame = pd.DataFrame(data = {'match_id':matches, 'team1_win':resultArray})

    resultFrame.to_csv(path_or_buf = '/kaggle/working/result' + str(datetime.datetime.now()) + '.csv', index=False)

    print("Result saving is done ^_^")
def tokenizeTestSelection(dataSet, tokenizerTournament_id,tokenizerTeams):

    for match in dataSet:

        match[1] = match[1].replace(" ", "")

        match[2] = match[2].replace(" ", "")

        match[4] = match[4].replace(" ", "")

        match[3] = str(match[3]).replace(",", ".")

        match[5] = str(match[5]).replace(",", ".")

        if float(match[3]) < 1.0:

            match[3] = 1.8

        if float(match[5]) < 1.0:

            match[5] = 1.8

    tokenizerTournament_id.fit_on_texts(dataSet[:, 1])

    tokenizerTeams.fit_on_texts(np.concatenate((dataSet[:, 2], dataSet[:,4])).transpose())

    for match in dataSet:

        match[1] = tokenizerTournament_id.word_index[match[1].lower()]

        match[2] = tokenizerTeams.word_index[match[2].lower()]

        match[4] = tokenizerTeams.word_index[match[4].lower()]

        match[6] = time.mktime(datetime.datetime.strptime(match[6].lower(), "%d.%m.%Y %H:%M").timetuple())

        match[6] = int(match[6])

    print("Team tokenizing finished ^_^")

    return dataSet
def injectELOtoTestDataSet(dataSet, eloRankings):

    dataSet = np.c_[ dataSet, np.ones(dataSet.shape[0]), np.ones(dataSet.shape[0])] 

    alreadyInjected = []

    for match in dataSet:

        match[2] = match[2].replace(" ", "")

        match[4] = match[4].replace(" ", "")

        try:

            recordedElo = eloRankings[match[2]]

            if alreadyInjected.count(recordedElo) > 0:

                match[7] = 0

            else:

                match[7] = recordedElo

                alreadyInjected += [match[2]]

        except:

            eloRankings[match[2]] = 1400

            match[7] = 1400

            

        try:

            recordedElo = eloRankings[match[4]]

            if alreadyInjected.count(recordedElo) > 0:

                match[8] = 0

            else:

                match[8] = recordedElo

                alreadyInjected += [match[2]]

        except:

            eloRankings[match[4]] = 1400

            match[8] = 1400

    return dataSet
import operator



def plotEloChangeGraphics(eloDict, dataSetTest, teamsWordIndex):

    eloDict = sorted(eloDict.items(), key=operator.itemgetter(1),reverse= True)

    topsTuple = eloDict[:15]

    index = 0

    for topc in topsTuple:

        newtopc = list(topc) 

        newtopc[0] = newtopc[0].lower().replace(" ", "")

        topsTuple[index] = tuple(newtopc)

        index += 1

    

    eloTeamHistory = {}

    for top in topsTuple:

        eloTeamHistory[teamsWordIndex[top[0]]] = [[top[1], 0]]

    

    topTeams = list(eloTeamHistory.keys())

    for match in dataSetTest:

        timeStamp = match[6] #time.mktime(datetime.datetime.strptime(match[8].lower(), "%d.%m.%Y %H:%M").timetuple())

        if topTeams.count(match[2]) > 0:

            eloTeamHistory[match[2]] += [[match[7], timeStamp]]

        if topTeams.count(match[4]) > 0:

            eloTeamHistory[match[4]] += [[match[8], timeStamp]]

    

    for key, value in eloTeamHistory.items():

        if len(eloTeamHistory[key]) > 1:

            eloTeamHistory[key][0][1] = eloTeamHistory[key][1][1] - 10000

        else:

            eloTeamHistory[key][0][1] = 1576427400

    

    ind = 0

    for keys, values in eloTeamHistory.items():

        reshapedTest = np.array(values).reshape(len(values), 2)

        reshapedTest = reshapedTest[np.argsort(reshapedTest[:, 1])]

        plt.subplot(1, 1, 1)

        plt.plot(reshapedTest[:, 1], reshapedTest[:, 0], label = topsTuple[ind][0])

        ind += 1

        plt.title("История ELO")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.show()    

    print("Elo diff plotting finished ^_^")

    

def foundTeamsName(teamsWordIndex, teamOneToken, teamTwoToken):

    teamOneName = ""

    for key, value in self.teamsWordIndex.items():

        if value == teamOneToken:

            teamOneName = key

        if value == teamTwoToken:

            teamTwoName = key

    return (teamOneName, teamTwoName)
class ELORecountCallback(tf.keras.callbacks.Callback):

    

    dataSetTrain = []

    eloByTeams = {}

    teamsWordIndex = {}

    

    def __init__(self, dataSetTrain, eloByTeams, teamsWordIndex):

        super(ELORecountCallback, self).__init__()

        self.dataSetTrain = dataSetTrain

        self.eloByTeams = eloByTeams

        self.teamsWordIndex = teamsWordIndex

    

    def on_predict_batch_begin(self, batch, logs=None):

        matchToRate = self.dataSetTrain[batch]

        teamOneName, teamTwoName = self.foundTeamsName(matchToRate[2], matchToRate[4])

         

        try:

            self.dataSetTrain[batch][7] = self.eloByTeams[teamOneName]

        except:

            self.eloByTeams[teamOneName] = 1400

            self.dataSetTrain[batch][7] = 1400

            

        try:

            self.dataSetTrain[batch][8] = self.eloByTeams[teamTwoName]

        except:

            self.eloByTeams[teamTwoName] = 1400

            self.dataSetTrain[batch][8] = 1400

        

    def on_predict_batch_end(self, batch, logs=None):

        isTeamOneWin = int(round(logs["outputs"][0][0][0]))

        ratedMatch = self.dataSetTrain[batch]

        teamOneName, teamTwoName = self.foundTeamsName(ratedMatch[2], ratedMatch[4])

        team1Elo = self.eloByTeams[teamOneName]

        team2Elo = self.eloByTeams[teamTwoName]

        newElo1, newElo2 = countNewELO(team1Elo, team2Elo, isTeamOneWin)

        self.eloByTeams[teamOneName] = newElo1

        self.eloByTeams[teamTwoName] = newElo2

    

    def foundTeamsName(self, teamOneToken, teamTwoToken):

        teamOneName = ""

        teamTwoName = ""

        for key, value in self.teamsWordIndex.items():

            if value == teamOneToken:

                teamOneName = key

            if value == teamTwoToken:

                teamTwoName = key

        return (teamOneName, teamTwoName)
#Reading data

dataSetPD = pd.read_csv(filepath_or_buffer = "/kaggle/input/game-predict-minor-2020/train_matches_semicolumn1.csv", sep = ";")

dataSetPD = dataSetPD.drop(columns = ["tournament"])

dataSetDF = dataSetPD.to_numpy(dtype = object)



#CRUTCH: некоторые данные приходят без поля team1_win, нужно проверять вручную по счёту

index = 0

for match in dataSetDF:

    if (math.isnan(match[9])):

        if match[4] > match[7]:

            dataSetDF[index][9] = 1

        else:

            dataSetDF[index][9] = 0

    index += 1



print("Data reading is done ^_^ ")
dataSetDF = peformClustering(dataSetDF)

print("Clusterisation done ^_^")

plotHistogramsForLeagues(dataSetDF)
groups = groupDataSetByTeams(dataSetDF)



dataSetDF, eloByTeams = countELOForDataSet(dataSetDF)

#Prepare data to fit

inputMatrix, tournamentTokenizer, teamTokenizer = tokenizeTeams(dataSetDF)

teamsWordIndex = teamTokenizer.word_index

inputTeam1 = inputMatrix[:,2]#[2,3]]

inputTeam2 = inputMatrix[:,5]#[5,6]]

inputTournament = dataSetDF[:,1]#[1, 8]]

elosAndOdds = dataSetDF[:,[11,12]]

outputMatrix = dataSetDF[:,[9]]

outputMatrix = to_categorical(outputMatrix)
#Prepare Model

vocab_size = max(np.concatenate((inputMatrix[:, 2], inputMatrix[:,5]))) + 5000

tournamentSize = int(max(inputMatrix[:, 1])) + 5000

model = createModel(vocab_size, tournamentSize)



sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile( loss = tf.keras.losses.BinaryCrossentropy(), 

               optimizer = "adam", 

               metrics=['accuracy']

             )

#model.compile(loss='mean_squared_error', optimizer='SGD')

print("compiling done ^_^")
#Model Fitting

groups = groupDataSetByTeams(dataSetDF)

index = 1

overallCount = len(list(groups.values()))



#model.fit([inputTeam1, inputTeam2, inputTournament, elosAndOdds], outputMatrix, epochs=5, batch_size=100, verbose=1)

for groupedM in groups.values():

    groupedM = np.array(groupedM)

    inputTeam1Gr = groupedM[:,2]#[2,3]]

    inputTeam2Gr = groupedM[:,5]#[5,6]]

    inputTournamentGr = groupedM[:,1]#[1, 8]]

    elosAndOddsGr = groupedM[:,[11,12]]

    outputMatrixGr = groupedM[:,[9]]

    outputMatrixGr = to_categorical(outputMatrixGr)

    if (np.amax(outputMatrixGr) != np.amin(outputMatrixGr)):

        model.fit([inputTeam1Gr, inputTeam2Gr, inputTournamentGr, elosAndOddsGr], outputMatrixGr, epochs=4, batch_size=1, verbose=0)

    print(str(index) + " of " + str(overallCount) + " ^_^")

    index+=1

_, accuracy = model.evaluate([inputTeam1, inputTeam2, inputTournament, elosAndOdds], outputMatrix)

print(accuracy)

print("fitting done ^_^")
#Perform predictions

dataSetTest = pd.read_csv(filepath_or_buffer = "/kaggle/input/game-predict-minor-2020/test_matches1.csv", sep = ";")

dataSetTest = dataSetTest.drop(columns = ["tournament"])

dataSetDFTest = dataSetTest.to_numpy(dtype = object)

dataSetDFTest = injectELOtoTestDataSet(dataSetDFTest, eloByTeams)

dataSetDFTest = tokenizeTestSelection(dataSetDFTest,tournamentTokenizer, teamTokenizer)

oldElo = eloByTeams.copy()

callback = ELORecountCallback(dataSetDFTest,eloByTeams, teamsWordIndex)

prediction = model.predict([dataSetDFTest[:,2], dataSetDFTest[:,4], dataSetDFTest[:,1], dataSetDFTest[:,[7,8]]], batch_size = 1, callbacks=[callback])

print("Prediction performing finished ^_^")
#Result Saving

writeResultsToFile(dataSetDFTest,prediction)

#plotEloGraphics

plotEloChangeGraphics(oldElo, dataSetDFTest, teamsWordIndex)