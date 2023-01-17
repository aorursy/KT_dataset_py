import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard

from time import time

from sklearn.metrics import accuracy_score

import random, operator
#Read train and test csv

train = pd.read_csv("../input/train/train.csv") 

test = pd.read_csv("../input/test/test.csv") 
x=(train[['Type', 'Age', 'Breed1', 'Breed2','Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','Quantity','Fee']])

y=(train[['AdoptionSpeed']])

testx=(test[['Type', 'Age', 'Breed1', 'Breed2','Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','Quantity','Fee']])
train_x,dev_x=x[200:],x[:200] 

train_y,dev_y=y[200:],y[:200]

#Pandas to numpy

train_x=train_x.values

train_y=train_y.values



flat_y = [item for sublist in dev_y.astype(float).values for item in sublist]

#dev_x=dev_x.values

#dev_y=dev_y.values

#Transform targets [0,3,...,4] to [[1,0,0,0,0],[0,0,0,1,0],...,[0,0,0,0,1]]

train_y = tf.keras.utils.to_categorical(train_y, 5)

dev_y = tf.keras.utils.to_categorical(dev_y, 5)



#Create the model

model = tf.keras.Sequential()



model.add(tf.keras.layers.Dense(5, input_shape=(16,),activation='softmax', use_bias=False)) #Dense Layer with softmax activation so it can predict one of the 5 Labels



model.compile(loss=tf.keras.losses.categorical_crossentropy,

              optimizer=tf.keras.optimizers.Adadelta(),

              metrics=['accuracy'])

print(model.summary())
#Train the model

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

bestEpoch=tf.keras.callbacks.ModelCheckpoint("logs/checkpoint", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)



model.fit(train_x, train_y,

          batch_size=1000,

          epochs=2000,

          verbose=1,

          validation_data=(dev_x, dev_y),

          callbacks=[tensorboard,bestEpoch])
model=tf.keras.models.load_model(

    "logs/checkpoint",

    custom_objects=None,

    compile=True

)



pred=model.predict(dev_x)

print(pred.argmax(axis=1)[:5])

print("Dev accuracy:",accuracy_score(pred.argmax(axis=1),flat_y))
class Weight:

    def __init__(self, w):

        self.w = w

    

    

    def __repr__(self):

        return "(" + str(self.w) + ")"
class Fitness:

    def __init__(self, weights):

        self.weights = weights   

    

    def routeFitness(self):

        model.set_weights(toNumpyWeights(self.weights))

        pred=model.predict(dev_x)

        self.fitness=-(1/accuracy_score(pred.argmax(axis=1),flat_y))

        return self.fitness
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):

    pop = initialPopulation(popSize, population)

    

    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    

    for i in range(0, generations):

        print("Iteration ", i+1," of ",generations)

        pop = nextGeneration(pop, eliteSize, mutationRate)

    

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))

    bestRouteIndex = rankRoutes(pop)[0][0]

    bestRoute = pop[bestRouteIndex]

    return bestRoute
def initialPopulation(popSize, weights):

    population = []

    population.append(weights)

    for i in range(0, popSize-1):

        print("\rCreating weights ", i+2," of ",popSize, end='', flush=True)

        population.append(createWeights(weights))

    print("\n")

    return population
def createWeights(weights):

    newWeights = random.sample(weights, len(weights))

    return newWeights
def rankRoutes(population):

    fitnessResults = {}

    for i in range(0,len(population)):

        

        fitnessResults[i] = Fitness(population[i]).routeFitness()



    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
def nextGeneration(currentGen, eliteSize, mutationRate):

    popRanked = rankRoutes(currentGen)

    selectionResults = selection(popRanked, eliteSize)

    matingpool = matingPool(currentGen, selectionResults)

    children = breedPopulation(matingpool, eliteSize)

    print("Generation score: ",1/Fitness(matingpool[0]).routeFitness())

    nextGeneration = mutatePopulation(eliteSize, children, mutationRate)

    return nextGeneration
def selection(popRanked, eliteSize):

    selectionResults = []

    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])

    df['cum_sum'] = df.Fitness.cumsum()

    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    

    for i in range(0, eliteSize):

        selectionResults.append(popRanked[i][0])

    for i in range(0, len(popRanked) - eliteSize):

        pick = 100*random.random()

        for i in range(0, len(popRanked)):

            if pick <= df.iat[i,3]:

                selectionResults.append(popRanked[i][0])

                break

    return selectionResults
def matingPool(population, selectionResults):

    matingpool = []

    for i in range(0, len(selectionResults)):

        index = selectionResults[i]

        matingpool.append(population[index])

    return matingpool
def breed(parent1, parent2):

    child = []

    childP1 = []

    childP2 = []

    

    geneA = int(random.random() * len(parent1))

    geneB = int(random.random() * len(parent1))

    

    startGene = min(geneA, geneB)

    endGene = max(geneA, geneB)





    for i in range(startGene, endGene):

        childP1.append(parent1[i])

    s = set(childP1)   

    childP2 = [item for item in parent2 if item not in s]



    child = childP1 + childP2

    return child


def breedPopulation(matingpool, eliteSize):

    children = []

    length = len(matingpool) - eliteSize

    pool = random.sample(matingpool, len(matingpool))



    for i in range(0,eliteSize):

        children.append(matingpool[i])

    

    for i in range(0, length-eliteSize):

        child = breed(pool[i], pool[len(matingpool)-i-1])

        children.append(child)

    return children
def mutate(individual, mutationRate):

    for swapped in range(len(individual)):

        if (random.random() < mutationRate): 

            swapWith = int(random.random() * len(individual))

            individual[swapWith] = Weight(random.uniform(-1, 1))

        elif(random.random() < mutationRate):

            swapWith = int(random.random() * len(individual))

            

            city1 = individual[swapped]

            city2 = individual[swapWith]

            

            individual[swapped] = Weight((city2.w+city1.w)/2)

            individual[swapWith] = Weight((city2.w+city1.w)/2)

        

    return individual
def mutatePopulation(eliteSize, population, mutationRate):

    mutatedPop = []

    for i in range(0,eliteSize):

        mutatedPop.append(population[i].copy())



    for ind in range(0, len(population)):

        mutatedInd = mutate(population[ind], mutationRate)

        mutatedPop.append(mutatedInd)

    return mutatedPop
def toNumpyWeights(w):

    curr=0

    wNumpy=[]

    for i in range(wSize):

        currW=[w[curr].w,w[curr+1].w,w[curr+2].w,w[curr+3].w,w[curr+4].w]

        curr+=5

        wNumpy.append(currW)

    toReturn = np.array([wNumpy],dtype=np.float32)

    #print(toReturn)

    return toReturn


def wtoww():

    w=model.get_weights()

    #print(w)

    w=list(w)[0]

    ww=[]

    wSize=len(w)

    for i in w:

        for ii in i:

            ww.append(Weight(ii))

    return ww
def evolve(ww):

    w=geneticAlgorithm(population=ww, popSize=100, eliteSize=2, mutationRate=0.01, generations=20)

    w = toNumpyWeights(w)

    return w
def score_model(w):  

    model.set_weights(w)

    pred=model.predict(dev_x)

    print(pred.argmax(axis=1)[:5])

    print("Dev accuracy:",accuracy_score(pred.argmax(axis=1),flat_y))
w=model.get_weights()

#print(w)

w=list(w)[0]

ww=[]

wSize=len(w)

for i in w:

    for ii in i:

        ww.append(Weight(ii))



for i in range(10):

    print(str(i+1)+" of 1000")

    ww=wtoww()

    w=evolve(ww)

    score_model(w)
pred=model.predict(dev_x)

print(pred.argmax(axis=1)[:5])

print("Dev accuracy:",accuracy_score(pred.argmax(axis=1),flat_y))

prediction=model.predict(testx)

yy=(test[['PetID']])



#Save results

final=pd.DataFrame(np.array(prediction.argmax(axis=1)),columns=['AdoptionSpeed'])

final['PetID']=yy

final=final[['PetID','AdoptionSpeed']]

final.to_csv("submission.csv", index=False)