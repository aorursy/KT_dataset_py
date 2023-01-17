# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.shape
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
y = df['target']
y
X = df.drop('target',axis=1)
X
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()  
scaler.fit(X)    
X_normalized = pd.DataFrame(scaler.transform(X),columns = X.columns)
X_normalized
X_train, X_test, y_train, y_test = train_test_split(X_normalized, 
                                                    y, test_size=0.30, 
                                                    random_state=101)
X_train.shape
logmodel = LogisticRegression(random_state=101)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print("Accuracy = "+ str(accuracy_score(y_test,predictions)))
#defining various steps required for the genetic algorithm
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],y_train)
        predictions = logmodel.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

# Random Mutation
def mutation(pop_after_cross,n_feat):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        chromosome = list(np.random.randint(0, 2, size=n_feat, dtype=np.bool))
        population_nextgen.append(chromosome)
    return population_nextgen

# Mutation using Bit Inversion
def mutation_2(pop_after_cross,n_feat):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    return population_nextgen

def generations(size,n_feat,n_parents,n_gen,X_train,
                                   X_test, y_train, y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation_2(pop_after_cross,n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score
chromo,score=generations(size=200,n_feat=13,n_parents=100,
                         n_gen=10,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

logmodel.fit(X_train.iloc[:,chromo[-1]],y_train)
predictions = logmodel.predict(X_test.iloc[:,chromo[-1]])
print("Accuracy score after genetic algorithm is= "+str(accuracy_score(y_test,predictions)))
plt.plot([i for i in range(10)],score,marker='o',linestyle='dashed',color='red',markerfacecolor='blue',markersize=10)
plt.xlabel("Generations")
plt.ylabel("Accuracy")
plt.show()
data = []
for i in chromo:
    j = list(np.array(i,dtype=int))
    data.append(j)
pd.DataFrame(data,columns=X_train.columns)