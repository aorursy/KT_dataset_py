
# Code source: Prateek Chanda

# importing the modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score
iris = datasets.load_iris()
x = iris.data
y = iris.target

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target_names'])

data1.head()
sns.pairplot(data1, hue='target_names', size=2.5);
sns.set()
yval = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    yval.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), yval)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Score') #within cluster sum of squares
plt.show()
silh = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    cluster_labels = kmeans.fit_predict(x)
    s = silhouette_score(x, labels = cluster_labels)
    silh.append(s)
    
plt.plot(range(2, 11), silh)
plt.title('Silhoutte Index Values')
plt.xlabel('Number of clusters')
plt.ylabel('Score') #within cluster sum of squares
plt.show()
## Importing Necessary Modules
import random
from deap import creator, base, tools, algorithms

## creating fitness values and weights
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Each gene (cluster) is assigned 1 or 0
toolbox.register("attr_bool", random.randint, 0, 1)
# total genes in a chromosome(individual) = set to 100 (the more genes the better result)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100) 
# creating population from all the chromosomes
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# eveluate each individual
def evalOneMax(individual):
    return sum(individual),

# evaluation of each chromosome
toolbox.register("evaluate", evalOneMax)
# mating / cross-over of two chromosomes
toolbox.register("mate", tools.cxTwoPoint)
# mutation with mutation probability = 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# tournament selection with size =3
toolbox.register("select", tools.selTournament, tournsize=3)

# creating total population of size 300
population = toolbox.population(n=300)

# next generation size = 40
NGEN=40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.5)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    
# selecting top 10 best chromosomes    
top10 = tools.selBest(population, k=10)
