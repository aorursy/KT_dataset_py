# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import operator, math, random, time

import numpy as np

from deap import algorithms, base, creator, tools, gp

from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.naive_bayes import GaussianNB

import multiprocessing
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv("../input/test.csv")
y_train = train_df['target'].values

# GP will be executed using 100 features.

X_train = train_df.drop(['target', 'ID_code'],axis = 1).iloc[:,100:].values

X_test = test_df.drop(['ID_code'],axis = 1).iloc[:,100:].values
# create 1 feature

max_features = 103

# maximum number of iteration: iteration will be terminated if maximum number of features reaches 'max_features'.

max_iterations = 100

# number of generation

num_generations = 3
n_features = X_train.shape[1]

clf = GaussianNB()

# initial AUC score 

prev_auc = np.mean(cross_val_score(clf, X_train, y_train, scoring="roc_auc", cv=5, n_jobs=-1))

print('initial AUC: ', prev_auc)
def protectedDiv(left, right):

    eps = 1.0e-7

    tmp = np.zeros(len(left))

    tmp[np.abs(right) >= eps] = left[np.abs(right) >= eps] / right[np.abs(right) >= eps]

    tmp[np.abs(right) < eps] = 1.0

    return tmp



# set random seed

random.seed(123)



# define tree structure that maximize fitness

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)





# main

# 5-fold CV scores(AUC) will be stored in 'results' array.

# functional definitions of generated features will be stored in 'exprs' array.

results = []

exprs = []

for i in range(max_iterations):

    # define mathematical operators that are available in the tree structure.

    pset = gp.PrimitiveSet("MAIN", n_features)

    pset.addPrimitive(operator.add, 2)

    pset.addPrimitive(operator.sub, 2)

    pset.addPrimitive(operator.mul, 2)

    pset.addPrimitive(protectedDiv, 2)

    pset.addPrimitive(operator.neg, 1)

    pset.addPrimitive(np.cos, 1)

    pset.addPrimitive(np.sin, 1)

    pset.addPrimitive(np.tan, 1)



    # set default value of the function

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)



    # set objective function

    # define function that evaluate 5-fold CV score

    def eval_genfeat(individual):

        func = toolbox.compile(expr=individual)

        features_train = [X_train[:,i] for i in range(n_features)]

        new_feat_train = func(*features_train)

        X_train_tmp = np.c_[X_train, new_feat_train]

        return np.mean(cross_val_score(clf, X_train_tmp, y_train, scoring="roc_auc", cv=5)),



    # define evaluation, selection, intersection and mutation

    toolbox.register("evaluate", eval_genfeat)

    toolbox.register("select", tools.selTournament, tournsize=10)

    toolbox.register("mate", gp.cxOnePoint)

    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)

    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



    # define restriction for intersection and mutation.

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5)) 



    # execute main process as multiprocess

    pool = multiprocessing.Pool()

    toolbox.register("map", pool.map)

    # set population

    pop = toolbox.population(n=100)

    # create instance that store best solution for each loop

    hof = tools.HallOfFame(1)



    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    mstats.register("avg", np.mean)

    mstats.register("std", np.std)

    mstats.register("min", np.min)

    mstats.register("max", np.max)



    # execution

    start_time = time.time()

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, num_generations, stats=mstats, halloffame=hof, verbose=True)

    end_time = time.time()



    # store best score and solution

    best_expr = hof[0]

    best_auc = mstats.compile(pop)["fitness"]["max"]



    # add new feature to the training data if that feature contributed to break the current best score.

    if prev_auc < best_auc:

        # add new feature to the training data

        func = toolbox.compile(expr=best_expr)

        features_train = [X_train[:,i] for i in range(n_features)]

        features_test = [X_test[:,i] for i in range(n_features)]

        new_feat_train = func(*features_train)

        X_train = np.c_[X_train, new_feat_train]

        new_feat_test = func(*features_test)

        X_test = np.c_[X_test, new_feat_test]

        

        # replace the best score

        prev_auc = best_auc

        n_features += 1

        

        # store functional definition of newly created feature.

        exprs.append(best_expr)



        # terminate the loop if the number of the features reaches 'max_features'

        if n_features >= max_features:

            break
print('AUC after GP: ', best_auc)
new_feat_train = X_train[:,100:]

new_feat_test = X_test[:,100:]
import pickle 

with open("./exprs_100_200.pkl", "wb") as f:

    pickle.dump(exprs, f)
# save newly created features

with open("./new_feat_train.pkl", "wb") as f:

    pickle.dump(new_feat_train, f)

with open("./new_feat_test.pkl", "wb") as f:

    pickle.dump(new_feat_test, f)