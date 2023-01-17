import time

import numpy as np

import pandas as pd

import timeit

import random 



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix



from sklearn.model_selection import train_test_split

    

from bayes_opt import BayesianOptimization

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from hyperopt import  tpe,hp,fmin







import warnings

warnings.simplefilter('ignore')
train_df = pd.read_csv('../input/career-con-2019/X_train.csv')

test_df = pd.read_csv('../input/career-con-2019/X_test.csv')

target = pd.read_csv('../input/career-con-2019/y_train.csv')

sub = pd.read_csv('../input/career-con-2019/sample_submission.csv')



# train = pd.read_csv("../input/titanic/train.csv")
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z

def feature_eng(data):

    data['total_angular_velocity'] = (data['angular_velocity_X'] ** 2 + data['angular_velocity_Y'] ** 2 + data['angular_velocity_Z'] ** 2) ** 0.5

    data['total_linear_acceleration'] = (data['linear_acceleration_X'] ** 2 + data['linear_acceleration_Y'] ** 2 + data['linear_acceleration_Z'] ** 2) ** 0.5

    data['acc_vs_vel'] = data['total_linear_acceleration'] / data['total_angular_velocity']



    x, y, z, w = data['orientation_X'].tolist(), data['orientation_Y'].tolist(), data['orientation_Z'].tolist(), data['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)



    data['euler_x'] = nx

    data['euler_y'] = ny

    data['euler_z'] = nz



    data['total_angle'] = (data['euler_x'] ** 2 + data['euler_y'] ** 2 + data['euler_z'] ** 2) ** 0.5

    data['angle_vs_acc'] = data['total_angle'] / data['total_linear_acceleration']

    data['angle_vs_vel'] = data['total_angle'] / data['total_angular_velocity']

    df = pd.DataFrame()

    

    for col in data.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_max_to_min'] = df[col + '_max'] / df[col + '_min']



        df[col + '_abs_max'] = data.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))

        

    return df

new_training_data = feature_eng(train_df)

new_test_data = feature_eng(test_df)
new_training_data.fillna(0, inplace = True)

new_test_data.fillna(0, inplace = True)

new_training_data.replace(-np.inf, 0, inplace = True)

new_training_data.replace(np.inf, 0, inplace = True)

new_test_data.replace(-np.inf, 0, inplace = True)

new_test_data.replace(np.inf, 0, inplace = True)
le = LabelEncoder()

target = le.fit_transform(target['surface'])
rf_param_grid = {

                 'max_depth' : list(range(8,100,10)),

                 'n_estimators': list(range(50,2000,100)),

                 'min_samples_split': [2,6,8],

                 'min_samples_leaf': [2,6,8],

                 'bootstrap': [True, False]

                 }

m = RandomForestClassifier(n_jobs=-1)
m_r = GridSearchCV(param_grid=rf_param_grid, estimator = m, scoring = "accuracy", cv = 4)
# %time m_r.fit(new_training_data, target)
m_r = RandomizedSearchCV(param_distributions=rf_param_grid, estimator = m, scoring = "accuracy", cv = 4,n_jobs=-1,n_iter=10)
%time m_r.fit(new_training_data,target)
m_r.best_params_
for param, score in zip(m_r.cv_results_['params'], m_r.cv_results_['mean_test_score']):

    print(param, score)
rf_bp = m_r.best_params_
rf_classifier=RandomForestClassifier(n_estimators=rf_bp["n_estimators"],

                                     min_samples_split=rf_bp['min_samples_split'],

                                     min_samples_leaf=rf_bp['min_samples_leaf'],

                                     max_depth=rf_bp['max_depth'],

                                     bootstrap=rf_bp['bootstrap'])
rf_classifier.fit(new_training_data,target)
y_pred = rf_classifier.predict(new_test_data)
sub['surface'] = le.inverse_transform(y_pred)

sub.to_csv('random_f_rs.csv', index=False)
n_fold = 4

folds = StratifiedKFold(n_splits=n_fold, shuffle=True)
def randomforest_evaluate(**params):

    

    params['n_estimators'] = int(round(params['n_estimators'],0))

    params['min_samples_split'] = int(round(params['min_samples_split'],0))

    params['min_samples_leaf'] = int(round(params['min_samples_leaf'],0))

    params['bootstrap'] = int(round(params['bootstrap'],0))

        

    

    test_pred_proba = np.zeros((new_training_data.shape[0],9))

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(new_training_data, target)):

        X_train, X_valid = new_training_data.iloc[train_idx], new_training_data.iloc[valid_idx]

        y_train, y_valid = target[train_idx], target[valid_idx]

        

        model = RandomForestClassifier(**params,n_jobs=-1)

        

        model.fit(X_train, y_train)

        

        y_pred_valid = model.predict_proba(new_training_data)

       

        test_pred_proba += y_pred_valid / folds.n_splits

        

  

 

    return accuracy_score(target, test_pred_proba.argmax(1))
rf_param_grid = {

                 'max_depth' : (8,100),

                 'n_estimators': (50,2000),

                 'min_samples_split': (2,10),

                 'min_samples_leaf': (2, 10),

                 'bootstrap': (True, False),

                 }



rf_b_o = BayesianOptimization(randomforest_evaluate, rf_param_grid)

%time rf_b_o.maximize(init_points=5, n_iter=20)
for i,n in rf_b_o.max["params"].items():

    print(i,int(round(n)))
model=RandomForestClassifier(n_estimators=int(round(rf_b_o.max["params"]["n_estimators"],0)),max_depth=int(round(rf_b_o.max["params"]["max_depth"])),

                             min_samples_leaf=int(round(rf_b_o.max["params"]["min_samples_leaf"])),min_samples_split=int(round(rf_b_o.max["params"]["min_samples_split"])))



model.fit(new_training_data,target)

y_pred = model.predict(new_test_data)

sub['surface'] = le.inverse_transform(y_pred)

sub.to_csv('random_f_bo.csv', index=False)
def randomforest_evaluate(params):

    print(params)

    params['n_estimators'] = int(round(params['n_estimators'],0))

    params['min_samples_split'] = int(round(params['min_samples_split'],0))

    params['min_samples_leaf'] = int(round(params['min_samples_leaf'],0))

    params['bootstrap'] = int(round(params['bootstrap'],0))

        

    test_pred_proba = np.zeros((new_training_data.shape[0],9))

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(new_training_data, target)):

        X_train, X_valid = new_training_data.iloc[train_idx], new_training_data.iloc[valid_idx]

        y_train, y_valid = target[train_idx], target[valid_idx]

        

        model = RandomForestClassifier(**params,n_jobs=-1)

        

        model.fit(X_train, y_train)

        

        y_pred_valid = model.predict_proba(new_training_data)

        test_pred_proba += y_pred_valid / folds.n_splits

        

  

    return accuracy_score(target, test_pred_proba.argmax(1))



# define a search space





space = {

    

        'max_depth': hp.uniform('max_depth', 8,100),

        'n_estimators': hp.uniform('n_estimators', 50, 2000),

        'min_samples_split' : hp.choice('min_samples_split',[2,3,6,9]),

        'min_samples_leaf' : hp.choice('min_samples_leaf', [2,3,6,9]),

        'bootstrap': hp.choice('bootstrap', [False,True]),

        

    }



# minimize the objective over the space



%time best = fmin(randomforest_evaluate, space, algo=tpe.suggest, max_evals=25)



if best["min_samples_split"] <=1:

    best["min_samples_split"] = 2

    

if best["min_samples_split"] <=1:

    best["min_samples_leaf"] = 2

    

print("Best Params:")

print(best)
model=RandomForestClassifier(n_estimators=int(round(best["n_estimators"],0)),max_depth=int(round(best["max_depth"],0)),

                             min_samples_leaf=int(round(best["min_samples_leaf"])),min_samples_split=int(round(best["min_samples_split"])))



model.fit(new_training_data,target)

y_pred = model.predict(new_test_data)

sub['surface'] = le.inverse_transform(y_pred)

sub.to_csv('random_f_tpe.csv', index=False)
!git clone https://github.com/thuijskens/scikit-hyperband.git 2>/dev/null 1>/dev/null
!cp -r scikit-hyperband/* .
!python setup.py install 2>/dev/null 1>/dev/null
from hyperband import HyperbandSearchCV



from scipy.stats import randint as sp_randint

from sklearn.preprocessing import LabelBinarizer



model = RandomForestClassifier()



param_dist = {

    

    'max_depth': sp_randint(8,100),

    'n_estimators': sp_randint(50,2000),

    'min_samples_split' : [2,3,6,9,10],

    'min_samples_leaf' : [2,3,6,9,10],

    'bootstrap': [True, False]

}



y = LabelBinarizer().fit_transform(target)



search = HyperbandSearchCV(model, param_dist, 

                           resource_param='n_estimators',

                           scoring='accuracy')

%time search.fit(new_training_data, y)

print(search.best_params_)
best=search.best_params_
model=RandomForestClassifier(n_estimators=int(round(best["n_estimators"],0)),max_depth=int(round(best["max_depth"],0)),

                             min_samples_leaf=best["min_samples_leaf"],min_samples_split=best["min_samples_split"])



model.fit(new_training_data,target)

y_pred = model.predict(new_test_data)

sub['surface'] = le.inverse_transform(y_pred)

sub.to_csv('random_f_hb.csv', index=False)

def initilialize_poplulation(numberOfParents):

    max_depth = np.empty([numberOfParents, 1], dtype = np.uint8)

    n_estimators = np.empty([numberOfParents, 1], dtype = np.uint8)

    min_samples_split = np.empty([numberOfParents, 1])

    min_samples_leaf = np.empty([numberOfParents, 1])

    bootstrap = np.empty([numberOfParents, 1])

    

    for i in range(numberOfParents):

        #print(i)

        max_depth[i] = int(random.randrange(8, 100,step=1))

        n_estimators[i] = int(random.randrange(50, 2000, step = 100))

        min_samples_split[i] = int(random.randrange(2, 10, step= 1))

        min_samples_leaf[i] = int(random.randrange(2, 10))

        bootstrap[i] = int(random.randint(0, 1))

       

    

    population = np.concatenate((max_depth, n_estimators, min_samples_split, min_samples_leaf, bootstrap), axis= 1)

    return population
def fitness_accscore(y_true, y_pred):

    fitness = round(accuracy_score(y_true, y_pred), 4)

    return fitness





#train the data annd find fitness score

def train_population(population, train,y_train, test, y_test):

    fScore = []

    for i in range(population.shape[0]):



        param_dist = {



                'max_depth': int(population[i][0]),

                'n_estimators': int(population[i][1]),

                'min_samples_split' : int(population[i][2]),

                'min_samples_leaf' : int(population[i][3]),

                'bootstrap': int(population[i][4])

            }

        #print(param_dist)           

        model = RandomForestClassifier(**param_dist)

        model.fit(train,y_train)

        preds = model.predict(test)

        fScore.append(fitness_accscore(y_test, preds))

    return fScore
#select parents for mating

def new_parents_selection(population, fitness, numParents):

    selectedParents = np.empty((numParents, population.shape[1])) #create an array to store fittest parents

    

    #find the top best performing parents

    for parentId in range(numParents):

        bestFitnessId = np.where(fitness == np.max(fitness))

        bestFitnessId  = bestFitnessId[0][0]

        selectedParents[parentId, :] = population[bestFitnessId, :]

        fitness[bestFitnessId] = -1 #set this value to negative, in case of F1-score, so this parent is not selected again

    return selectedParents
'''

Mate these parents to create children having parameters from these parents (we are using uniform crossover method)

'''

def crossover_uniform(parents, childrenSize):

    

    crossoverPointIndex = np.arange(0, np.uint8(childrenSize[1]), 1, dtype= np.uint8) #get all the index

    crossoverPointIndex1 = np.random.randint(0, np.uint8(childrenSize[1]), np.uint8(childrenSize[1]/2)) # select half  of the indexes randomly

    crossoverPointIndex2 = np.array(list(set(crossoverPointIndex) - set(crossoverPointIndex1))) #select leftover indexes

    

    children = np.empty(childrenSize)

    

    '''

    Create child by choosing parameters from two parents selected using new_parent_selection function. The parameter values

    will be picked from the indexes, which were randomly selected above. 

    '''

    for i in range(childrenSize[0]):

        

        #find parent 1 index 

        parent1_index = i%parents.shape[0]

        #find parent 2 index

        parent2_index = (i+1)%parents.shape[0]

        #insert parameters based on random selected indexes in parent 1

        children[i, crossoverPointIndex1] = parents[parent1_index, crossoverPointIndex1]

        #insert parameters based on random selected indexes in parent 1

        children[i, crossoverPointIndex2] = parents[parent2_index, crossoverPointIndex2]

    return children
def mutation(crossover, numberOfParameters):

    #Define minimum and maximum values allowed for each parameter

    minMaxValue = np.zeros((numberOfParameters, 2))



    minMaxValue[0:] = [8, 100] #min/max max depth

    minMaxValue[1, :] = [50, 2000] #min/max n_estimator

    minMaxValue[2, :] = [2, 10] #min/max min_samples_split

    minMaxValue[3, :] = [2, 10] #min/max min_samples_leaf

    minMaxValue[4, :] = [0, 1] #min/max boostrap



 

    # Mutation changes a single gene in each offspring randomly.

    mutationValue = 0

    parameterSelect = np.random.randint(0,5, 1)

    #print(parameterSelect)

    if parameterSelect == 0: #depth

        mutationValue = np.random.randint(-50, 50, 1)

    if parameterSelect == 1: #n_estimator

        mutationValue = np.random.randint(-200, 200, 1)

    if parameterSelect == 2: #min_samples_split

        mutationValue = np.random.randint(-10, 10, 1)

    if parameterSelect == 3: #min_samples_leaf

        mutationValue = np.random.randint(-10, 10, 1)

    if parameterSelect == 4: #boostrap

        mutationValue = 0



  

    #indtroduce mutation by changing one parameter, and set to max or min if it goes out of range

    for idx in range(crossover.shape[0]):

        crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue

        if(crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):

            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]

        if(crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):

            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]    

    return crossover


X_train, X_test, y_train, y_test = train_test_split(new_training_data, target, test_size = 0.20, random_state = 97)

start = timeit.default_timer()

numberOfParents = 8 #number of parents to start

numberOfParentsMating = 4 #number of parents that will mate

numberOfParameters = 5 #number of parameters that will be optimized

numberOfGenerations = 4 #number of genration that will be created

#define the population size

populationSize = (numberOfParents, numberOfParameters)

#initialize the population with randomly generated parameters

population = initilialize_poplulation(numberOfParents)

#print(population)

#define an array to store the fitness  hitory

fitnessHistory = np.empty([numberOfGenerations+1, numberOfParents])

#define an array to store the value of each parameter for each parent and generation

populationHistory = np.empty([(numberOfGenerations+1)*numberOfParents, numberOfParameters])

#insert the value of initial parameters in history

populationHistory[0:numberOfParents, :] = population

for generation in range(numberOfGenerations):

    print("This is number %s generation" % (generation))

    

    #train the dataset and obtain fitness

    fitnessValue = train_population(population=population, train=X_train,y_train=y_train, test=X_test, y_test=y_test)

    fitnessHistory[generation, :] = fitnessValue

    

    #best score in the current iteration

    print('Best accuracy score in the this iteration = {}'.format(np.max(fitnessHistory[generation, :])))

    #survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be selected

    parents = new_parents_selection(population=population, fitness=fitnessValue, numParents=numberOfParentsMating)

    

    #mate these parents to create children having parameters from these parents (we are using uniform crossover)

    children = crossover_uniform(parents=parents, childrenSize=(populationSize[0] - parents.shape[0], numberOfParameters))

    

    #add mutation to create genetic diversity

    children_mutated = mutation(children, numberOfParameters)

    

    '''

    We will create new population, which will contain parents that where selected previously based on the

    fitness score and rest of them  will be children

    '''

    population[0:parents.shape[0], :] = parents #fittest parents

    population[parents.shape[0]:, :] = children_mutated #children

    

    populationHistory[(generation+1)*numberOfParents : (generation+1)*numberOfParents+ numberOfParents , :] = population #srore parent information





stop = timeit.default_timer()

print('Execution time: ', (stop - start)/60)  
#Best solution from the final iteration

fitness = train_population(population=population, train=X_train,y_train=y_train, test=X_test, y_test=y_test)

fitnessHistory[generation+1, :] = fitness

#index of the best solution

bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]

#Best fitness

print("Best fitness is =", fitness[bestFitnessIndex])

#Best parameters

print("Best parameters are:")

print('Max_depth', int(population[bestFitnessIndex][0]))

print('n_estimators', int(population[bestFitnessIndex][1]))

print('min_samples_split', int(population[bestFitnessIndex][2])) 

print('min_samples_leaf', int(population[bestFitnessIndex][3]))

print('bootstrap', int(population[bestFitnessIndex][4]))

model=RandomForestClassifier(n_estimators=int(population[bestFitnessIndex][1]),max_depth=int(population[bestFitnessIndex][0]),

                             min_samples_leaf=int(population[bestFitnessIndex][3]),min_samples_split=int(population[bestFitnessIndex][2]))



model.fit(new_training_data,target)

y_pred = model.predict(new_test_data)

sub['surface'] = le.inverse_transform(y_pred)

sub.to_csv('random_f_ga.csv', index=False)
