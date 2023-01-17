# Imports

import pandas as pd

import numpy as np

import math as mt

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



# Classifiers

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import cross_validate

from sklearn.ensemble import RandomForestClassifier
# Util Func's

def get_map(series):

    _map = {}

    for i, col in enumerate(series.unique()):

        _map[col] = i

    return _map



@ignore_warnings(category=ConvergenceWarning)

def simulate(models, xTrain, yTrain, xTest, yTest):

    errors = []

    for model in models:

        results = cross_validate(model, xTrain, yTrain, cv=3)

        scores = results['test_score']

        errors.append((1-(sum(scores)/len(scores)))*100.)

    return errors
# Load Mushroom Data

shroom_data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

# Load NASA Datta

nasa_data = pd.read_csv('../input/nasa-asteroids-classification/nasa.csv')



# Data Preparation

for col in shroom_data.columns:

    shroom_data[col] = shroom_data[col].map(get_map(shroom_data[col]))    

for col in ['Hazardous']:

    nasa_data[col] = nasa_data[col].map(get_map(nasa_data[col]))
# Shroom Data Split

split = .20

test_sz = mt.ceil(len(shroom_data)*split)

train_sz = mt.floor(len(shroom_data)*(1-split))

shroom_train, shroom_test = train_test_split(shroom_data, train_size=train_sz, test_size=test_sz)



# Nasa Data Split

test_sz = mt.ceil(len(nasa_data)*split)

train_sz = mt.floor(len(nasa_data)*(1-split))

nasa_train, nasa_test = train_test_split(nasa_data, train_size=train_sz, test_size=test_sz)
# Feature Selection

nasa_ftrs = ['Absolute Magnitude', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Miles per hour', 'Miss Dist.(miles)', 'Orbit Uncertainity', 'Minimum Orbit Intersection', 'Absolute Magnitude', 'Orbit Uncertainity', 'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',

       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion']

shroom_ftrs = ['cap-shape','cap-color','gill-size','gill-color', 'veil-type', 'veil-color', 'population']



# Feature Extraction

shroom_xTrain = shroom_train.copy()

shroom_xTest  = shroom_test.copy()

nasa_xTrain = nasa_train.copy()

nasa_xTest  = nasa_test.copy()



shroom_xTrain = shroom_xTrain[shroom_ftrs] / shroom_xTrain[shroom_ftrs].max().replace(to_replace=0, method='ffill')

shroom_yTrain = shroom_train['class']

shroom_xTest  = shroom_xTest[shroom_ftrs] / shroom_xTest[shroom_ftrs].max().replace(to_replace=0, method='ffill')

shroom_yTest  = shroom_test['class']



nasa_xTrain = nasa_xTrain[nasa_ftrs] / nasa_xTrain[nasa_ftrs].max()

nasa_yTrain = nasa_train['Hazardous']

nasa_xTest  = nasa_xTest[nasa_ftrs] / nasa_xTest[nasa_ftrs].max()

nasa_yTest  = nasa_test['Hazardous']
nasa_models = [ DecisionTreeClassifier(criterion='entropy'), MLPClassifier(max_iter=100), SVC(), KNeighborsClassifier(), RandomForestClassifier(criterion='entropy') ]

baselineMap = {}



# Nasa Models

baselineMap['NASA'] = simulate(nasa_models, nasa_xTrain, nasa_yTrain, nasa_xTest, nasa_yTest)



nasa_df = pd.DataFrame(baselineMap['NASA'], columns=['NASA'], index=['Decision Tree', 'Neural Network', 'SVC', 'KNN', 'Boosting'])

nasa_baseline_chart = nasa_df.plot.bar(rot=0, title='Model Baseline Error', xlabel='Models', ylabel='Percent Error')
pd.DataFrame(baselineMap['NASA'], columns=['NASA'], index=['Decision Tree', 'Neural Network', 'SVC', 'KNN', 'Boosting'])
dt_experiments = {

    'MaxFeatures' : [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

  , 'MaxLeafNodes' : [ 10, 25, 50, 100, 150, 200, 400, 500, 1000, 1500, 2000]

}



def generate_DT_test_models():

    experiment_models = { 'Criterion' : [ DecisionTreeClassifier(criterion='gini')  ], 'MaxFeatures'  : [ ]

                        , 'Splitter'  : [ DecisionTreeClassifier(splitter='random') ], 'MaxLeafNodes' : [ ] }

    experiment_results = { 'Criterion' : [ ], 'MaxFeatures' : [ ], 'Splitter' : [ ], 'MaxLeafNodes' : [ ] }

    

    for key in dt_experiments.keys():

        for setting in dt_experiments[key]:

            if ('MaxFeatures' == key):

                experiment_models[key].append(DecisionTreeClassifier(max_features=setting))

            elif ('MaxLeafNodes' == key):

                experiment_models[key].append(DecisionTreeClassifier(max_leaf_nodes=setting))

    

    return experiment_models, experiment_results
# Nasa Models

all_models, experiment_results = generate_DT_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], nasa_xTrain, nasa_yTrain, nasa_xTest, nasa_yTest))



# Criterion: Entropy v Gini

data = [ baselineMap['NASA'][0], experiment_results['Criterion'][0][0] ]

df = pd.DataFrame(data, columns=['NASA'], index=['Entropy', 'Gini'])

chart = df.plot.bar(rot=0, title='Entropy v Gini', xlabel='Criterion', ylabel='Percent Error')



# Splitter: Best v Random

data = [ baselineMap['NASA'][0], experiment_results['Splitter'][0][0] ]

df = pd.DataFrame(data, columns=['NASA'], index=['Best', 'Random'])

chart = df.plot.bar(rot=0, title='Best v Random', xlabel='Splitter', ylabel='Percent Error')



# Effects of Max Features

data = experiment_results['MaxFeatures'][0]

df = pd.DataFrame(data, columns=['NASA'], index=dt_experiments['MaxFeatures'])

chart = df.plot(title='Effects of Max Features', xlabel='Max Features', ylabel='Percent Error')



# Effects of Max Leaf Nodes

data = experiment_results['MaxLeafNodes'][0]

df = pd.DataFrame(data, columns=['NASA'], index=dt_experiments['MaxLeafNodes'])

chart = df.plot(title='Effects of Max Leaf Nodes', xlabel='# of Nodes', ylabel='Percent Error')
rf_experiments = {

    'Trees' : [ 1, 2, 5, 10, 20, 30, 40, 50, 75, 100 ]

  , 'MaxFeatures' : [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

  , 'MaxLeafNodes' : [ 10, 25, 50, 100, 150, 200, 400, 500, 1000, 1500, 2000]

}



def generate_RF_test_models():

    experiment_models = { 'Criterion' : [ DecisionTreeClassifier(criterion='gini')  ], 'MaxFeatures'  : [ ]

                        , 'Trees'  : [ ], 'MaxLeafNodes' : [ ] }

    experiment_results = { 'Criterion' : [ ], 'MaxFeatures' : [ ], 'Trees' : [ ], 'MaxLeafNodes' : [ ] }

    

    for key in rf_experiments.keys():

        for setting in rf_experiments[key]:

            if ('Trees' == key):

                experiment_models[key].append(RandomForestClassifier(n_estimators=setting))

            elif ('MaxFeatures' == key):

                experiment_models[key].append(RandomForestClassifier(max_features=setting))

            elif ('MaxLeafNodes' == key):

                experiment_models[key].append(RandomForestClassifier(max_leaf_nodes=setting))

    

    return experiment_models, experiment_results
# Nasa Models

all_models, experiment_results = generate_RF_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], nasa_xTrain, nasa_yTrain, nasa_xTest, nasa_yTest))



# Criterion: Entropy v Gini

data = [ baselineMap['NASA'][0], experiment_results['Criterion'][0][0] ]

df = pd.DataFrame(data, columns=['NASA'], index=['Entropy', 'Gini'])

chart = df.plot.bar(rot=0, title='Entropy v Gini', xlabel='Criterion', ylabel='Percent Error')



# Effects of Max Features

data = experiment_results['MaxFeatures'][0]

df = pd.DataFrame(data, columns=['NASA'], index=rf_experiments['MaxFeatures'])

chart = df.plot(title='Effects of Max Features', xlabel='Max Features', ylabel='Percent Error')



# Effects of Max Leaf Nodes

data = experiment_results['MaxLeafNodes'][0]

df = pd.DataFrame(data, columns=['NASA'], index=rf_experiments['MaxLeafNodes'])

chart = df.plot(title='Effects of Max Leaf Nodes', xlabel='# of Nodes', ylabel='Percent Error')



# Effects of Max Features

data = experiment_results['Trees'][0]

df = pd.DataFrame(data, columns=['NASA'], index=rf_experiments['Trees'])

chart = df.plot(title='Effects of Tree Count', xlabel='# of Trees', ylabel='Percent Error')
nn_experiments = { 

  'LearningRates' : [ 0.00001, 0.0000625, 0.0001, 0.000125, 0.00025, 0.0005, 0.001, 0.002, 0.004 ]

, 'HiddenLayers'  : [ 1, 5, 10, 20, 25, 50, 100, 200 ]

, 'MaxIterations' : [ 1, 5, 10 , 20, 25, 50, 100, 200, 400 ]

, 'Momentum'      : [ 0.001, 0.1, 0.25, 0.45, 0.9, 0.99, 0.999, 0.9999 ]

}



def generate_NN_test_models():

    all_models = { 'LearningRates' : [], 'HiddenLayers' : [], 'MaxIterations' : [], 'Momentum' : [] }

    experiment_results = { 'LearningRates' : [], 'HiddenLayers' : [], 'MaxIterations' : [], 'Momentum' : [] }

    

    for key in nn_experiments.keys():

        for setting in nn_experiments[key]:

            if ('LearningRates' == key):

                    all_models['LearningRates'].append(MLPClassifier(learning_rate_init=setting, max_iter=100))

            elif ('HiddenLayers' == key):

                    all_models['HiddenLayers'].append(MLPClassifier(hidden_layer_sizes=(setting,), max_iter=100))

            elif ('MaxIterations' == key):

                    all_models['MaxIterations'].append(MLPClassifier(max_iter=setting))

            elif ('Momentum' == key):

                    all_models['Momentum'].append(MLPClassifier(solver='sgd', momentum=setting, max_iter=100))

    

    return all_models, experiment_results

                
# NASA Models

all_models, experiment_results = generate_NN_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], nasa_xTrain, nasa_yTrain, nasa_xTest, nasa_yTest))

    

# Effects of Learning Rate

data = experiment_results['LearningRates'][0]

df = pd.DataFrame(data, columns=['NASA'], index=nn_experiments['LearningRates'])

chart = df.plot(title='Effects of Learning Rate', xlabel='Learning Rate', ylabel='Percent Error')



# Effects of Hidden Layer

data = experiment_results['HiddenLayers'][0]

df = pd.DataFrame(data, columns=['NASA'], index=nn_experiments['HiddenLayers'])

chart = df.plot(title='Effects of Hidden Layers', xlabel='# of Hidden Layers', ylabel='Percent Error')



# Effects of Max Iterations

data = experiment_results['MaxIterations'][0]

df = pd.DataFrame(data, columns=['NASA'], index=nn_experiments['MaxIterations'])

chart = df.plot(title='Effects of Max Iterations', xlabel='# of Max Iterations', ylabel='Percent Error')



# Effects of Momentum

data = experiment_results['Momentum'][0]

df = pd.DataFrame(data, columns=['NASA'], index=nn_experiments['Momentum'])

chart = df.plot(title='Effects of Momentum', xlabel='Momentum', ylabel='Percent Error')
knn_experiments = { 

  'VaryingK' : [ 1, 5, 10, 20, 50, 100, 200, 500, 1000 ]

, 'Weights'  : [ 'distance' ]

}



def generate_KNN_test_models():

    all_models = { 'VaryingK' : [], 'Weights' : [ KNeighborsClassifier(weights='distance') ] }

    experiment_results = { 'VaryingK' : [], 'Weights' : [] }

    

    for key in knn_experiments.keys():

        for setting in knn_experiments[key]:

            if ('VaryingK' == key):

                    all_models['VaryingK'].append(KNeighborsClassifier(n_neighbors=setting, weights='distance'))

    

    return all_models, experiment_results
# Nasa Models

all_models, experiment_results = generate_KNN_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], nasa_xTrain, nasa_yTrain, nasa_xTest, nasa_yTest))

    

# Effects of K

data = experiment_results['VaryingK'][0]

df = pd.DataFrame(data, columns=['NASA'], index=knn_experiments['VaryingK'])

chart = df.plot(title='Effects of K', xlabel='K', ylabel='Percent Error')



# Weight: Uniform v Distance

data = [ baselineMap['NASA'][3], experiment_results['Weights'][0][0] ]

df = pd.DataFrame(data, columns=['NASA'], index=['Uniform', 'Distance'])

chart = df.plot.bar(rot=0, title='Uniform v Distance', xlabel='Weight', ylabel='Percent Error')
svc_experiments = { 

  'Kernel'  : [ 'linear', 'poly', 'rbf', 'sigmoid' ]

, 'Degree'  : [ 1, 2, 3, 4, 5, 6, 7, 8 ]

, 'Gamma'   : [ 'auto' ]

, 'MaxIter' : [ 1, 5, 10 , 20, 25, 50, 100, 200, 400 ]

}



def generate_SVC_test_models():

    all_models = { 'Kernel' : [], 'Degree' : [ ], 'Gamma' : [ ], 'MaxIter' : [ ] }

    experiment_results = { 'Kernel' : [], 'Degree' : [ ], 'Gamma' : [ ], 'MaxIter' : [ ] }

    

    for key in svc_experiments.keys():

        for setting in svc_experiments[key]:

            if ('Kernel' == key):

                all_models['Kernel'].append(SVC(kernel=setting))

            elif ('Degree' == key):

                all_models['Degree'].append(SVC(kernel='poly', degree=setting))

            elif ('Gamma' == key):

                all_models['Gamma'].append(SVC(gamma=setting))

            elif ('MaxIter' == key):

                all_models['MaxIter'].append(SVC(max_iter=setting))

    

    return all_models, experiment_results
# Nasa Models

all_models, experiment_results = generate_SVC_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], nasa_xTrain, nasa_yTrain, nasa_xTest, nasa_yTest))

    

# Effects of Degree

data = experiment_results['Degree'][0]

df = pd.DataFrame(data, columns=['NASA'], index=svc_experiments['Degree'])

chart = df.plot(title='Effects of Degree', xlabel='# of Degrees', ylabel='Percent Error')



# Effects of Max Iterations

data = experiment_results['MaxIter'][0]

df = pd.DataFrame(data, columns=['NASA'], index=svc_experiments['MaxIter'])

chart = df.plot(title='Effects of Max Iterations', xlabel='# of Max Iterations', ylabel='Percent Error')



# Effects of Kernels

data = [ baselineMap['NASA'][2] ]

for i in experiment_results['Kernel'][0]:

    data.append(i)

df = pd.DataFrame(data, columns=['NASA'], index=['base', 'linear', 'poly', 'rbf', 'sigmoid'])

chart = df.plot.bar(rot=0, title='Effects of Kernels', xlabel='Kernel Type', ylabel='Percent Error')



# Effects of Gamma

data = [ baselineMap['NASA'][2], experiment_results['Gamma'][0][0] ]

df = pd.DataFrame(data, columns=['NASA'], index=['Scale', 'Auto'])

chart = df.plot.bar(rot=0, title='Scale v Auto', xlabel='Gamma Type', ylabel='Percent Error')
nasa_models = [ DecisionTreeClassifier(criterion='entropy', splitter='best', max_features=0.6), MLPClassifier(learning_rate_init=0.004, hidden_layer_sizes=200, max_iter=400, momentum=0.999), SVC(kernel='poly', degree=6), KNeighborsClassifier(n_neighbors=20, weights='distance'), RandomForestClassifier(n_estimators=10, max_leaf_nodes=200, max_features=0.4, criterion='entropy') ]

improved = simulate(nasa_models, nasa_xTrain, nasa_yTrain, nasa_xTest, nasa_yTest)



imprv_df = pd.DataFrame(improved, columns=['Improved'], index=['Decision Tree', 'Neural Network', 'SVC', 'KNN', 'Boosting']).join(nasa_df)

nasa_baseline_chart = imprv_df.plot.bar(rot=0, title='Model Baseline Error', xlabel='Models', ylabel='Percent Error')
# Shroom Models

shroom_models = [ DecisionTreeClassifier(criterion='entropy'), MLPClassifier(max_iter=100), SVC(), KNeighborsClassifier(), RandomForestClassifier() ]

baselineMap['MUSHROOM'] = simulate(shroom_models, shroom_xTrain, shroom_yTrain, shroom_xTest, shroom_yTest)



shroom_df = pd.DataFrame(baselineMap['MUSHROOM'], columns=['MUSHROOM'], index=['Decision Tree', 'Neural Network', 'SVC', 'KNN', 'Boosting'])

shroom_baseline_chart = shroom_df.plot.bar(rot=0, title='Model Baseline Error', xlabel='Models', ylabel='Percent Error')
pd.DataFrame(baselineMap['MUSHROOM'], columns=['MUSHROOM'], index=['Decision Tree', 'Neural Network', 'SVC', 'KNN', 'Boosting'])
# Shroom Models

all_models, experiment_results = generate_DT_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], shroom_xTrain, shroom_yTrain, shroom_xTest, shroom_yTest))

    

# Criterion: Entropy v Gini

data = [ baselineMap['MUSHROOM'][0], experiment_results['Criterion'][0][0] ]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=['Entropy', 'Gini'])

chart = df.plot.bar(rot=0, title='Entropy v Gini', xlabel='Criterion', ylabel='Percent Error')



# Splitter: Best v Random

data = [ baselineMap['MUSHROOM'][0], experiment_results['Splitter'][0][0] ]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=['Best', 'Random'])

chart = df.plot.bar(rot=0, title='Best v Random', xlabel='Splitter', ylabel='Percent Error')



# Effects of Max Features

data = experiment_results['MaxFeatures'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=dt_experiments['MaxFeatures'])

chart = df.plot(title='Effects of Max Features', xlabel='Ratio of Max Features', ylabel='Percent Error')



# Effects of Max Leaf Nodes

data = experiment_results['MaxLeafNodes'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=dt_experiments['MaxLeafNodes'])

chart = df.plot(title='Effects of Max Leaf Nodes', xlabel='# of Nodes', ylabel='Percent Error')
# Nasa Models

all_models, experiment_results = generate_RF_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], nasa_xTrain, nasa_yTrain, nasa_xTest, nasa_yTest))



# Criterion: Entropy v Gini

data = [ baselineMap['MUSHROOM'][0], experiment_results['Criterion'][0][0] ]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=['Entropy', 'Gini'])

chart = df.plot.bar(rot=0, title='Entropy v Gini', xlabel='Criterion', ylabel='Percent Error')



# Effects of Max Features

data = experiment_results['MaxFeatures'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=rf_experiments['MaxFeatures'])

chart = df.plot(title='Effects of Max Features', xlabel='Max Features', ylabel='Percent Error')



# Effects of Max Leaf Nodes

data = experiment_results['MaxLeafNodes'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=rf_experiments['MaxLeafNodes'])

chart = df.plot(title='Effects of Max Leaf Nodes', xlabel='# of Nodes', ylabel='Percent Error')



# Effects of Max Features

data = experiment_results['Trees'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=rf_experiments['Trees'])

chart = df.plot(title='Effects of Tree Count', xlabel='# of Trees', ylabel='Percent Error')
# Shroom Models

all_models, experiment_results = generate_NN_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], shroom_xTrain, shroom_yTrain, shroom_xTest, shroom_yTest))



# Effects of Learning Rate

data = experiment_results['LearningRates'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=nn_experiments['LearningRates'])

chart = df.plot(title='Effects of Learning Rate', xlabel='Learning Rate', ylabel='Percent Error')



# Effects of Hidden Layer

data = experiment_results['HiddenLayers'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=nn_experiments['HiddenLayers'])

chart = df.plot(title='Effects of Hidden Layers', xlabel='# of Hidden Layers', ylabel='Percent Error')



# Effects of Max Iterations

data = experiment_results['MaxIterations'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=nn_experiments['MaxIterations'])

chart = df.plot(title='Effects of Max Iterations', xlabel='# of Max Iterations', ylabel='Percent Error')



# Effects of Momentum

data = experiment_results['Momentum'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=nn_experiments['Momentum'])

chart = df.plot(title='Effects of Momentum', xlabel='Momentum', ylabel='Percent Error')
# Shroom Models

all_models, experiment_results = generate_KNN_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], shroom_xTrain, shroom_yTrain, shroom_xTest, shroom_yTest))

    

# Effects of K

data = experiment_results['VaryingK'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=knn_experiments['VaryingK'])

chart = df.plot(title='Effects of K', xlabel='K', ylabel='Percent Error')



# Weight: Uniform v Distance

data = [ baselineMap['MUSHROOM'][3], experiment_results['Weights'][0][0] ]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=['Uniform', 'Distance'])

chart = df.plot.bar(rot=0, title='Uniform v Distance', xlabel='Weight', ylabel='Percent Error')
# Shroom Models

all_models, experiment_results = generate_SVC_test_models()

for key in all_models.keys():

    experiment_results[key].append(simulate(all_models[key], shroom_xTrain, shroom_yTrain, shroom_xTest, shroom_yTest))

    

# Effects of Degree

data = experiment_results['Degree'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=svc_experiments['Degree'])

chart = df.plot(title='Effects of Degree', xlabel='# of Degrees', ylabel='Percent Error')



# Effects of Max Iterations

data = experiment_results['MaxIter'][0]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=svc_experiments['MaxIter'])

chart = df.plot(title='Effects of Max Iterations', xlabel='# of Max Iterations', ylabel='Percent Error')



# Effects of Kernels

data = [ baselineMap['MUSHROOM'][2] ]

for i in experiment_results['Kernel'][0]:

    data.append(i)

df = pd.DataFrame(data, columns=['MUSHROOM'], index=['base', 'linear', 'poly', 'rbf', 'sigmoid'])

chart = df.plot.bar(rot=0, title='Effects of Kernels', xlabel='Kernel Type', ylabel='Percent Error')



# Effects of Gamma

data = [ baselineMap['MUSHROOM'][2], experiment_results['Gamma'][0][0] ]

df = pd.DataFrame(data, columns=['MUSHROOM'], index=['Scale', 'Auto'])

chart = df.plot.bar(rot=0, title='Scale v Auto', xlabel='Gamma Type', ylabel='Percent Error')
shroom_models = [ DecisionTreeClassifier(max_features=0.8), MLPClassifier(learning_rate_init=0.004, hidden_layer_sizes=200, max_iter=100, momentum=0.999), SVC(kernel='poly', degree=8), KNeighborsClassifier(n_neighbors=50, weights='distance'), RandomForestClassifier(criterion='gini') ]

improved = simulate(shroom_models, shroom_xTrain, shroom_yTrain, shroom_xTest, shroom_yTest)



df = pd.DataFrame(improved, columns=['Improved'], index=['Decision Tree', 'Neural Network', 'SVC', 'KNN', 'Boosting']).join(shroom_df)

shroom_baseline_chart = df.plot.bar(rot=0, title='Model Baseline v Improved', xlabel='Models', ylabel='Percent Error')