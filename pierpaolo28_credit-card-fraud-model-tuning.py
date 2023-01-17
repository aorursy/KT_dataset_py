# Source: https://scipy-lectures.org/intro/scipy/auto_examples/plot_optimize_example2.html

import numpy as np 

import matplotlib.pyplot as plt

from scipy import optimize



# Creating a function to examine

x = np.arange(-20, 15, 0.3)

def f(x):

    return x**2 - (5*x)/7 - 50*np.cos(x)



# Global optimization

grid = (-20, 15, 0.3)

xmin_global = optimize.brute(f, (grid, ))

print("Global minima (-20-15) at: {}".format(float(xmin_global)))



# Constrained optimization

xmin_local = optimize.fminbound(f, 5, 15)

print("Local minimum (5-15) at: {}".format(xmin_local))



# Plotting the function

fig = plt.figure(figsize=(10, 8))

plt.plot(x, f(x), 'b', label="f(x)")



# Plotting horizontal line where possible roots can be found 

plt.axhline(0, color='gray', label="Roots Level")



# Plotting the function minima

xmins = np.array([xmin_global[0], xmin_local])

plt.plot(xmins, f(xmins), 'go', label="Minima")

plt.xlabel("x")

plt.ylabel("f(x)")

plt.title("Finding the minimum of a function")

plt.legend(loc='best')

plt.show()
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib.pyplot import figure

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")

df.head()
print(df.shape)

print(df.columns)
percent_missing = df.isnull().sum() * 100 / len(df)

missing_values = pd.DataFrame({'percent_missing': percent_missing})

missing_values.sort_values(by ='percent_missing' , ascending=False)
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')



sns.set(style="ticks")

f = sns.countplot(x="Class", data=df, palette="bwr")

plt.show()
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')



corr=df.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
df2 = df[df.Class == 1][0:400]

print(df2.shape)

df3 = df[df.Class == 0][0:400]

print(df3.shape)



df = df2.append(df3, ignore_index=True)

#df4.head()

df.shape
df.head()
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')



sns.set(style="ticks")

f = sns.countplot(x="Class", data=df, palette="bwr")

plt.show()
X = df.drop(['Class'], axis = 1).values

Y = df['Class']



X = StandardScaler().fit_transform(X)



X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)
model = RandomForestClassifier(n_estimators=300).fit(X_Train,Y_Train)

predictionforest = model.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')



feat_importances = pd.Series(model.feature_importances_, index=df.drop(df[['Class']], 

                                                                       axis=1).columns)

feat_importances.nlargest(30).plot(kind='barh')
df[['V17', 'V9', 'V6', 'V12','Class']].head()
X = df[['V17', 'V9', 'V6', 'V12']]

Y = df['Class']



X = StandardScaler().fit_transform(X)



X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)
model = RandomForestClassifier(random_state= 101).fit(X_Train,Y_Train)

predictionforest = model.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))

acc1 = accuracy_score(Y_Test,predictionforest)
model = RandomForestClassifier(n_estimators=10, random_state= 101).fit(X_Train,Y_Train)

predictionforest = model.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))

acc2 = accuracy_score(Y_Test,predictionforest)
model = RandomForestClassifier(n_estimators= 200, max_features = "log2", min_samples_leaf = 20,

                               random_state= 101).fit(X_Train,Y_Train)

predictionforest = model.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))
from sklearn.model_selection import RandomizedSearchCV



random_search = {

               'max_features': ['auto', 'sqrt','log2', None],

               'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}



print(random_search)
clf = RandomForestClassifier()

model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 80, 

                               cv = 4, verbose= 5, random_state= 101, n_jobs = -1)

model.fit(X_Train,Y_Train)
table = pd.pivot_table(pd.DataFrame(model.cv_results_),

    values='mean_test_score', index='param_n_estimators', columns='param_max_features')

     

sns.heatmap(table)
from sklearn.metrics import make_scorer

from sklearn.metrics import roc_auc_score



def auc_metric(true, pred):

    auc_score = roc_auc_score(pred, true)    

    return auc_score



metric = make_scorer(auc_metric, greater_is_better=True)



clf = RandomForestClassifier()

model = RandomizedSearchCV(estimator = clf, param_distributions = random_search,  

                               n_iter = 80, cv = 4, verbose= 5, random_state= 101, 

                               n_jobs = -1, scoring = metric)



model.fit(X_Train,Y_Train)



n_estimators, max_features = list(np.linspace(151, 1200, 10, dtype = int)), ['auto', 'sqrt','log2', None]



res = model.cv_results_['mean_test_score'].reshape(len(n_estimators), len(max_features))
table = pd.pivot_table(pd.DataFrame(model.cv_results_), 

                       values='mean_test_score', index='param_n_estimators', columns='param_max_features')

     

sns.heatmap(table)

plt.title("Using Area Under the Curve instead of Accuracy as evaluation metric")
def search_plot(grid, param1, param2, name1, name2):



    grid = grid.cv_results_

    scores_mean = grid['mean_test_score']

    scores_mean = np.array(scores_mean).reshape(len(param2),len(param1))



    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

    for idx, val in enumerate(param2):

        plt.plot(param1, scores_mean[idx,:], '-o', label= name2 + ': ' + str(val))



    plt.title("Random Search Accuracy", fontsize=15)

    plt.xlabel(name1, fontsize=12)

    plt.ylabel('Cross-Validation Average Accuracy', fontsize=12)

    plt.legend(loc="best", fontsize=12)

    plt.grid('on')

    

search_plot(model, n_estimators, max_features, 'Estimators Number', 'Max Features')
# From: https://matthewbilyeu.com/blog/2019-02-05/validation-curve-plot-from-gridsearchcv-results

import numbers



def plot_grid_search_validation_curve(grid, param_to_vary, title='Validation Accuracy Curve', ylim=None,

                                      xlim=None, log=None):



    df_cv_results = pd.DataFrame(grid.cv_results_)

    valid_scores_mean = df_cv_results['mean_test_score']

    valid_scores_std = df_cv_results['std_test_score']



    param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']

    param_ranges = [grid.param_distributions[p[6:]] for p in param_cols]

    param_ranges_lengths = [len(pr) for pr in param_ranges]



    valid_scores_mean = np.array(valid_scores_mean).reshape(*param_ranges_lengths)

    valid_scores_std = np.array(valid_scores_std).reshape(*param_ranges_lengths)



    param_to_vary_idx = param_cols.index('param_{}'.format(param_to_vary))



    slices = []

    for idx, param in enumerate(grid.best_params_):

        if (idx == param_to_vary_idx):

            slices.append(slice(None))

            continue

        best_param_val = grid.best_params_[param]

        idx_of_best_param = 0

        if isinstance(param_ranges[idx], np.ndarray):

            idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)

        else:

            idx_of_best_param = param_ranges[idx].index(best_param_val)

        slices.append(idx_of_best_param)



    valid_scores_mean = valid_scores_mean[tuple(slices)]

    valid_scores_std = valid_scores_std[tuple(slices)]



    plt.clf()

    plt.title(title)

    plt.xlabel(param_to_vary)

    plt.ylabel('Score')



    if (ylim is None):

        plt.ylim(0.0, 1.1)

    else:

        plt.ylim(*ylim)

    if (not (xlim is None)):

        plt.xlim(*xlim)

        

    lw = 2

    plot_fn = plt.plot

    if log:

        plot_fn = plt.semilogx



    param_range = param_ranges[param_to_vary_idx]

    if (not isinstance(param_range[0], numbers.Number)):

        param_range = [str(x) for x in param_range]

    plot_fn(param_range, valid_scores_mean, label='Cross-Validation Accuracy',

            color='b', lw=lw)

    plt.fill_between(param_range, valid_scores_mean - valid_scores_std,

                     valid_scores_mean + valid_scores_std, alpha=0.1,

                     color='b', lw=lw)

    plt.legend(loc='lower right')

    plt.show()
plot_grid_search_validation_curve(model, 'n_estimators', log=True, ylim=(.88, 1.02))
random_search = {'criterion': ['entropy', 'gini'],

               'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],

               'max_features': ['auto', 'sqrt','log2', None],

               'min_samples_leaf': [4, 6, 8, 12],

               'min_samples_split': [5, 7, 10, 14],

               'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}



clf = RandomForestClassifier()

model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 80, 

                               cv = 4, verbose= 5, random_state= 101, n_jobs = -1)

model.fit(X_Train,Y_Train)



model.best_params_
table = pd.pivot_table(pd.DataFrame(model.cv_results_),

    values='mean_test_score', index='param_max_depth', columns='param_min_samples_split')

     

sns.heatmap(table)
table = pd.pivot_table(pd.DataFrame(model.cv_results_),

    values='mean_test_score', index='param_max_depth', columns='param_min_samples_leaf')

     

sns.heatmap(table)
table = pd.pivot_table(pd.DataFrame(model.cv_results_),

    values='mean_test_score', index='param_n_estimators', columns='param_criterion')

     

sns.heatmap(table)
predictionforest = model.best_estimator_.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))

acc3 = accuracy_score(Y_Test,predictionforest)
df = pd.DataFrame.from_dict(model.cv_results_)

df['param_min_samples_leaf'] = df['param_min_samples_leaf'].astype(float)

df['param_n_estimators'] = df['param_n_estimators'].astype(float)

df['param_min_samples_split'] = df['param_min_samples_split'].astype(float)

df['mean_fit_time'] = df['mean_fit_time'].astype(float)

df['mean_test_score'] = df['mean_test_score'].astype(float)

df.head()
text = list(

    zip(

        'max_features: ' + df['param_max_features'].apply(str),

        'n_estimators: ' + df['param_n_estimators'].apply(str),

        'min_samples_split: ' + df['param_min_samples_split'].apply(str),

        'Test score: ' + df['mean_test_score'].round(3).apply(str),

        'Training time: ' + (df['mean_fit_time'] % 60).round(3).apply(str) + ' s',

    )

)



hower_text = ['<br>'.join(i) for i in text]

df['hover_text'] = hower_text
import plotly.graph_objs as go

from ipywidgets import interactive, VBox, widgets, interact



trace = go.Scatter3d(

    x= df['param_n_estimators'],

    y= df['param_min_samples_leaf'],

    z= df['param_min_samples_split'],

    mode='markers', 

    marker=dict(

        size= 7,

        color= df['mean_test_score'],

        colorscale= 'Jet',

        colorbar= dict(title = 'Cross-Validation Accuracy')

    ),

    text= df['hover_text'],

    hoverinfo= 'text'

)



data = [trace]

layout = go.Layout(

    scene = dict(

        camera = dict(

            up=dict(x=0, y=0, z=1),

            center=dict(x=0, y=0, z=0),

            eye=dict(x=2, y=2, z=1.25)

        ),

        xaxis = dict(

            title='n_estimators',

            range=[min(df['param_n_estimators']), max(df['param_n_estimators'])]

        ),

        yaxis = dict(

            title='min_leaf',

            type='log'

        ),

        zaxis = dict(

            title='min_split',

            type='log'



        ),

    ),

)



fig = go.FigureWidget(data,layout)



xmin, xmax = fig['layout']['scene']['xaxis']['range']



slider = widgets.FloatRangeSlider(

    value= fig.layout.scene.xaxis.range,

    min= xmin,

    max= xmax,

    step= (xmax - xmin) / 50,

    description= 'n_estimators')

slider.layout.width = '700px'



def update_range(y):

    fig.layout.scene.xaxis.range = [y[0], y[1]]

    

box = VBox((interactive(update_range, y=slider), fig))

box.layout.align_items = 'center'

box
listn = list(set(df['param_n_estimators']))

listn.sort()



data = []

for i, n in enumerate(listn):

    filtered_df = df[df.param_n_estimators==n]

    trace = [

        go.Scatter3d(

    x= filtered_df['param_n_estimators'],

    y= filtered_df['param_min_samples_leaf'],

    z= filtered_df['param_min_samples_split'],

    mode='markers', 

    marker=dict(

        size= 7,

        color= df['mean_test_score'],

        colorscale= 'Jet',

        colorbar= dict(title = 'Cross-Validation Accuracy')

    ),

    text= filtered_df['hover_text'],

    hoverinfo= 'text'

)

    ]

    

    data.append(trace[0])

    data[i].showlegend=False



steps = []

for i, n in enumerate(listn):

    step = dict(

        method='restyle',

        args = ['visible', [False] * len(data) * 2]

    )

    step['args'][1][i] = True # toggle i'th traces to 'visible'

    step['label'] = str(n)

    steps.append(step)







sliders = [dict(

    active = 4,

    currentvalue = {"prefix": "n_estimators: "},

    pad = {"t": 10, 'b': 20},

    steps = steps,

    len=.5,

    xanchor = 'center',

    x = 0.5

)]



layout = go.Layout(

    title = "Cross-Validation Accuracy Varying Hyperparameters",

    width=700,

    height=600,

    sliders = sliders,

    scene = dict(

        camera = dict(

            up=dict(x=0, y=0, z=1),

            center=dict(x=0, y=0, z=0),

            eye=dict(x=2, y=2, z=1.25)

        ),

        xaxis = dict(

            title='n_estimators',

        ),

        yaxis = dict(

            title='min_leaf',

            type='log'

        ),

        zaxis = dict(

            title='min_split',

            type='log'



        ),

    ),

)





fig = go.FigureWidget(data,layout)

fig
data = []

for i, n in enumerate(listn):

    filtered_df = df[df.param_n_estimators==n]

    trace = [

        go.Scatter3d(

    x= filtered_df['param_n_estimators'],

    y= filtered_df['param_min_samples_leaf'],

    z= filtered_df['mean_test_score'],

    mode='markers', 

    marker=dict(

        size= 7,

        color= df['mean_test_score'],

        colorscale= 'Jet',

    ),

    text= filtered_df['hover_text'],

    hoverinfo= 'text'

)

    ]

    

    data.append(trace[0])

    data[i].showlegend=False



steps = []

for i, n in enumerate(listn):

    step = dict(

        method='restyle',

        args = ['visible', [False] * len(data) * 2]

    )

    step['args'][1][i] = True # toggle i'th traces to 'visible'

    step['label'] = str(n)

    steps.append(step)







sliders = [dict(

    active = 4,

    currentvalue = {"prefix": "n_estimators: "},

    pad = {"t": 10, 'b': 20},

    steps = steps,

    len=.5,

    xanchor = 'center',

    x = 0.5

)]



layout = go.Layout(

    title = "Cross-Validation Accuracy Varying n_estimators and min_leaf",

    width=700,

    height=600,

    sliders = sliders,

    scene = dict(

        camera = dict(

            up=dict(x=0, y=0, z=1),

            center=dict(x=0, y=0, z=0),

            eye=dict(x=2, y=2, z=1.25)

        ),

        xaxis = dict(

            title='n_estimators',

        ),

        yaxis = dict(

            title='min_leaf',

            type='log'

        ),

        zaxis = dict(

            title='accuracy',

            type='log'



        ),

    ),

)



fig = go.FigureWidget(data,layout)

fig
from sklearn.model_selection import GridSearchCV



grid_search = {

    'criterion': [model.best_params_['criterion']],

    'max_depth': [model.best_params_['max_depth']],

    'max_features': [model.best_params_['max_features']],

    'min_samples_leaf': [model.best_params_['min_samples_leaf'] - 2, 

                         model.best_params_['min_samples_leaf'], 

                         model.best_params_['min_samples_leaf'] + 2],

    'min_samples_split': [model.best_params_['min_samples_split'] - 3, 

                          model.best_params_['min_samples_split'], 

                          model.best_params_['min_samples_split'] + 3],

    'n_estimators': [model.best_params_['n_estimators'] - 150, model.best_params_['n_estimators'] - 100, 

                     model.best_params_['n_estimators'], 

                     model.best_params_['n_estimators'] + 100, model.best_params_['n_estimators'] + 150]

}



print(grid_search)
clf = RandomForestClassifier()

model = GridSearchCV(estimator = clf, param_grid = grid_search, 

                               cv = 4, verbose= 5, n_jobs = -1)

model.fit(X_Train,Y_Train)
model.best_params_
predictionforest = model.best_estimator_.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))

acc4 = accuracy_score(Y_Test,predictionforest)
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.metrics import accuracy_score
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),

        'max_depth': hp.quniform('max_depth', 10, 1200, 10),

        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),

        'min_samples_leaf': hp.uniform ('min_samples_leaf', 0, 0.5),

        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),

        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200])

    }
def objective(space):

    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],

                                 max_features = space['max_features'],

                                 min_samples_leaf = space['min_samples_leaf'],

                                 min_samples_split = space['min_samples_split'],

                                 n_estimators = space['n_estimators'], 

                                 )

    

    accuracy = cross_val_score(model, X_Train, Y_Train, cv = 4).mean()



    # We aim to maximize accuracy, therefore we return it as a negative value

    return {'loss': -accuracy, 'status': STATUS_OK }
trials = Trials()

best = fmin(fn= objective,

            space= space,

            algo= tpe.suggest,

            max_evals = 80,

            trials= trials)

best
# From: https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce

parameters = ['criterion', 'max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split',

              'n_estimators']

f, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,10))

f.tight_layout()

cmap = plt.cm.jet

for i, val in enumerate(parameters):

    print(i, val)

    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()

    ys = [-t['result']['loss'] for t in trials.trials]

    xs, ys = zip(*sorted(zip(xs, ys)))

    ys = np.array(ys)

    axes[i//2,i%2].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i)/len(parameters)))

    axes[i//2,i%2].set_title(val)
crit = {0: 'entropy', 1: 'gini'}

feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}

est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200}



print(crit[best['criterion']])

print(feat[best['max_features']])

print(est[best['n_estimators']])
trainedforest = RandomForestClassifier(criterion = crit[best['criterion']], max_depth = best['max_depth'], 

                                       max_features = feat[best['max_features']], 

                                       min_samples_leaf = best['min_samples_leaf'], 

                                       min_samples_split = best['min_samples_split'], 

                                       n_estimators = est[best['n_estimators']]).fit(X_Train,Y_Train)

predictionforest = trainedforest.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))

acc5 = accuracy_score(Y_Test,predictionforest)
parameters = {'criterion': ['entropy', 'gini'],

               'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],

               'max_features': ['auto', 'sqrt','log2', None],

               'min_samples_leaf': [4, 12],

               'min_samples_split': [5, 10],

               'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}
from tpot import TPOTClassifier

from deap.gp import Primitive





tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,

                                 verbosity= 2, early_stop= 12,

                                 config_dict={'sklearn.ensemble.RandomForestClassifier': parameters}, 

                                 cv = 4, scoring = 'accuracy')

tpot_classifier.fit(X_Train,Y_Train)
acc6 = tpot_classifier.score(X_Test, Y_Test)

print(acc6)
# https://medium.com/cindicator/genetic-algorithms-and-hyperparameters-weekend-of-a-data-scientist-8f069669015e

args = {}

for arg in tpot_classifier._optimized_pipeline:

    if type(arg) != Primitive:

        try:

            if arg.value.split('__')[1].split('=')[0] in ['criterion', 'max_depth', 

                                                          'max_features', 'min_samples_leaf', 

                                                          'min_samples_split',

                                                          'n_estimators']:

                args[arg.value.split('__')[1].split('=')[0]] = int(arg.value.split('__')[1].split('=')[1])

            else:

                args[arg.value.split('__')[1].split('=')[0]] = float(arg.value.split('__')[1].split('=')[1])

        except:

            pass

params = args
params
model = RandomForestClassifier( max_depth = params['max_depth'],

                             min_samples_leaf = params['min_samples_leaf'],

                             min_samples_split = params['min_samples_split'],

                             n_estimators = params['n_estimators'], 

                             )

model.fit(X_Train,Y_Train)

predictionforest = model.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))

accuracy_score(Y_Test,predictionforest)
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier



def DL_Model(activation= 'linear', neurons= 5, optimizer='Adam'):

    model = Sequential()

    model.add(Dense(neurons, input_dim= 4, activation= activation))

    model.add(Dense(neurons, activation= activation))

    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])

    return model



# Definying grid parameters

activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']

neurons = [5, 10, 15, 25, 35, 50]

optimizer = ['SGD', 'Adam', 'Adamax']

param_grid = dict(activation = activation, neurons = neurons, optimizer = optimizer)



clf = KerasClassifier(build_fn= DL_Model, epochs= 80, batch_size=40, verbose= 0)



model = GridSearchCV(estimator= clf, param_grid=param_grid, n_jobs=-1)

model.fit(X_Train,Y_Train)



print("Max Accuracy Registred: {} using {}".format(round(model.best_score_,3), model.best_params_))

acc = model.cv_results_['mean_test_score']

hyper = model.cv_results_['params']



for mean, param in zip(acc, hyper):

    print("Overall accuracy of {} % using: {}".format(round(mean, 3), param))
predictionforest = model.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))

acc7 = accuracy_score(Y_Test,predictionforest)
print('Base Accuracy vs Manual Search {:0.4f}%.'.format( 100 * (acc2 - acc1) / acc1))

print('Base Accuracy vs Random Search {:0.4f}%.'.format( 100 * (acc3 - acc1) / acc1))

print('Base Accuracy vs Grid Search {:0.4f}%.'.format( 100 * (acc4 - acc1) / acc1))

print('Base Accuracy vs Bayesian Optimization Accuracy {:0.4f}%.'.format( 100 * (acc5 - acc1) / acc1))

print('Base Accuracy vs Evolutionary Algorithms {:0.4f}%.'.format( 100 * (acc6 - acc1) / acc1))

print('Base Accuracy vs Optimized ANN {:0.4f}%.'.format( 100 * (acc7 - acc1) / acc1))