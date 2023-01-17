import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib as mpl

import os

from sklearn_pandas import CategoricalImputer

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

df.head()
percent_missing = df.isnull().sum() * 100 / len(df)

missing_values = pd.DataFrame({'column_name': df.columns,

                               'percent_missing': percent_missing})

missing_values
labels = df['Genre']

df['Genre'].value_counts()
imputer = CategoricalImputer()

df['Year'] = imputer.fit_transform(df['Year'].values)

df['Publisher'] = imputer.fit_transform(df['Publisher'].values)

percent_missing = df.isnull().sum() * 100 / len(df)

missing_values = pd.DataFrame({'column_name': df.columns,

                               'percent_missing': percent_missing})

missing_values
df = df.drop(['Rank', 'Year'], axis=1)

df = df.apply(preprocessing.LabelEncoder().fit_transform)

enc_labels = df['Genre']

df = pd.get_dummies(df)

df.head()
labels.unique()
plt.rcParams['legend.fontsize'] = '16'
from pandas.plotting import parallel_coordinates



df2 = df.drop(['Genre'], axis=1)

df2['Genre'] = labels



#df.plot(figsize=(10,10), fontsize=24)

fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(111)

ax.set_title("Parallel Coordinates Example", fontsize=20)

ax.tick_params(axis='x', rotation=30)

ax.tick_params(axis='both', labelsize=20)

parallel_coordinates(df2, class_column='Genre', ax=ax)
X = df.drop(['Genre'], axis=1)

y = df['Genre']

scaler = StandardScaler().fit(X)

X2 = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.33, random_state=42)
from yellowbrick.features import ParallelCoordinates



# Specify the features of interest and the classes of the target

features = list(X.columns)

classes = list(labels.unique())



# Fit the visualizer and display it

plt.rcParams['legend.fontsize'] = '16'

plt.rcParams['axes.titlesize'] = '20'

fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(111)

ax.tick_params(axis='x', rotation=30)

ax.tick_params(axis='both', labelsize=20)



# Instantiate the visualizer

visualizer = ParallelCoordinates(ax=ax,

    classes=classes, features=features,

    normalize='standard', sample=0.05, shuffle=True

)



visualizer.fit_transform(X, labels)

visualizer.show(fontsize=20)
clf = RandomForestClassifier(max_depth=5)

clf.fit(X_train, y_train)

predictionforest = clf.predict(X_test)

print(confusion_matrix(y_test,predictionforest))

print(classification_report(y_test,predictionforest))
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score



random_search = {'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],

               'min_samples_leaf': [4, 6, 8, 12],

               'min_samples_split': [5, 7, 10, 14],

               'max_leaf_nodes': list(np.linspace(10, 120, 10, dtype = int)),

               'n_estimators': list(np.linspace(151, 1200, 10, dtype = int)),

                 'bootstrap': [True, False]

                }



clf = RandomForestClassifier()

model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 7, 

                               cv = 2, verbose= 5, random_state= 101, n_jobs = -1)

model.fit(X_train,y_train)
import seaborn as sns



table = pd.pivot_table(pd.DataFrame(model.cv_results_),

    values='mean_test_score', index='param_n_estimators', columns='param_min_samples_leaf')

     

sns.heatmap(table)
predictionforest = model.best_estimator_.predict(X_test)

print(confusion_matrix(y_test,predictionforest))

print(classification_report(y_test,predictionforest))
df2 = pd.DataFrame(model.cv_results_)

df2.head()
enc_bootstrap = preprocessing.LabelEncoder().fit_transform(df2['param_bootstrap'])

df2 = df2.drop(['params', 'param_bootstrap'], axis=1)

df2['param_bootstrap'] = enc_bootstrap

df2['mean_test_score'] = df2['mean_test_score']*100

df2 = df2[['param_bootstrap', 'param_n_estimators', 'param_min_samples_split', 'param_min_samples_leaf', 'param_max_leaf_nodes',

           'param_max_depth', 'mean_test_score']].astype(int)
df2['mean_test_score']
# From: https://stackoverflow.com/questions/23547347/parallel-coordinates-plot-for-continous-data-in-pandas

def parallel_coordinates2(frame, class_column, cols=None, ax=None, color=None,

                     use_columns=False, xticks=None, colormap=None,

                     **kwds):



    n = len(frame)

    class_col = frame[class_column]

    class_min = np.amin(class_col)

    class_max = np.amax(class_col)



    if cols is None:

        df = frame.drop(class_column, axis=1)

    else:

        df = frame[cols]



    used_legends = set([])



    ncols = len(df.columns)



    # determine values to use for xticks

    if use_columns is True:

        if not np.all(np.isreal(list(df.columns))):

            raise ValueError('Columns must be numeric to be used as xticks')

        x = df.columns

    elif xticks is not None:

        if not np.all(np.isreal(xticks)):

            raise ValueError('xticks specified must be numeric')

        elif len(xticks) != ncols:

            raise ValueError('Length of xticks must match number of columns')

        x = xticks

    else:

        x = range(ncols)



    fig = plt.figure(figsize=(18,7))

    ax = fig.add_subplot(111)



    Colorm = plt.get_cmap(colormap)



    for i in range(n):

        y = df.iloc[i].values

        kls = class_col.iat[i]

        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), 

                alpha=(kls - class_min)/(class_max-class_min), **kwds)



    for i in x:

        ax.axvline(i, linewidth=1, color='black')



    ax.set_xticks(x)

    ax.set_xticklabels(df.columns)

    ax.set_xlim(x[0], x[-1])

    ax.set_title("Parallel Coordinates Example", fontsize=20)

    ax.tick_params(axis='x', rotation=30)

    ax.tick_params(axis='both', labelsize=20)

    ax.grid(False)



    bounds = np.linspace(class_min,class_max,10)

    cax,_ = mpl.colorbar.make_axes(ax)

    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, 

                                   boundaries=bounds, format='%.2f')



    return fig
fig = parallel_coordinates2(df2, class_column='mean_test_score',  colormap="viridis")
!pip install chart-studio
import chart_studio

chart_studio.tools.set_credentials_file(username='TODO', api_key='TODO')
import plotly.express as px

import chart_studio.plotly as py



fig = px.parallel_coordinates(df2, color="mean_test_score", 

                             labels=dict(zip(list(df2.columns), 

                                             list(['_'.join(i.split('_')[1:]) for i in df2.columns]))),

                             color_continuous_scale=px.colors.diverging.Tealrose,

                             color_continuous_midpoint=27)

#py.plot(fig, filename = 'Parallel Coordinates', auto_open=True)

fig.show()
fig = px.parallel_coordinates(df, color="Genre", 

                             labels=list(df.columns),

                             color_continuous_scale=px.colors.diverging.Tealrose,

                             color_continuous_midpoint=27)

fig.show()
!pip install datawrapper
from datawrapper import Datawrapper

dw = Datawrapper(access_token = "TODO")
df3 = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

#df3.head()

plt.barh(df3['Publisher'].value_counts().index[:10], df3['Publisher'].value_counts().values[:10])

plt.title("Most Frequent Publishers")

plt.xlabel("Number of Games");
res = {'Publisher Name': df3['Publisher'].value_counts().index[:10], 'Occurrences': df3['Publisher'].value_counts().values[:10]}

res = pd.DataFrame(data=res)
# games_chart = dw.create_chart(title = "Most Frequent Game Publishers", chart_type = 'd3-bars', data = res)
# dw.update_description(

#     games_chart['id'],

#     source_name = 'Video Game Sales',

#     source_url = 'https://www.kaggle.com/gregorut/videogamesales',

#     byline = 'Pier Paolo Ippolito',

# )
from IPython.display import IFrame



# dw.publish_chart(games_chart['id'])
from IPython.display import HTML



HTML('<iframe title="Most Frequent Game Publishers" aria-label="Bar Chart" id="datawrapper-chart-YEUFF" src="https://datawrapper.dwcdn.net/YEUFF/1/" scrolling="no" frameborder="0" style="width: 0; min-width: 100% !important; border: none;" height="undefined"></iframe><script type="text/javascript">!function(){"use strict";window.addEventListener("message",(function(a){if(void 0!==a.data["datawrapper-height"])for(var e in a.data["datawrapper-height"]){var t=document.getElementById("datawrapper-chart-"+e)||document.querySelector("iframe[src*='"+e+"']");t&&(t.style.height=a.data["datawrapper-height"][e]+"px")}}))}();</script>')
!pip install dtreeviz
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

df.drop(df[(df.Genre == 'Racing') | (df.Genre == 'Shooter') | (df.Genre == 'Role-Playing') ].index, inplace=True)

labels = df['Genre']

imputer = CategoricalImputer()

df['Year'] = imputer.fit_transform(df['Year'].values)

df['Publisher'] = imputer.fit_transform(df['Publisher'].values)

df = df.drop(['Rank', 'Year'], axis=1)

df = df.apply(preprocessing.LabelEncoder().fit_transform)

enc_labels = df['Genre']

df = pd.get_dummies(df)

df.head()
X = df.drop(['Genre'], axis=1)

y = df['Genre']

scaler = StandardScaler().fit(X)

X2 = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier(max_depth=3)

clf.fit(X_train, y_train)

predictionforest = clf.predict(X_test)

print(confusion_matrix(y_test,predictionforest))

print(classification_report(y_test,predictionforest))
from dtreeviz.trees import *



viz = dtreeviz(clf,

               X_train,

               y_train.values,

               target_name='Genre',

               feature_names=list(X.columns),

               class_names=list(labels.unique()),

               histtype='bar', 

               orientation ='TD')

              

viz
#viz.svg()
!pip install ann_visualizer
from keras.models import Sequential

from keras.layers import Dense

from ann_visualizer.visualize import ann_viz



model = Sequential()

model.add(Dense(units=4,activation='relu',

                  input_dim=7))

model.add(Dense(units=4,activation='sigmoid'))

model.add(Dense(units=2,activation='relu'))



ann_viz(model, view=True, filename="example", title="Example ANN")