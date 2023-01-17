import pandas as pd

import random as rd

import math as math

import numpy as np



import warnings

warnings.simplefilter('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('../input/neutron-moderator/neutron_moderator.csv')
df1.head()
df1.shape
df1.info()
df1['colisions'] = df1['colisions'].astype('int64')

df1['target'] = df1['target'].astype('int64')

df1['time, 10^(-5) c'] = df1['time, 10^(-5) c'].astype('float64')



df1.info()
df1.describe()
df1['colisions'] = df1['colisions'].astype('int8')

df1['target'] = df1['target'].astype('int8')

df1['time, 10^(-5) c'] = df1['time, 10^(-5) c'].astype('float64')



df1.info()
df1.describe(include=['object'])
df1['target'].value_counts()
df1['target'].value_counts(normalize=True)
columns_to_show = ['time, 10^(-5) c', 'colisions']

df1.groupby(['target'])[columns_to_show].describe(percentiles=[0.25, 0.50, 0.99])
df1.groupby(['target'])[columns_to_show].agg([np.mean, np.std, np.min, np.max])
df1.pivot_table(['colisions', 'time, 10^(-5) c'], ['target'], aggfunc='mean').head(10)
import seaborn as sns

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg'



import matplotlib as mpl

mpl.rcParams["xtick.bottom"] = True



flatui = ["red", "green", "blue"]

sns.palplot(sns.color_palette(flatui))



sns.set(font_scale = 1.5,

       style='white',

       palette=flatui,

       )



sns.set_style("ticks", {"xtick.bottom": True})
sns.boxplot(y="colisions", x="target",

            data=df1)
sns.boxplot(y="time, 10^(-5) c", x="target",

            data=df1)
df1['colisions'] = df1['colisions']/100

df1.rename(columns={'colisions': 'colisions, 10^2'}, inplace=True)
sns.set(font_scale = 1.5,

       style='white',

       )



g = sns.PairGrid(df1[['time, 10^(-5) c','colisions, 10^2', 'Material']],

                 hue='Material',

                 height=5.5,

                 palette= flatui,

                 despine = False,

                 hue_kws={

                     "marker": ["o", "o", "o"]

                 })



g.map_diag(sns.distplot, norm_hist = True, bins = np.linspace(0.0, 2.4, num=(6*4*10), endpoint=True))

g.map_upper(sns.kdeplot, shade=False, shade_lowest=False)

g.map_lower(plt.scatter, s = 35, alpha=0.3, linewidth=1.5, edgecolor="black")

g.fig.set_size_inches(10, 10)





plt.legend(loc='upper center', bbox_to_anchor=(1, 1.25), ncol=3) #vertical legend



# Подписи к осям

g.axes[0,0].yaxis.set_label_text(r'$\rho (t) $')

g.axes[0,0].xaxis.set_label_text('$t, 10^{-5} c$')

g.axes[1,1].yaxis.set_label_text(r'$\rho (colisions) $')

g.axes[1,1].xaxis.set_label_text('$colisions, 10^{2}$')

g.axes[1,0].yaxis.set_label_text('$colisions, 10^{2}$')

g.axes[1,0].xaxis.set_label_text('$t, 10^{-5} c$')

g.axes[0,1].yaxis.set_label_text('$t, 10^{-5} c$')

g.axes[0,1].xaxis.set_label_text('$colisions, 10^{2}$')



# Тики видны на y

g.axes[0,0].yaxis.set_tick_params(labelleft=False)

g.axes[1,1].yaxis.set_tick_params(labelleft=False)

g.axes[1,0].yaxis.set_tick_params(labelleft=True, bottom = True, direction = 'in')

g.axes[0,1].yaxis.set_tick_params(labelleft=True, bottom = True, direction = 'in')



# Тики видны на x

g.axes[0,0].xaxis.set_tick_params(labelleft=True, bottom = True, direction = 'in')

g.axes[1,1].xaxis.set_tick_params(labelleft=True, bottom = True, direction = 'in')

g.axes[1,0].xaxis.set_tick_params(labelleft=True, bottom = True, direction = 'in')

g.axes[0,1].xaxis.set_tick_params(labelleft=True, bottom = True, direction = 'in')



#Лимиты тиков на x

g.axes[1,0].set_xlim(0, 2.5)

g.axes[0,1].set_xlim(0, 0.6)



#Лимиты тиков на y

g.axes[1,0].set_ylim(0.00001, 0.6)

g.axes[0,0].set_ylim(0.00001, 2.5)



plt.subplots_adjust(hspace=0.5, wspace=0.2)
from sklearn import model_selection

from sklearn import metrics

from sklearn import linear_model
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(

    df1[['time, 10^(-5) c', 'colisions, 10^2']], df1[['target']],

    test_size = 0.3, random_state = 0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_text

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)

decision_tree = decision_tree.fit(train_data, train_labels)
r = export_text(decision_tree, feature_names=list(train_data))

print(r)
import matplotlib.style

import matplotlib as mpl

mpl.style.use('seaborn-dark')
import matplotlib.pyplot as plt



from sklearn.tree import plot_tree



mpl.style.use('classic')

mpl.rcParams['figure.facecolor'] = 'white'



plt.figure(figsize=(7.5,7.5), dpi=80)



plot_tree(decision_tree,

          class_names  = True,

          label = 'all',

          filled=True,

          node_ids = True,

          proportion = True,

          feature_names=list(train_data))

plt.show()
model_predictions = decision_tree.predict(test_data)

print(metrics.accuracy_score(test_labels, model_predictions))

print(metrics.classification_report(test_labels, model_predictions))
mpl.style.use('classic')



font = {'family': 'serif',

        'color':  'black',

        'weight': 'normal',

        'size': 12,

        }



mpl.rcParams["xtick.top"] = False

mpl.rcParams["ytick.right"] = False

mpl.rcParams['figure.facecolor'] = 'white'



from matplotlib.colors import ListedColormap

colors = ListedColormap(['red', 'green','blue'])
new_df = test_data

new_df
new_df['target'] = test_labels

new_df
new_df_0 = new_df

new_df_1 = new_df

new_df_2 = new_df
new_df_0 = new_df_0.drop(new_df_0[new_df_0.target != 0].index)

new_df_1 = new_df_1.drop(new_df_1[new_df_1.target != 1].index)

new_df_2 = new_df_2.drop(new_df_2[new_df_2.target != 2].index)

display(new_df.head(5))

display(new_df_0.head(5))

display(new_df_1.head(5))

display(new_df_2.head(5))
new_df_0['target'].value_counts()
new_df_1['target'].value_counts()
new_df_2['target'].value_counts()
new_df['target'].value_counts()
error = [] # 1, if error, 0 true

time = []

colisions = []



np_test_labels = test_labels['target'].to_numpy()

np_test_time = test_data['time, 10^(-5) c'].to_numpy()

np_test_colisions = test_data['colisions, 10^2'].to_numpy()



for i in range(len(np_test_labels)):

    if np_test_labels[i] == model_predictions[i]:

        error.append(0)

    else:

        error.append(1)

        

for i in range(len(error)):

    if error[i] == 1:

        time.append(np_test_time[i])

        colisions.append(np_test_colisions[i])   
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(12,16))



#  Заголовок области Figure:

fig.suptitle('Decision tree',

             y= 0.95,

             fontsize = 20)



plt.subplots_adjust(wspace=0.15, hspace=0.2)



ax_1 = fig.add_subplot(3, 2, 1)

ax_2 = fig.add_subplot(3, 2, 4)

ax_3 = fig.add_subplot(3, 2, 3)

ax_4 = fig.add_subplot(3, 2, 2)



ax_1.set_xlim(0, 3)

ax_1.set_ylim(0, 0.7)

ax_2.set_xlim(0, 3)

ax_2.set_ylim(0, 0.7)

ax_3.set_xlim(0, 3)

ax_3.set_ylim(0, 0.7)

ax_4.set_xlim(0, 3)

ax_4.set_ylim(0, 0.7)



ax_1.set(title = 'Model')

def get_grid():

    x_min, x_max = 0, 4

    y_min, y_max = 0, 1

    return np.meshgrid(np.arange(x_min, x_max, 0.025),

                         np.arange(y_min, y_max, 0.025))

xx, yy = get_grid()

predicted = decision_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax_1.pcolormesh(xx, yy, predicted, cmap=colors)

ax_1.scatter(test_data['time, 10^(-5) c'], test_data['colisions, 10^2'], c=test_labels['target'], s=15,

             cmap=colors, edgecolors='black', linewidth=1.5)

ax_1.set_ylabel('colisions, 10^2')

ax_1.set_xlabel('time, 10^(-5) c')





ax_2.scatter(new_df_0['time, 10^(-5) c'], new_df_0['colisions, 10^2'], c='red',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Water')

ax_2.scatter(new_df_1['time, 10^(-5) c'], new_df_1['colisions, 10^2'], c='green',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Heavy Plexiglass')

ax_2.scatter(new_df_2['time, 10^(-5) c'], new_df_2['colisions, 10^2'], c='blue',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Heavy Water')

ax_2.set_ylabel('colisions, 10^2')

ax_2.set_xlabel('time, 10^(-5) c')

ax_2.set_title('test data')

ax_2.set_xlim(left = 0)

ax_2.legend()



ax_3.scatter(test_data['time, 10^(-5) c'], test_data['colisions, 10^2'], c=model_predictions, cmap=colors, s = 15

            , edgecolors = 'black', linewidths = 1, alpha = 0.5)

ax_3.set_ylabel('colisions, 10^2')

ax_3.set_xlabel('time, 10^(-5) c')

ax_3.set_title('Predictions')

ax_3.set_xlim(left = 0)



ax_4.pcolormesh(xx, yy, predicted, cmap=colors)

ax_4.scatter(time, colisions, c='yellow',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5)

ax_4.set_ylabel('colisions, 10^2')

ax_4.set_xlabel('time, 10^(-5) c')

ax_4.set_title('Errors')

ax_4.set_xlim(left = 0)

ax_4.set_xlim(right = 3)

ax_4.set_ylim(bottom = 0)

ax_4.set_ylim(top = 0.7)

plt.show()



print(metrics.classification_report(test_labels, model_predictions))
display(train_data.head(2))

display(test_data.head(2))

display(train_labels.head(2))

display(test_labels.head(2))
test_data.drop(['target'], axis='columns', inplace=True)

display(train_data.head(2))

display(test_data.head(2))

display(train_labels.head(2))

display(test_labels.head(2))
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=15)

neigh.fit(train_data, train_labels)
model_predictions_knn = neigh.predict(test_data)

print(metrics.accuracy_score(test_labels, model_predictions_knn))

print(metrics.classification_report(test_labels, model_predictions_knn))
error = []

time = []

colisions = []





for i in range(len(np_test_labels)):

    if np_test_labels[i] == model_predictions_knn[i]:

        error.append(0)

    else:

        error.append(1)

        

for i in range(len(error)):

    if error[i] == 1:

        time.append(np_test_time[i])

        colisions.append(np_test_colisions[i]) 
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(12,16))



#  Заголовок области Figure:

fig.suptitle('knn',

             y= 0.95,

             fontsize = 20)



plt.subplots_adjust(wspace=0.15, hspace=0.2)



ax_1 = fig.add_subplot(3, 2, 1)

ax_2 = fig.add_subplot(3, 2, 4)

ax_3 = fig.add_subplot(3, 2, 3)

ax_4 = fig.add_subplot(3, 2, 2)



ax_1.set_xlim(0, 3)

ax_1.set_ylim(0, 0.7)

ax_2.set_xlim(0, 3)

ax_2.set_ylim(0, 0.7)

ax_3.set_xlim(0, 3)

ax_3.set_ylim(0, 0.7)

ax_4.set_xlim(0, 3)

ax_4.set_ylim(0, 0.7)



ax_1.set(title = 'Model')

def get_grid():

    x_min, x_max = 0, 4

    y_min, y_max = 0, 1

    return np.meshgrid(np.arange(x_min, x_max, 0.025),

                         np.arange(y_min, y_max, 0.025))

xx, yy = get_grid()

predicted = neigh.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax_1.pcolormesh(xx, yy, predicted, cmap=colors)

ax_1.scatter(test_data['time, 10^(-5) c'], test_data['colisions, 10^2'], c=test_labels['target'], s=15,

             cmap=colors, edgecolors='black', linewidth=1.5)

ax_1.set_ylabel('colisions, 10^2')

ax_1.set_xlabel('time, 10^(-5) c')





ax_2.scatter(new_df_0['time, 10^(-5) c'], new_df_0['colisions, 10^2'], c='red',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Water')

ax_2.scatter(new_df_1['time, 10^(-5) c'], new_df_1['colisions, 10^2'], c='green',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Heavy Plexiglass')

ax_2.scatter(new_df_2['time, 10^(-5) c'], new_df_2['colisions, 10^2'], c='blue',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Heavy Water')

ax_2.set_ylabel('colisions, 10^2')

ax_2.set_xlabel('time, 10^(-5) c')

ax_2.set_title('Test data')

ax_2.set_xlim(left = 0)

ax_2.legend()



ax_3.scatter(test_data['time, 10^(-5) c'], test_data['colisions, 10^2'], c=model_predictions_knn, cmap=colors, s = 15

            , edgecolors = 'black', linewidths = 1, alpha = 0.5)

ax_3.set_title('Predictions')

ax_3.set_xlim(left = 0)

ax_3.set_ylabel('colisions, 10^2')

ax_3.set_xlabel('time, 10^(-5) c')



ax_4.pcolormesh(xx, yy, predicted, cmap=colors)

ax_4.scatter(time, colisions, c='yellow',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5)

ax_4.set_title('Errors')

ax_4.set_xlim(left = 0)

ax_4.set_xlim(right = 3)

ax_4.set_ylim(bottom = 0)

ax_4.set_ylim(top = 0.7)

ax_4.set_ylabel('colisions, 10^2')

ax_4.set_xlabel('time, 10^(-5) c')



plt.show()



print(metrics.classification_report(test_labels, model_predictions_knn))
model_SGD = linear_model.SGDClassifier(alpha=0.0003, random_state = 1)

model_SGD.fit(train_data, train_labels)

model_predictions_SGD = model_SGD.predict(test_data)

print("accurancy = ", metrics.accuracy_score(test_labels, model_predictions_SGD))

print(metrics.classification_report(test_labels, model_predictions_SGD))
error = []

time = []

colisions = []





for i in range(len(np_test_labels)):

    if np_test_labels[i] == model_predictions_SGD[i]:

        error.append(0)

    else:

        error.append(1)

        

for i in range(len(error)):

    if error[i] == 1:

        time.append(np_test_time[i])

        colisions.append(np_test_colisions[i]) 
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(12,16))



fig.suptitle('SGD',

             y= 0.95,

             fontsize = 20)



plt.subplots_adjust(wspace=0.15, hspace=0.2)



ax_1 = fig.add_subplot(3, 2, 1)

ax_2 = fig.add_subplot(3, 2, 4)

ax_3 = fig.add_subplot(3, 2, 3)

ax_4 = fig.add_subplot(3, 2, 2)



ax_1.set_xlim(0, 3)

ax_1.set_ylim(0, 0.7)

ax_2.set_xlim(0, 3)

ax_2.set_ylim(0, 0.7)

ax_3.set_xlim(0, 3)

ax_3.set_ylim(0, 0.7)

ax_4.set_xlim(0, 3)

ax_4.set_ylim(0, 0.7)



ax_1.set(title = 'Model')

def get_grid():

    x_min, x_max = 0, 4

    y_min, y_max = 0, 1

    return np.meshgrid(np.arange(x_min, x_max, 0.02),

                         np.arange(y_min, y_max, 0.02))

xx, yy = get_grid()

predicted = model_SGD.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax_1.pcolormesh(xx, yy, predicted, cmap=colors)

ax_1.scatter(test_data['time, 10^(-5) c'], test_data['colisions, 10^2'], c=test_labels['target'], s=15,

             cmap=colors, edgecolors='black', linewidth=1.5)

ax_1.set_ylabel('colisions, 10^2')

ax_1.set_xlabel('time, 10^(-5) c')



ax_2.scatter(new_df_0['time, 10^(-5) c'], new_df_0['colisions, 10^2'], c='red',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Water')

ax_2.scatter(new_df_1['time, 10^(-5) c'], new_df_1['colisions, 10^2'], c='green',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Heavy Plexiglass')

ax_2.scatter(new_df_2['time, 10^(-5) c'], new_df_2['colisions, 10^2'], c='blue',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5, label = 'Heavy Water')

ax_2.set_ylabel('colisions, 10^2')

ax_2.set_xlabel('time, 10^(-5) c')

ax_2.set_title('test data')

ax_2.set_xlim(left = 0)

ax_2.legend()



ax_3.scatter(test_data['time, 10^(-5) c'], test_data['colisions, 10^2'], c=model_predictions_SGD, cmap=colors, s = 15

            , edgecolors = 'black', linewidths = 1, alpha = 0.5)

ax_3.set_ylabel('colisions, 10^2')

ax_3.set_xlabel('time, 10^(-5) c')

ax_3.set_title('Predictions')

ax_3.set_xlim(left = 0)



ax_4.pcolormesh(xx, yy, predicted, cmap=colors)

ax_4.scatter(time, colisions, c='yellow',

               s = 15, edgecolors = 'black', linewidths = 1, alpha = 0.5)

ax_4.set_ylabel('colisions, 10^2')

ax_4.set_xlabel('time, 10^(-5) c')

ax_4.set_title('Errors')

ax_4.set_xlim(left = 0)

ax_4.set_xlim(right = 3)

ax_4.set_ylim(bottom = 0)

ax_4.set_ylim(top = 0.7)





plt.show()



print(metrics.classification_report(test_labels, model_predictions_SGD))