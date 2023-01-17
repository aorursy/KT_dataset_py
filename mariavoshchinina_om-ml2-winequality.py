%matplotlib inline



import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import pandas_profiling

import seaborn as sns

import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.shape
df.head(7)
df.info()
df.columns
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

           'pH', 'sulphates', 'alcohol']

target = ['quality']
pandas_profiling.ProfileReport(df)
for i in features:

    plt.figure(i, figsize=(7,7))

    sns.set_palette("Blues")

    ax = sns.boxplot(x="quality", y=i, data=df) 

    plt.title('Boxplot для %s' %i) 
for i in features:

    plt.hist(df[i], 70, alpha=0.75, color ='k')

    plt.xlabel('%s' %i)

    plt.title('Гистограмма %s' %i)

    plt.grid(True)

    plt.show()
log = ['residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide']

for i in log:

    plt.hist(np.log(df[i]), 70, alpha=0.75, color ='k')

    plt.xlabel('%s' %i)

    plt.title('Гистограмма log(%s)' %i)

    plt.grid(True)

    plt.show()
for i in features:

    for j in list(set(df['quality'])):       

        subset = df[df['quality'] == j]

        plt.figure(i, figsize=(15,3))

        sns.set_palette("Blues")

        ax = sns.distplot(subset[i],  hist = False, kde = True,

                  kde_kws = {'linewidth': 3}, 

                  label = j)        

        plt.legend(prop={'size': 16}, title = 'quality')

        plt.title('Гистограмма для %s' %i)

        plt.xlabel('%s' %i)
for i in features:

    #plt.figure(i, figsize=(7,7))

    sns.set_palette("Blues", 1)

    ax = sns.jointplot(y=i, x="quality", data=df.sample(frac=0.5))

    plt.title('Диаграмма рассеяния для %s' %i, loc='center')

    #plt.subplots_adjust(hspace = 3)

    plt.xlabel('%s' %i)
fig = px.parallel_coordinates(df.sample(frac=0.5),color="quality", color_continuous_scale=px.colors.diverging.Tealrose)

fig.show()
df_means = pd.DataFrame(df.groupby('quality', as_index = False).agg('mean'))

df1 = df_means.pop('quality')

df_means['quality'] = df1

df_means.head()
fig = px.parallel_coordinates(df_means,color="quality", color_continuous_scale=px.colors.diverging.Tealrose)

fig.show()
scaler = MinMaxScaler(feature_range=(0,1)) 

df_means_scaled = pd.DataFrame(scaler.fit_transform(df_means[features]), columns = features)

df_means_scaled['quality'] = df_means['quality']
labels = features

stats=df_means_scaled.loc[5,labels].values

stats0=df_means_scaled.loc[0,labels].values



angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

stats = np.concatenate((stats,[stats[0]]))

stats0 = np.concatenate((stats0,[stats0[0]]))

angles = np.concatenate((angles,[angles[0]]))



# Plot stuff

fig = plt.figure(figsize=(7,7))

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2, color = 'r')

ax.fill(angles, stats, alpha=0.2, color = 'r')

ax.plot(angles, stats0, 'o-', linewidth=2, color = 'g')

ax.fill(angles, stats0, alpha=0.2, color = 'g')

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title("Сравнение вин разного качества")

ax.legend(('quality = 8', 'quality = 3'), loc =1)

ax.grid(True)

plt.show()