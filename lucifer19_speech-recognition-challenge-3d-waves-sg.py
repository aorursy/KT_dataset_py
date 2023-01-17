SAMPLE_RATE = 128 # [256 - > 16000] Wave Focus Index





import os

from pathlib import Path

import IPython.display as ipd



import os

from os.path import isdir, join

from pathlib import Path



import pandas as pd 

from scipy.fftpack import fft

from scipy import signal

from scipy.io import wavfile

from sklearn.decomposition import PCA



import matplotlib.pyplot as plt

import seaborn as sns

import IPython.display as ipd



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import pandas as pd

from collections import Counter



%matplotlib inline



from subprocess import check_output



folders = os.listdir("../input/train/audio")



def custom_fft(y, fs):

    T = 1.0 / fs

    N = y.shape[0]

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    vals = 2.0/N * np.abs(yf[0:N//2])

    return xf, vals
cols = ['class', 'filename', 'samplerate']

cid = 1

for cid in range(SAMPLE_RATE):

    cols.append("S"+str(cid))



train =  pd.DataFrame( columns = cols)

train[:5]



train_audio_path = '../input/train/audio'



train_labels = os.listdir(train_audio_path)

train_labels.remove('_background_noise_')

print(f'Number of labels: {len(train_labels)}')



labels_to_keep = ['yes', 'no', 'up', 'down', 'left',

                  'right', 'on', 'off', 'stop', 'go', 'silence']



i = 1

train_file_labels = dict()

for label in train_labels:

    files = os.listdir(train_audio_path + '/' + label)

    for f in files:

        train_file_labels[label + '/' + f] = label

        sample_rate, samples = wavfile.read(str(train_audio_path) + '/' + label + '/' + f)

        resampled = signal.resample(samples, int(SAMPLE_RATE/sample_rate * samples.shape[0]))

        if len(resampled) == SAMPLE_RATE:

            arow = [label, f, sample_rate]

            sraw = resampled.tolist()

            rawdata = arow + sraw

            train.loc[len(train)] = rawdata

            print(f + " : " + str(i), end='\r')

            i = i + 1

        else:

            print('Wrong samplerate!    ', end='\r')
train.to_csv("LCFR_SPEECH_FULL_TRAIN_" + str(SAMPLE_RATE) + ".csv")
train['class'] = train['class'].astype('category')

train['y'] = train['class'].cat.codes
train['y'].value_counts()
PROF_train = train



PROF_train = PROF_train.drop('class', 1)

PROF_train = PROF_train.drop('filename', 1)

PROF_train = PROF_train.drop('samplerate', 1)

PROF_train[:10]
data = [go.Bar(

            x = PROF_train["y"].value_counts().index.values,

            y = PROF_train["y"].value_counts().values,

            text='Distribution of target variable'

    )]



layout = go.Layout(

    title='Class distribution'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='basic-bar')
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=2000, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

rf.fit(PROF_train.drop(['y'],axis=1), PROF_train.y)

features = PROF_train.drop(['y'],axis=1).columns.values

print("----- Training Done -----")
# Scatter plot 

trace = go.Scatter(

    y = rf.feature_importances_,

    x = features,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = rf.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = features

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), 

                                                            reverse = False)))

trace2 = go.Bar(

    x=x ,

    y=y,

    marker=dict(

        color=x,

        colorscale = 'Viridis',

        reversescale = True

    ),

    name='Random Forest Feature importance',

    orientation='h',

)



layout = dict(

    title='Speech dataset / Feature importances',

     width = 800, height = 500,

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

#         domain=[0, 0.85],

    ))



fig1 = go.Figure(data=[trace2])

fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, min_samples_leaf=4, max_features=0.2, random_state=0)

gb.fit(PROF_train.drop(['y'],axis=1), PROF_train.y)

features = PROF_train.drop(['y'],axis=1).columns.values

print("----- Training Done -----")
# Scatter plot 

trace = go.Scatter(

    y = gb.feature_importances_,

    x = features,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 10,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = gb.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = features

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Machine Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
# The Wave



colormap = plt.cm.jet

plt.figure(figsize=(25,25))

plt.title('Pearson correlation of All the features', y=1.05, size=15)

sns.heatmap(PROF_train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=False)