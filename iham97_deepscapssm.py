# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import statsmodels.api as sm
from sklearn import linear_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings

%matplotlib inline

# figure size in inches
rcParams['figure.figsize'] = 12,6
# Any results you write to the current directory are saved as output.

df_features = pd.read_csv("../input/scapFeaturesData.csv", encoding='ISO-8859-1' )
#We delete the MeshID feature from our dataset
del df_features['MeshID']
df_features.head(n=3).transpose()

trace1 = go.Histogram(
    x=np.log(df_features['CSA']).sample(800), histnorm='percent', autobinx=True,
    showlegend=True, name='CSA')
    
trace2 = go.Histogram(
    x=np.log(df_features['Version']).sample(800), histnorm='percent', autobinx=True,
    showlegend=True, name='Version')

trace3 = go.Histogram(
    x=np.log(df_features['Tilt']).sample(800), histnorm='percent', autobinx=True,
    showlegend=True, name='Tilt')
    
trace4 = go.Histogram(
    x=np.log(df_features['Glene Width']).sample(800), histnorm='percent', autobinx=True,
    showlegend=True, name='Glene Width')
    
trace5 = go.Histogram(
    x=np.log(df_features['Glene Length']).sample(800), histnorm='percent', autobinx=True,
    showlegend=True, name='Glene Length')

#Creating the grid
fig = tls.make_subplots(rows=2, cols=3, specs=[[{'colspan': 2}, None, {}], [{}, {}, {}]],
                          subplot_titles=("CSA",
                                          "Version", 
                                          "Tilt",
                                          "Glene Width", 
                                          "Glene Length"))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 3)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 2, 3)

fig['layout'].update(showlegend=True, title="Features Distribution")
iplot(fig)

print('The standard deviation and the mean for the CSA are %(stdCS)f and %(meanCS)f .' %{'stdCS':df_features["CSA"].std() , "meanCS": df_features["CSA"].mean()})
print('The standard deviation and the mean for the version are %(stdV)f and %(meanV)f .' %{'stdV':df_features["Version"].std() , "meanV": df_features["Version"].mean()})
print('The standard deviation and the mean for the tilt are %(stdT)f and %(meanT)f .' %{'stdT':df_features["Tilt"].std() , "meanT": df_features["Tilt"].mean()})
print('The standard deviation and the mean for the glene width are %(stdW)f and %(meanW)f .' %{'stdW':df_features["Glene Width"].std() , "meanW": df_features["Glene Width"].mean()})
print('The standard deviation and the mean for the glene length are %(stdL)f and %(meanL)f .' %{'stdL':df_features["Glene Length"].std() , "meanL": df_features["Glene Length"].mean()})
# We define the targets
target = pd.DataFrame(df_features, columns=["CSA","Version","Tilt","Glene Width","Glene Length"])

# We define the predictors
df = pd.DataFrame(df_features, columns=["First PC","Second PC","Third PC","Fourth PC","Fifth PC","Sixth PC","Seventh PC","Ninth PC","Tenth PC"])
X = df
y = target

# I now fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)
print(predictions[0:5].transpose())
print(lm.score(X,y))
df_features.head(n=5).transpose().head(n=5).transpose()
# We do the necessary imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, MaxPooling1D
from keras import optimizers
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.applications import vgg16
model = Sequential()
# Our input will be a 10 size vector containing the coefficients for each eigenvector
model.add(Dense(100, input_dim=9))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Dense(100))
#model.add(MaxPooling1D(pool_size=3))
model.add(Dense(5))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01,decay=0.1), metrics=['accuracy'])
model.summary()
#sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
hist = model.fit(X, y, epochs=1000, verbose=1, validation_split=0.2)
y_pred = model.predict(X) 

print(y_pred[0:5].transpose())
scores = model.evaluate(X, y, verbose=1)
print('%(score)f percent accuracy.'%{'score':scores[1]*100})
df_features.head(n=5).transpose().head(n=5).transpose()
