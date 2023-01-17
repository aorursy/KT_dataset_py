import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import plotly.graph_objs as go



import pydicom # for DICOM images

from skimage.transform import resize



# SKLearn

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



import warnings

warnings.filterwarnings("ignore")
list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progressiont/"



train_data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train_data.head(5)
test_data.head(5)
train_data.shape
test_data.shape
train_data.info()
test_data.info()
train_data.isnull().sum().sort_values(ascending=False)
test_data.isnull().sum().sort_values(ascending=False)
t_smoker = train_data['SmokingStatus'].value_counts()
t_smoker
tt_smoker = pd.DataFrame(train_data['SmokingStatus'].value_counts().reset_index().values,

                        columns=['SmokingStatus', 't_smoker'])



tt_smoker = tt_smoker.sort_values('t_smoker', ascending=False)

group_by = tt_smoker.groupby('SmokingStatus')['t_smoker'].sum().reset_index()

fig = px.bar(group_by.sort_values('SmokingStatus', ascending = False)[:20][::-1], x = 'SmokingStatus', y = 't_smoker',

            title = 'Total value counts for smokers and non smokers', text = 't_smoker', height = 500, orientation = 'v' )

fig.show()
t_sex = train_data['Sex'].value_counts()
t_sex
tt_smoker = pd.DataFrame(train_data['Sex'].value_counts().reset_index().values,

                        columns=['Sex', 't_sex'])



tt_smoker = tt_smoker.sort_values('t_sex', ascending=False)

group_by = tt_smoker.groupby('Sex')['t_sex'].sum().reset_index()

fig = px.bar(group_by.sort_values('Sex', ascending = False)[:20][::-1], x = 'Sex', y = 't_sex',

            title = 'Total value counts for male and female', text = 't_sex', height = 500, orientation = 'v', color_discrete_sequence=['darkred'] )

fig.show()
t_percentage = train_data['Percent'].value_counts()
from scipy.stats import norm
ax = sns.distplot(train_data['Percent'],

                  bins=100,

                  kde=True,

                  color='skyblue',

                  hist_kws={"linewidth": 15,'alpha':1})

ax.set(xlabel='Percent', ylabel='Frequency')
xsmoker = train_data[train_data.SmokingStatus=='Ex-smoker']

csmoker = train_data[train_data.SmokingStatus=='Currently smokes']

nsmoker = train_data[train_data.SmokingStatus=='Never smoked']
from plotly.offline import init_notebook_mode,iplot
trace1 = go.Histogram(

    x=xsmoker.Age,

    opacity=0.75,

    name='Ex-Smoker')



trace2 = go.Histogram(

    x=csmoker.Age,

    opacity=0.75,

    name='Currently Smokes')



trace3 = go.Histogram(

    x=nsmoker.Age,

    opacity=0.75,

    name='Never Smoked')



data = [trace1, trace2,trace3]

layout = go.Layout(barmode='stack',

                   title='Age count according to smoking status',

                   xaxis=dict(title='Smoker'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Histogram(

    x=xsmoker.Sex,

    opacity=0.75,

    name='Ex-Smoker')



trace2 = go.Histogram(

    x=csmoker.Sex,

    opacity=0.75,

    name='Currently Smokes')



trace3 = go.Histogram(

    x=nsmoker.Sex,

    opacity=0.75,

    name='Never Smoked')



data = [trace1, trace2, trace3]

layout = go.Layout(barmode='stack',

                   title='Smokers Counts According to Sex',

                   xaxis=dict(title='Sex'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
IMAGE_PATH


import gc

import cv2



import pydicom # for DICOM images

from skimage.transform import resize
filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"

data = pydicom.dcmread(filename)

plt.imshow(data.pixel_array, cmap=plt.cm.bone) 
plt.imshow(data.pixel_array) 
from sklearn.model_selection import train_test_split
desired_factor = ['SmokingStatus', 'Sex', 'Age',  'Percent',  'FVC',  'Weeks']
train_d = train_data[desired_factor]

test_d = test_data[desired_factor]

y = train_data.FVC
train_d['Sex'] = train_d['Sex'].map({'Male':1, 'Female':0})

train_d['SmokingStatus'] = train_d['SmokingStatus'].map({'Ex-smoker':0, 'Currently smokes':1, 'Never smoked':2})
test_d['Sex'] = test_d['Sex'].map({'Male':1, 'Female':0})

test_d['SmokingStatus'] = test_d['SmokingStatus'].map({'Ex-smoker':0, 'Currently smokes':1, 'Never smoked':2})
train_d
train_d.shape
test_d.shape
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

regressor.fit(train_d, y)
predictions = regressor.predict(train_d)
predictions
y_pred = regressor.predict(test_d)
y_pred
test_d['FVC'].mean()
y_pred.mean()
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

print(submission.shape)

submission.head()