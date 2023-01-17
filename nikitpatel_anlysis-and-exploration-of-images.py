# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pydicom

import os
print(os.listdir("../input"))
from os import listdir
from os.path import isfile, join

# Any results you write to the current directory are saved as output.

### plot packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import matplotlib as plt
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline
import cufflinks as cf
cf.go_offline()
train_images_dir = '../input/stage_1_train_images/'
train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]
test_images_dir = '../input/stage_1_test_images/'
test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]
print('5 Training images', train_images[:5]) # Print the first 5
train_label = pd.read_csv("../input/stage_1_train_labels.csv")
class_info = pd.read_csv("../input/stage_1_detailed_class_info.csv")
### plot packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import matplotlib as plt
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline
import cufflinks as cf
cf.go_offline()

trace1 = go.Bar(
            x=class_info['class'].value_counts().index,
            y=class_info['class'].value_counts().values,
        marker=dict(
            color='rgba(222,45,38,0.8)',
        )
    )

data = [trace1]
layout = go.Layout(
    title = 'Class Count of stage_1_detailed'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
trace1  = go.Bar(
            x=train_label['Target'].value_counts().index,
            y=train_label['Target'].value_counts().values,
         marker=dict(
                color='rgba(222,99,38,0.8)',
            )
    )

data = [trace1]
layout = go.Layout(
    title = 'Target Variable'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
# Forked from `https://www.kaggle.com/peterchang77/exploratory-data-analysis`
def parse_data(df):
    
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

parsed = parse_data(train_label)

def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        #rgb = np.floor(np.random.rand(3) * 256).astype('int')
        rgb = [255, 251, 204] # Just use yellow
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=15)

    plt.imshow(im, cmap=plt.cm.gist_gray)
    plt.axis('off')

def overlay_box(im, box, rgb, stroke=2):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
import matplotlib.pylab as plt
plt.style.use('default')
fig=plt.figure(figsize=(25, 25))
columns = 3; rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[train_label['patientId'].unique()[i]])
    fig.add_subplot
opacity = class_info \
    .loc[class_info['class'] == 'Lung Opacity'] \
    .reset_index()
not_normal = class_info \
    .loc[class_info['class'] == 'No Lung Opacity / Not Normal'] \
    .reset_index()
normal = class_info \
    .loc[class_info['class'] == 'Normal'] \
    .reset_index()
plt.style.use('default')
fig=plt.figure(figsize=(25, 25))
columns = 3; rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[opacity['patientId'].unique()[i]])
plt.style.use('default')
fig=plt.figure(figsize=(25, 25))
columns = 3; rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[not_normal['patientId'].loc[i]])
plt.style.use('default')
fig=plt.figure(figsize=(25, 25))
columns = 3; rows = 3
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    draw(parsed[normal['patientId'].loc[i]])
