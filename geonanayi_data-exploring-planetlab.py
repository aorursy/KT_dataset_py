import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

train.head()
len(train)
sam_sub = pd.read_csv('../input/sample_submission.csv')

sam_sub.head()
import cv2

import matplotlib.pyplot as plt 

%matplotlib inline

###by anokas: https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/data-exploration-analysis



new_style = {'grid': False}

plt.rc('axes', **new_style)

_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20, 20))

i = 0

for f, l in train[10:19].values:

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))

    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))

    #ax[i // 4, i % 4].show()

    i += 1

    

plt.show()
labels = train.tags.unique()
labels
len(labels)
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

#by anokas: https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/data-exploration-analysis 

keywords = train['tags'].apply(lambda x: x.split(' '))

from collections import Counter, defaultdict

counts = defaultdict(int)

for k in keywords:

    for l2 in k:

        counts[l2] += 1



data=[go.Bar(x=list(counts.keys()), y=list(counts.values()))]

layout=dict(height=800, width=800, title='Distribution of training labels')

fig=dict(data=data, layout=layout)

py.iplot(data, filename='train-label-dist')