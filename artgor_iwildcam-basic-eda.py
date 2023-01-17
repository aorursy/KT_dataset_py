# libraries

import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

import time 

from PIL import Image

train_on_gpu = True

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import accuracy_score

import cv2



import altair as alt

from  altair.vega import v3

from IPython.display import HTML

import json
!pip install albumentations > /dev/null 2>&1

!pip install pretrainedmodels > /dev/null 2>&1

!pip install kekas > /dev/null 2>&1

!pip install adabound > /dev/null 2>&1
import albumentations

from albumentations import torch as AT

import pretrainedmodels

import adabound



from kekas import Keker, DataOwner, DataKek

from kekas.transformations import Transformer, to_torch, normalize

from kekas.metrics import accuracy

from kekas.modules import Flatten, AdaptiveConcatPool2d

from kekas.callbacks import Callback, Callbacks, DebuggerCallback

from kekas.utils import DotDict
# Preparing altair. I use code from this great kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey



vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>",

)))
os.listdir('../input/')
classes = """empty, 0

deer, 1

moose, 2

squirrel, 3

rodent, 4

small_mammal, 5

elk, 6

pronghorn_antelope, 7

rabbit, 8

bighorn_sheep, 9

fox, 10

coyote, 11

black_bear, 12

raccoon, 13

skunk, 14

wolf, 15

bobcat, 16

cat, 17

dog, 18

opossum, 19

bison, 20

mountain_goat, 21

mountain_lion, 22""".split('\n')

classes = {int(i.split(', ')[1]): i.split(', ')[0] for i in classes}

classes
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

train['classes'] = train['category_id'].apply(lambda x: classes[x])
train.head()
train.classes.unique()
fig = plt.figure(figsize=(25, 60))

imgs = [np.random.choice(train.loc[train['classes'] == i, 'file_name'], 4) for i in train.classes.unique()]

imgs = [i for j in imgs for i in j]

labels = [[i] * 4 for i in train.classes.unique()]

labels = [i for j in labels for i in j]

for idx, img in enumerate(imgs):

    ax = fig.add_subplot(14, 4, idx + 1, xticks=[], yticks=[])

    im = Image.open("../input/train_images/" + img)

    plt.imshow(im)

    ax.set_title(f'Label: {labels[idx]}')
target_count = train['classes'].value_counts().reset_index().rename(columns={'index': 'target'})

render(alt.Chart(target_count).mark_bar().encode(

    y=alt.Y("target:N", axis=alt.Axis(title='Surface'), sort=list(target_count['target'])),

    x=alt.X('classes:Q', axis=alt.Axis(title='Count')),

    tooltip=['target', 'classes']

).properties(title="Counts of target classes", width=400).interactive())
lc = pd.crosstab(train['location'], train['classes']).reset_index()

lc = pd.melt(lc, 'location', lc.columns.tolist()[1:], 'class', 'count')

lc = lc.loc[lc['class'] != 'empty']

render(alt.Chart(lc).mark_circle().encode(

    y='location:N',

    x='class',

    size='count',

    tooltip=['location', 'class', 'count']

).properties(title="Animals and locations", width=400).interactive())