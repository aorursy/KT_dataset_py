# !conda install -c plotly plotly-orca

import os

os.system('pip install --target=/kaggle/working pymap3d==2.1.0')

os.system('pip install --target=/kaggle/working protobuf==3.12.2')

os.system('pip install --target=/kaggle/working transforms3d')

os.system('pip install --target=/kaggle/working zarr')

os.system('pip install --target=/kaggle/working ptable')



os.system('pip install --no-dependencies --target=/kaggle/working l5kit')

!pip install -U kaleido
%matplotlib inline

import pandas as pd

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

import datetime

import time

import numpy as np

import plotly.graph_objects as go

import plotly.express as px

from tqdm import tqdm



from l5kit.data import ChunkedDataset
DATA_PATH = '/kaggle/input/lyft-motion-prediction-autonomous-vehicles'
train_dt = ChunkedDataset(DATA_PATH+'/scenes/train.zarr').open(cached=False)

valid_dt = ChunkedDataset(DATA_PATH+'/scenes/validate.zarr').open(cached=False)

test_dt = ChunkedDataset(DATA_PATH+'/scenes/test.zarr').open(cached=False)





pd.DataFrame([

    ['train', len(train_dt.frames), len(train_dt.agents), len(train_dt.scenes), len(train_dt.tl_faces)],

    ['valid', len(valid_dt.frames), len(valid_dt.agents), len(valid_dt.scenes), len(valid_dt.tl_faces)],

    ['test', len(test_dt.frames), len(test_dt.agents), len(test_dt.scenes), len(test_dt.tl_faces)],

], columns=['set', 'frames', 'agents', 'scenes', 'tl_faces'])
def ts_to_dt(ts):

    return datetime.datetime.fromtimestamp(ts // 10**9)
train_scenes = pd.DataFrame(

    [[f['host'], f['start_time'], f['end_time']] for f in train_dt.scenes],

    columns=['host', 'start', 'end']

)

valid_scenes = pd.DataFrame(

    [[f['host'], f['start_time'], f['end_time']] for f in valid_dt.scenes],

    columns=['host', 'start', 'end']

)

test_scenes = pd.DataFrame(

    [[f['host'], f['start_time'], f['end_time']] for f in test_dt.scenes],

    columns=['host', 'start', 'end']

)

train_scenes['set'] = 'train'

test_scenes['set'] = 'test'

valid_scenes['set'] = 'valid'

scenes = pd.concat([train_scenes, test_scenes, valid_scenes])
scenes['start_time'] = scenes.start.apply(ts_to_dt)

scenes['end_time'] = scenes.end.apply(ts_to_dt)

scenes['duration'] = (scenes.end - scenes.start) / 10**9



'duration', scenes.duration.unique()

pd.concat([

    scenes.groupby(['set', 'host']).start_time.min(),

    scenes.groupby(['set', 'host']).end_time.max()

], axis=1)
train_times = pd.DataFrame({'t': [ts_to_dt(f['timestamp']) for f in train_dt.frames]})

test_times = pd.DataFrame({'t': [ts_to_dt(f['timestamp']) for f in test_dt.frames]})

valid_times = pd.DataFrame({'t': [ts_to_dt(f['timestamp']) for f in valid_dt.frames]})





train_times['set'] = 'train'

test_times['set'] = 'test'

valid_times['set'] = 'valid'

times = pd.concat([train_times, test_times, valid_times])



times['cnt'] = 1

times['h'] = times.t.dt.round("H")

times['day'] = times.t.dt.round("D")





times



df = times.groupby(['set', 'day']).sum().reset_index()
fig = px.bar(df, x='day', y='cnt', color='set', title='Train-Valid-Test split')

fig.write_image('train-test-split.png')

fig.show()
train_locs = pd.DataFrame(

    [f['ego_translation'] for f in train_dt.frames], columns=['x', 'y', 'z'])

test_locs = pd.DataFrame(

    [f['ego_translation'] for f in test_dt.frames], columns=['x', 'y', 'z'])

valid_locs = pd.DataFrame(

    [f['ego_translation'] for f in valid_dt.frames], columns=['x', 'y', 'z'])



train_locs = train_locs.round()



train_locs['set'] = 'train'

test_locs['set'] = 'test'

valid_locs['set'] = 'valid'

locs = pd.concat([train_locs, test_locs, valid_locs])

locs = locs.round()

locs['cnt'] = 1



df = locs.groupby(['set', 'x', 'y']).sum().reset_index()
f1 = px.scatter(df[df.set == 'train'], x='x', y='y', size='cnt', title='Train - Locations', opacity=0.5)

f1.update_traces(marker=dict(color='red', line_width=0))

f1.write_image('train-locations.png')

f2 = px.scatter(df[df.set == 'test'], x='x', y='y', size='cnt', title='Test - Locations', opacity=0.5)

f2.update_traces(marker=dict(color='blue', line_width=0))

f2.write_image('test-locations.png')

f3 = px.scatter(df[df.set == 'valid'], x='x', y='y', size='cnt', title='Valid - Locations', opacity=0.5)

f3.update_traces(marker=dict(color='green', line_width=0))