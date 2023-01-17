import pandas as pd

import matplotlib.pylab as plt

from tqdm import tqdm

pd.set_option('max_columns', 100)

colors = plt.cm.datad["tab20"]['listed']

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_T = train.T.drop(['ID_code','target'])

train_T['feat_mean'] = train_T.mean(axis=1)



train_T['feat_max'] = train_T[[x for x in range(0,200000)]].max(axis=1)

train_T['feat_min'] = train_T[[x for x in range(0,200000)]].min(axis=1)

train_T['feat_std'] = train_T[[x for x in range(0,200000)]].std(axis=1)

train_T['feat_var'] = train_T[[x for x in range(0,200000)]].var(axis=1)

train_T['feat_range'] = train_T['feat_max'] - train_T['feat_min']
for row in tqdm(range(0, 500)):

    target_of_this = train.loc[row]['target']

    if target_of_this == 0:

        train_T.sort_values('feat_mean')[row].plot(figsize=(15, 5), color=colors[0], alpha=0.2)

    elif target_of_this == 1:

        train_T.sort_values('feat_mean')[row].plot(figsize=(15, 5), color=colors[2], alpha=0.6)
# Identify the index of postitive and negative observations

train_T.index = [x for x in range(0,200)]

negative_index = train.loc[train['target'] == 0].index.values

positive_index = train.loc[train['target'] == 1].index.values



# plot each one

for plotcount in tqdm(range(0, 5)):

    ax = train_T.sort_values('feat_mean')[negative_index[plotcount]].reset_index(drop=True).plot(figsize=(15, 5),

                                                                                                 color=colors[0],

                                                                                                 alpha=0.5)

    train_T.sort_values('feat_mean')[positive_index[plotcount]].reset_index(drop=True).plot(figsize=(15, 5),

                                                                                            color=colors[2],

                                                                                            alpha=0.5)

    ax.tick_params(which='minor', length=4, color='r')
from bokeh.io import show, output_notebook

output_notebook()

from bokeh.plotting import figure
# plotting only 50 examples of each

p = figure(tools="pan,wheel_zoom,box_zoom,reset,previewsave,hover",

          plot_height=400, plot_width=800)

for plotcount in tqdm(range(0, 50)):

    p.line(train_T.sort_values('feat_mean')[negative_index[plotcount]].reset_index(drop=True).index,

           train_T.sort_values('feat_mean')[negative_index[plotcount]].reset_index(drop=True).values)

    p.line(train_T.sort_values('feat_mean')[positive_index[plotcount]].reset_index(drop=True).index,

           train_T.sort_values('feat_mean')[positive_index[plotcount]].reset_index(drop=True).values,

          color='orange')

show(p)
from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

scaled = ss.fit_transform(train[['var_{}'.format(x) for x in range(0, 200)]])



scaleddf = pd.DataFrame(scaled)

scaleddf_T = scaleddf.T
for plotcount in tqdm(range(0, 5000)):

    scaleddf_T[negative_index[plotcount]].sort_values().reset_index(drop=True).plot(figsize=(15, 15), color=colors[0], alpha=0.2)

    scaleddf_T[positive_index[plotcount]].sort_values().reset_index(drop=True).plot(figsize=(15, 15), color=colors[2], alpha=0.2)