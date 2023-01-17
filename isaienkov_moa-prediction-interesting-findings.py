import numpy as np 

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from scipy import stats
SCATTER_SIZE = 600

HIST_WIDTH = 700

HIST_HEIGHT = 500
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_target = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
check = train_target.copy()

check['cp_type'] = train['cp_type']

zeros = check[check['cp_type'] == 'ctl_vehicle']



total_sum = 0

for col in zeros.columns:

    if col in ['sig_id', 'cp_type']:

        continue

    else:

        total_sum += zeros[col].sum()

        

print('Total sum over all columns: ', total_sum)
check = train_target.copy()

check['cp_dose'] = train['cp_dose']

zeros = check[check['cp_dose'] == 'D2']



total_sum = 0

for col in zeros.columns:

    if col in ['atp-sensitive_potassium_channel_antagonist',  'erbb2_inhibitor']:

        total_sum += zeros[col].sum()

        

print('Total sum over all columns: ', total_sum)
check = train_target.copy()

check['cp_time'] = train['cp_time']

zeros = check[check['cp_time'] != 48]



total_sum = 0

for col in zeros.columns:

    if col in ['atp-sensitive_potassium_channel_antagonist',  'erbb2_inhibitor']:

        total_sum += zeros[col].sum()

        

print('Total sum over all columns: ', total_sum)
print('Number of samples in atp-sensitive_potassium_channel_antagonist: ', train_target['atp-sensitive_potassium_channel_antagonist'].sum())

print('Number of samples in erbb2_inhibitor: ', train_target['erbb2_inhibitor'].sum())
check = train[['c-31', 'c-32', 'c-78']]

check['proteasome_inhibitor'] = train_target['proteasome_inhibitor']

check['size'] = 1

check.loc[check['proteasome_inhibitor']==1, 'size'] = 5
fig = px.scatter_3d(

    check, 

    x='c-78', 

    y='c-32',

    z='c-31', 

    color="proteasome_inhibitor", 

    size="size",

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for proteasome_inhibitor'

)



fig.show()
fig = px.scatter(

    check, 

    x='c-32', 

    y='c-31',

    color="proteasome_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for proteasome_inhibitor'

)



fig.show()
fig = px.scatter(

    check, 

    x='c-78', 

    y='c-31',

    color="proteasome_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for proteasome_inhibitor'

)



fig.show()
fig = px.scatter(

    check, 

    x='c-78', 

    y='c-32',

    color="proteasome_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for proteasome_inhibitor'

)



fig.show()
def plot_combined_histograms(plot_list, name):

    fig = make_subplots(rows=2, cols=3)

    traces = [go.Histogram(x=train[col], nbinsx=100, name=col + ' train') for col in plot_list]

    for col in plot_list:

        traces.append(go.Histogram(x=test[col], nbinsx=100, name=col + ' test'))



    for i in range(len(traces)):

        fig.append_trace(traces[i], (i // 3) + 1, (i % 3) + 1)



    fig.update_layout(

        title_text='Mostly correlated features with ' + name,

        height=800,

        width=1000

    )

    fig.show()
plot_combined_histograms(['c-31', 'c-32', 'c-78'], 'proteasome_inhibitor')
def plot_histograms(column, bins=20):

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=train[column], nbinsx=bins, name=column + ' train', histnorm='percent'))

    fig.add_trace(go.Histogram(x=test[column], nbinsx=bins,  name=column + ' test', histnorm='percent'))



    fig.update_layout(

        barmode='overlay',

        height=HIST_HEIGHT,

        width=HIST_WIDTH,

        title_text='Normalized ' + column + ' train & test sets'

    )

    fig.update_traces(opacity=0.6)

    fig.show()
plot_histograms('c-31')
stats.ttest_ind(train['c-31'], test['c-31'])
plot_histograms('c-32')
stats.ttest_ind(train['c-32'], test['c-32'])
plot_histograms('c-78')
stats.ttest_ind(train['c-78'], test['c-78'])
check = train[['g-202', 'g-431', 'g-769']]

check['raf_inhibitor'] = train_target['raf_inhibitor']

check['size'] = 1

check.loc[check['raf_inhibitor']==1, 'size'] = 5
fig = px.scatter_3d(

    check, 

    x='g-202', 

    y='g-431',

    z='g-769', 

    color='raf_inhibitor', 

    size="size",

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for raf_inhibitor'

)



fig.show()
fig = px.scatter(

    check, 

    x='g-202', 

    y='g-431',

    color="raf_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for raf_inhibitor'

)



fig.show()
fig = px.scatter(

    check, 

    x='g-202', 

    y='g-769',

    color="raf_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for raf_inhibitor'

)



fig.show()
fig = px.scatter(

    check, 

    x='g-431', 

    y='g-769',

    color="raf_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for raf_inhibitor'

)



fig.show()
plot_combined_histograms(['g-202', 'g-431', 'g-769'], 'raf_inhibitor')
plot_histograms('g-202')
plot_histograms('g-431')
plot_histograms('g-769')
check = train[['g-235', 'g-635', 'g-745']]

check['egfr_inhibitor'] = train_target['egfr_inhibitor']

check['size'] = 1

check.loc[check['egfr_inhibitor']==1, 'size'] = 5
fig = px.scatter_3d(

    check, 

    x='g-235', 

    y='g-635',

    z='g-745', 

    color='egfr_inhibitor', 

    size="size",

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for egfr_inhibitor'

)

fig.show()
fig = px.scatter(

    check, 

    x='g-235', 

    y='g-635',

    color="egfr_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for egfr_inhibitor'

)

fig.show()
fig = px.scatter(

    check, 

    x='g-235', 

    y='g-745',

    color="egfr_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for egfr_inhibitor'

)

fig.show()
fig = px.scatter(

    check, 

    x='g-745', 

    y='g-635',

    color="egfr_inhibitor", 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for egfr_inhibitor'

)

fig.show()
plot_combined_histograms(['g-235', 'g-635', 'g-745'], 'egfr_inhibitor')
plot_histograms('g-235')
plot_histograms('g-635')
plot_histograms('g-745')
check = train[['g-599', 'g-165', 'g-699']]

check['mtor_inhibitor'] = train_target['mtor_inhibitor']

check['size'] = 1

check.loc[check['mtor_inhibitor']==1, 'size'] = 5
fig = px.scatter_3d(

    check, 

    x='g-599', 

    y='g-165',

    z='g-699', 

    color='mtor_inhibitor', 

    size="size",

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for mtor_inhibitor'

)

fig.show()
plot_combined_histograms(['g-599', 'g-165', 'g-699'], 'mtor_inhibitor')
plot_histograms('g-599')
plot_histograms('g-165')
plot_histograms('g-699')
check = train[['g-392', 'g-361', 'c-48']]

check['tubulin_inhibitor'] = train_target['tubulin_inhibitor']

check['size'] = 1

check.loc[check['tubulin_inhibitor']==1, 'size'] = 5
fig = px.scatter_3d(

    check, 

    x='g-392', 

    y='g-361',

    z='c-48', 

    color='tubulin_inhibitor', 

    size="size",

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for tubulin_inhibitor'

)

fig.show()
plot_combined_histograms(['g-392', 'g-361', 'c-48'], 'tubulin_inhibitor')
plot_histograms('g-392')
plot_histograms('g-361')
plot_histograms('c-48')
check = train[['g-476', 'g-619', 'g-705']]

check['hdac_inhibitor'] = train_target['hdac_inhibitor']

check['size'] = 1

check.loc[check['hdac_inhibitor']==1, 'size'] = 5
fig = px.scatter_3d(

    check, 

    x='g-476', 

    y='g-619',

    z='g-705', 

    color='hdac_inhibitor', 

    size="size",

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for hdac_inhibitor'

)



fig.show()
plot_combined_histograms(['g-476', 'g-619', 'g-705'], 'hdac_inhibitor')
plot_histograms('g-476')
plot_histograms('g-619')
plot_histograms('g-705')
plot_combined_histograms(['g-392', 'g-206', 'g-100'], 'cyclooxygenase_inhibitor')
plot_histograms('g-392')
plot_histograms('g-206')
plot_histograms('g-100')
count = 0

for col in train.columns:

    if col in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']:

        continue

    if stats.ttest_ind(train[col], test[col]).pvalue < 0.05:

        print(col, stats.ttest_ind(train[col], test[col]).pvalue)

        count += 1
print('Number of features with non acepted 0 hypothesis:', count)