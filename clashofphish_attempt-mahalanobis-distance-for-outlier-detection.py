from pprint import pprint



import pandas as pd

import numpy as np



# Standard plotly imports

import plotly.graph_objs as go

import plotly.figure_factory as pff

from plotly.subplots import make_subplots



import cufflinks

cufflinks.go_offline(connected=True)



import matplotlib.pyplot as plt



from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
data_dir = '../input/tool-wear-detection-in-cnc-mill/'

outcomes = pd.read_csv(data_dir + 'train.csv')
outcomes.info()
outcomes
part_out = outcomes[outcomes['passed_visual_inspection'].notna()]

pprint(pd.crosstab(part_out.tool_condition, part_out.passed_visual_inspection))



passed = part_out.passed_visual_inspection.eq('yes').mul(1)

wear = part_out.tool_condition.eq('unworn').mul(1)



print('\nPearson correlation coefficient: {:.2f}'.format(matthews_corrcoef(passed, wear)))
experiment1 = pd.read_csv(data_dir + 'experiment_01.csv')

experiment1.reset_index(inplace=True)
experiment1.info()
experiment1.head()
experiment1[['Machining_Process', 'index']].groupby('Machining_Process').count()
# Prep data functions

def clean_data(data):

    """Use this function to keep only CNC active cutting actions"""

    keep_act = ['Layer 1 Down', 'Layer 1 Up', 'Layer 2 Down', 'Layer 2 Up', 'Layer 3 Down', 'Layer 3 Up']

    data = data[data['Machining_Process'].isin(keep_act)]

#     print(data[['Machining_Process', 'index']].groupby('Machining_Process').count())

    

    data.drop('Machining_Process', inplace=True, axis=1)

    return data



def scale_decompose(data, pca_n=3):

    # Scale data

    scaler = MinMaxScaler(feature_range=(0, 1))

    scale_exper = scaler.fit_transform(data)



    # Apply PCA to data

    pca = PCA(n_components=pca_n, svd_solver='full')

    return pca.fit_transform(scale_exper)



# Calculate Mahalanobis dist functions

def is_pos_def(A):

    if np.allclose(A, A.T):

        try:

            np.linalg.cholesky(A)

            return True

        except np.linalg.LinAlgError:

            return False

    else:

        return False

    

def cov_matrix(data, verbose=False):

    covariance_matrix = np.cov(data, rowvar=False)

    if is_pos_def(covariance_matrix):

        inv_covariance_matrix = np.linalg.inv(covariance_matrix)

        if is_pos_def(inv_covariance_matrix):

            return covariance_matrix, inv_covariance_matrix

        else:

            print("Error: Inverse of Covariance Matrix is not positive definite!")

    else:

        print("Error: Covariance Matrix is not positive definite!")

        

def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):

    diff = data - mean_distr

    md = []

    for i in range(len(diff)):

        md.append(np.sqrt(diff[i].dot(inv_cov_matrix).dot(diff[i])))

    return md



def MD_detectOutliers(dist, extreme=False, verbose=False):

    k = 3. if extreme else 2.

    threshold = np.mean(dist) * k

    outliers = []

    for i in range(len(dist)):

        if dist[i] >= threshold:

            outliers.append(i)  # index of the outlier

    return np.array(outliers)



def MD_threshold(dist, extreme=False, verbose=False):

    k = 3. if extreme else 2.

    threshold = np.mean(dist) * k

    return threshold



def full_process(experiment_n, components=2, chi2_print=True, exper_num=None):

    """Experiment data should only contain the columns that are desireable"""

    exper_pca = scale_decompose(experiment_n, pca_n=components)



    cov, inv_cov = cov_matrix(exper_pca)

    mean_dist = exper_pca.mean(axis=0)



    m_dist = MahalanobisDist(inv_cov, mean_dist, exper_pca)

    

    if chi2_print:

        fig_x = go.Figure(

            data=[go.Histogram(x=np.square(m_dist))],

            layout=go.Layout({'title': 'X^2 Distribution'})

        )

        fig_x.show()

    

    if exper_num:

        title = 'Mahalanobis Distribution Experiment {}'.format(exper_num)

    else:

        title = 'Mahalanobis Distribution'

    fig_m = pff.create_distplot([m_dist], group_labels=['m_dist'], bin_size=0.15)

    fig_m.update_layout(title_text=title)

    fig_m.show()

    

    return exper_pca, m_dist
def corr_actualcommand(data, corr_cols):

    look_data = data[corr_cols]

    corr_data = look_data.corr()



    fig = go.Figure(data=go.Heatmap(

        z=corr_data.values,

        x=list(corr_data.index),

        y=list(corr_data.columns)

    ))

    fig.show()
# Unworn data

# Dataset loaded above

# Clean up data

exper1_clean = clean_data(experiment1)

clean_ex1 = exper1_clean[(exper1_clean['M1_CURRENT_FEEDRATE']!=50) & (exper1_clean['X1_ActualPosition']!=198)]

print('')

print('Length of data before inaccurate points removed {:d}'.format(len(exper1_clean)))

print('Length of data after inaccurate points removed {:d}'.format(len(clean_ex1)))
columns = exper1_clean.columns

corr_cols = list(filter(lambda x: ('Actual' in x) | ('Command' in x), columns))

corr_actualcommand(exper1_clean, corr_cols)
fig = go.Figure()

fig.add_trace(go.Scatter(

    x=exper1_clean['index'], y=exper1_clean['X1_ActualPosition'],

    mode='lines',

    name='x-actual'

))

fig.add_trace(go.Scatter(

    x=exper1_clean['index'], y=exper1_clean['Y1_ActualPosition'],

    mode='lines',

    name='y-actual'

))



fig.show()
# Worn data

# Load data

experiment6 = pd.read_csv(data_dir + 'experiment_06.csv')

experiment6.reset_index(inplace=True)



# Clean up data

exper6_clean = clean_data(experiment6)

clean_ex6 = exper6_clean[(exper6_clean['M1_CURRENT_FEEDRATE']!=50) & (exper6_clean['X1_ActualPosition']!=198)]

print('')

print('Length of data before inaccurate points removed {:d}'.format(len(exper6_clean)))

print('Length of data after inaccurate points removed {:d}'.format(len(clean_ex6)))
columns = exper6_clean.columns

corr_cols = list(filter(lambda x: ('Actual' in x) | ('Command' in x), columns))

corr_actualcommand(exper6_clean, corr_cols)
fig = make_subplots(rows=3, cols=1)

fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],

        mode='lines',

        name='x-actual'

    ),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['Y1_ActualPosition'],

        mode='lines',

        name='y-actual'

    ),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],

        mode='lines',

        name='x-actual'

    ),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['Z1_ActualPosition'],

        mode='lines',

        name='z-actual'

    ),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],

        mode='lines',

        name='x-actual'

    ),

    row=3, col=1

)

fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['S1_ActualVelocity'],

        mode='lines',

        name='s-actual'

    ),

    row=3, col=1

)



fig.show()
columns = exper6_clean.columns

raw_cols = list(filter(lambda x: ('Current' in x) | ('Voltage' in x) | ('Power' in x), columns))

corr_actualcommand(exper6_clean, raw_cols+['X1_ActualPosition'])
fig = make_subplots(rows=3, cols=1)

fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],

        mode='lines',

        name='x-actual'

    ),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['S1_OutputVoltage'],

        mode='lines',

        name='s-volt'

    ),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],

        mode='lines',

        name='x-actual'

    ),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['S1_OutputPower']*100,

        mode='lines',

        name='s-power'

    ),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['X1_ActualPosition'],

        mode='lines',

        name='x-actual'

    ),

    row=3, col=1

)

fig.add_trace(

    go.Scatter(

        x=exper6_clean['index'], y=exper6_clean['S1_DCBusVoltage']*100,

        mode='lines',

        name='s-bus'

    ),

    row=3, col=1

)



fig.show()
# Collect columns that are desireable in further analysis

keeper_cols = list(filter(lambda x: 'Z1' not in x, raw_cols))



# Define number of PCA components

component = 2 # should use 2 because this fits the basic definition of the Mahalanobis distance
# Perform outlier analysis

exper1_pca, exper1_mdist = full_process(exper1_clean[keeper_cols], components=component)



thresh = MD_threshold(exper1_mdist)
# Perform outlier analysis

exper6_pca, exper6_mdist = full_process(exper6_clean[keeper_cols], components=component)
# Load another dataset - worn

experiment8 = pd.read_csv(data_dir + 'experiment_08.csv')

experiment8.reset_index(inplace=True)



# Clean up data

exper8_clean = clean_data(experiment8)

clean_ex8 = exper8_clean[(exper8_clean['M1_CURRENT_FEEDRATE']!=50) & (exper8_clean['X1_ActualPosition']!=198)]

print('')

print('Length of data before inaccurate points removed {:d}'.format(len(exper8_clean)))

print('Length of data after inaccurate points removed {:d}'.format(len(clean_ex8)))
# Perform outlier analysis

exper8_pca, exper8_mdist = full_process(exper8_clean[keeper_cols], components=component)
# Load another dataset - unworn

experiment3 = pd.read_csv(data_dir + 'experiment_03.csv')

experiment3.reset_index(inplace=True)



# Clean up data

exper3_clean = clean_data(experiment3)

clean_ex3 = exper3_clean[(exper3_clean['M1_CURRENT_FEEDRATE']!=50) & (exper3_clean['X1_ActualPosition']!=198)]

print('')

print('Length of data before inaccurate points removed {:d}'.format(len(exper3_clean)))

print('Length of data after inaccurate points removed {:d}'.format(len(clean_ex3)))
# Perform outlier analysis

exper3_pca, exper3_mdist = full_process(exper3_clean[keeper_cols], components=component)
fig = go.Figure()

fig.add_trace(go.Scatter(

    x=exper1_clean.reset_index().index, y=exper1_mdist,

    mode='lines',

    name='unworn_01'

))

fig.add_trace(go.Scatter(

    x=exper6_clean.reset_index().index, y=exper6_mdist,

    mode='lines',

    name='worn_06'

))

fig.add_trace(go.Scatter(

    x=exper8_clean.reset_index().index, y=exper8_mdist,

    mode='lines',

    name='worn_08'

))

fig.add_trace(go.Scatter(

    x=exper3_clean.reset_index().index, y=exper3_mdist,

    mode='lines',

    name='unworn_03'

))

fig.add_shape(

    type='line',

    y0=thresh,

    y1=thresh,

    x0=0,

    x1=max([len(exper1_mdist), len(exper6_mdist)]),

    line=dict(color='RoyalBlue', width=2, dash='dot')

)

fig.update_shapes(dict(xref='x', yref='y'))

fig.show()
completed_exper = outcomes[outcomes['machining_finalized']=='yes']



unworn = []

idx_unworn = []

worn = []

idx_worn = []

for i, r in completed_exper.iterrows():

    if r['tool_condition'] == 'unworn':

        if r['No'] < 10:

            unw_data = pd.read_csv(data_dir + 'experiment_0{}.csv'.format(r['No']))

        else:

            unw_data = pd.read_csv(data_dir + 'experiment_{}.csv'.format(r['No']))

        unw_data['Experiment'] = r['No']

        unworn.append(unw_data)

        idx_unworn.append(r['No'])

    elif r['tool_condition'] == 'worn':

        if r['No'] < 10:

            w_data = pd.read_csv(data_dir + 'experiment_0{}.csv'.format(r['No']))

        else:

            w_data = pd.read_csv(data_dir + 'experiment_{}.csv'.format(r['No']))

        w_data['Experiment'] = r['No']

        worn.append(w_data)

        idx_worn.append(r['No'])

    

unworn_df = pd.concat(unworn, ignore_index=True)

worn_df = pd.concat(worn, ignore_index=True)
unworn_clean = clean_data(unworn_df)

worn_clean = clean_data(worn_df)



reduce_unworn = unworn_clean[(unworn_clean['M1_CURRENT_FEEDRATE']!=50) & (unworn_clean['X1_ActualPosition']!=198)]

reduce_worn = worn_clean[(worn_clean['M1_CURRENT_FEEDRATE']!=50) & (worn_clean['X1_ActualPosition']!=198)]



print('Unworn data')

for ix in idx_unworn:

    print('% data with noise experiment {}: {:.2f}'.format(

        ix,

        (1

         - len(reduce_unworn[reduce_unworn['Experiment']==ix])

         / len(unworn_clean[unworn_clean['Experiment']==ix]))

    ))



print('Worn data')

for ix in idx_worn:

    print('% data with noise experiment {}: {:.2f}'.format(

        ix,

        (1

         - len(reduce_worn[reduce_worn['Experiment']==ix])

         / len(worn_clean[worn_clean['Experiment']==ix]))

    ))
# Remove bad experiment

unworn_clean = unworn_clean[unworn_clean['Experiment']!=2]



# Perform outlier analysis

unworn_pca, unworn_mdist = full_process(unworn_clean[keeper_cols], components=component)



thresh = MD_threshold(unworn_mdist)

print('Threshold: {:0.2f}'.format(thresh))
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=list(range(0, len(unworn_mdist))),

    y=unworn_mdist,

    mode='lines',

    name='unworn'

))



fig.add_shape(

    type='line',

    y0=thresh,

    y1=thresh,

    x0=0,

    x1=len(unworn_mdist),

    line=dict(color='RoyalBlue', width=2, dash='dot')

)

fig.update_shapes(dict(xref='x', yref='y'))

fig.update_layout(title_text='Mahalanobis Distance Trance All Unworn Data')

fig.show()
unworn_test = pd.DataFrame(unworn_pca)



fig = go.Figure()



fig.add_trace(go.Scatter(

    x=unworn_test[0],

    y=unworn_test[1],

    mode='markers',

    name='unworn'

))



fig.update_layout(title_text='PCA Plot')

fig.show()
# Perform outlier analysis

worn_pca = dict()

worn_mdist = dict()

for ix in idx_worn:

    worn_pca_n, worn_mdist_n = full_process(

        worn_clean[worn_clean['Experiment']==ix][keeper_cols],

        components=component,

        chi2_print=False,

        exper_num=ix

    )

    worn_pca[ix] = worn_pca_n

    worn_mdist[ix] = worn_mdist_n
fig = go.Figure()



x_size = []

for key in worn_mdist:

    x_size.append(len(worn_mdist[key]))

    fig.add_trace(go.Scatter(

        x=list(range(0, len(worn_mdist[key]))),

        y=worn_mdist[key],

        mode='lines',

        name='exper_{}'.format(key)

    ))



fig.add_shape(

    type='line',

    y0=thresh,

    y1=thresh,

    x0=0,

    x1=max(x_size),

    line=dict(color='RoyalBlue', width=2, dash='dot')

)

fig.update_shapes(dict(xref='x', yref='y'))

fig.update_layout(title_text='Mahalanobis Distance Trance of Each Worn Experiment')

fig.show()
fig = go.Figure()



for key in worn_pca:

    worn_test = pd.DataFrame(worn_pca[key])



    fig.add_trace(go.Scatter(

        x=worn_test[0],

        y=worn_test[1],

        mode='markers',

        name='exper_{}'.format(key)

    ))

    

fig.update_layout(title_text='PCA Plot')

fig.show()