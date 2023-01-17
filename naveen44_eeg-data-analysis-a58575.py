import numpy as np

import pandas as pd 

import os

import random

from tqdm import tqdm

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from plotly import tools

from scipy.stats import mannwhitneyu



init_notebook_mode(connected=True) ## plotly init

seed = 123

random.seed = seed
print('Total amount of files in SMNI_CMI_TRAIN directory: ' + str(len(os.listdir('../input/SMNI_CMI_TRAIN/Train'))))
filenames_list = os.listdir('../input/SMNI_CMI_TRAIN/Train/') ## list of file names in the directory

EEG_data = pd.DataFrame({}) ## create an empty df that will hold data from each file



for file_name in tqdm(filenames_list):

    temp_df = pd.read_csv('../input/SMNI_CMI_TRAIN/Train/' + file_name,engine = 'python') ## read from the file to df

    EEG_data = EEG_data.append(temp_df) ## add the file data to the main df

    

EEG_data = EEG_data.drop(['Unnamed: 0'], axis=1) ## remove the unused column

EEG_data.loc[EEG_data['matching condition'] == 'S2 nomatch,', 'matching condition'] =  'S2 nomatch' ## remove comma sign from stimulus name
## here is how the data looks like

EEG_data.head()
## replace some 'sensor position' values

EEG_data.loc[EEG_data['sensor position'] == 'AF1', 'sensor position'] = 'AF3'

EEG_data.loc[EEG_data['sensor position'] == 'AF2', 'sensor position'] = 'AF4'

EEG_data.loc[EEG_data['sensor position'] == 'PO1', 'sensor position'] = 'PO3'

EEG_data.loc[EEG_data['sensor position'] == 'PO2', 'sensor position'] = 'PO4'

## remove rows with undefined positions

EEG_data = EEG_data[(EEG_data['sensor position'] != 'X') & (EEG_data['sensor position'] != 'Y') & (EEG_data['sensor position'] != 'nd')]
def sample_data(stimulus, random_id=random.randint(0,7)):

    """Function merged data frame - one data frame for randomly selected subject from control group and 

    one data frame for randomly selected subject from alcoholic group"""

    ## random choose the name_id of subject from alcoholic/control group

    alcoholic_id = EEG_data['name'][(EEG_data['subject identifier'] == 'a') & 

                                    (EEG_data['matching condition'] == stimulus)].unique()[random_id]

    control_id = EEG_data['name'][(EEG_data['subject identifier'] == 'c') & 

                                  (EEG_data['matching condition'] == stimulus)].unique()[random_id]

    

    ## get min trial numbers for each group

    alcoholic_trial_number = EEG_data['trial number'][(EEG_data['name'] == alcoholic_id) & (EEG_data['matching condition'] == stimulus)].min()

    control_trial_number = EEG_data['trial number'][(EEG_data['name'] == control_id) & (EEG_data['matching condition'] == stimulus)].min()



    ## filter the EEG DF

    alcoholic_df = EEG_data[(EEG_data['name'] == alcoholic_id) & (EEG_data['trial number'] == alcoholic_trial_number)]

    control_df = EEG_data[(EEG_data['name'] == control_id) & (EEG_data['trial number'] == control_trial_number)]

    

    return alcoholic_df.append(control_df)
sensor_positions = EEG_data[['sensor position', 'channel']].drop_duplicates().reset_index(drop=True).drop(['channel'], axis=1).reset_index(drop=False).rename(columns={'index':'channel'})['sensor position']

channels = EEG_data[['sensor position', 'channel']].drop_duplicates().reset_index(drop=True).drop(['channel'], axis=1).reset_index(drop=False).rename(columns={'index':'channel'})['channel']



def plot_3dSurface_and_heatmap(stimulus, group, df):

    

    if group == 'c':

        group_name = 'Control'

    else:

        group_name = 'Alcoholic'

        

    temp_df = pd.pivot_table(df[['channel', 'sample num', 'sensor value']][(df['subject identifier'] == group) & (df['matching condition'] == stimulus)],

                                          index='channel', columns='sample num', values='sensor value').values.tolist()

    data = [go.Surface(z=temp_df, colorscale='Bluered')]



    layout = go.Layout(

        title='<br>3d Surface and Heatmap of Sensor Values for ' + stimulus + ' Stimulus for ' + group_name + ' Group',

        width=800,

        height=900,

        autosize=False,

        margin=dict(t=0, b=0, l=0, r=0),

        scene=dict(

            xaxis=dict(

                title='Time (sample num)',

                gridcolor='rgb(255, 255, 255)',

    #             erolinecolor='rgb(255, 255, 255)',

                showbackground=True,

                backgroundcolor='rgb(230, 230,230)'

            ),

            yaxis=dict(

                title='Channel',

                tickvals=channels,

                ticktext=sensor_positions,

                gridcolor='rgb(255, 255, 255)',

                zerolinecolor='rgb(255, 255, 255)',

                showbackground=True,

                backgroundcolor='rgb(230, 230, 230)'

            ),

            zaxis=dict(

                title='Sensor Value',

                gridcolor='rgb(255, 255, 255)',

                zerolinecolor='rgb(255, 255, 255)',

                showbackground=True,

                backgroundcolor='rgb(230, 230,230)'

            ),

            aspectratio = dict(x=1, y=1, z=0.5),

            aspectmode = 'manual'

        )

    )



    updatemenus=list([

        dict(

            buttons=list([   

                dict(

                    args=['type', 'surface'],

                    label='3D Surface',

                    method='restyle'

                ),

                dict(

                    args=['type', 'heatmap'],

                    label='Heatmap',

                    method='restyle'

                )             

            ]),

            direction = 'left',

            pad = {'r': 10, 't': 10},

            showactive = True,

            type = 'buttons',

            x = 0.1,

            xanchor = 'left',

            y = 1.1,

            yanchor = 'top' 

        ),

    ])



    annotations = list([

        dict(text='Trace type:', x=0, y=1.085, yref='paper', align='left', showarrow=False)

    ])

    layout['updatemenus'] = updatemenus

    layout['annotations'] = annotations



    fig = dict(data=data, layout=layout)

    iplot(fig)
stimulus = 'S1 obj'

S1_sample_df = sample_data(stimulus=stimulus, random_id=1)
plot_3dSurface_and_heatmap(stimulus=stimulus, group='a', df=S1_sample_df)
plot_3dSurface_and_heatmap(stimulus=stimulus, group='c', df=S1_sample_df)
stimulus = 'S2 match'

S2_m_sample_df = sample_data(stimulus=stimulus, random_id=1)
plot_3dSurface_and_heatmap(stimulus=stimulus, group='a', df=S2_m_sample_df)
plot_3dSurface_and_heatmap(stimulus=stimulus, group='c', df=S2_m_sample_df)
stimulus = 'S2 nomatch'

S2_nm_sample_df = sample_data(stimulus=stimulus, random_id=1)
plot_3dSurface_and_heatmap(stimulus=stimulus, group='a', df=S2_nm_sample_df)
plot_3dSurface_and_heatmap(stimulus=stimulus, group='c', df=S2_nm_sample_df)
## create the list of possible channel pairs

sample_corr_df = pd.pivot_table(S2_nm_sample_df[S2_nm_sample_df['subject identifier'] == 'a'], values='sensor value', index='sample num', columns='sensor position').corr()



list_of_pairs = []

j = 0

for column in sample_corr_df.columns:

    j += 1

    for i in range(j, len(sample_corr_df)):

        if column != sample_corr_df.index[i]:

            temp_pair = [column + '-' + sample_corr_df.index[i]]

            list_of_pairs.append(temp_pair)
def get_correlated_pairs_sample(threshold, correlation_df, group):

    ## create dictionary wheke keys are the pairs and values are the amount of high correlation pair

    corr_pairs_dict = {}

    for i in range(len(list_of_pairs)):

        temp_corr_pair = dict(zip(list_of_pairs[i], [0]))

        corr_pairs_dict.update(temp_corr_pair)



    j = 0

    for column in correlation_df.columns:

        j += 1

        for i in range(j, len(correlation_df)):

            if ((correlation_df[column][i] >= threshold) & (column != correlation_df.index[i])):

                corr_pairs_dict[column + '-' + correlation_df.index[i]] += 1



    corr_count = pd.DataFrame(corr_pairs_dict, index=['count']).T.reset_index(drop=False).rename(columns={'index': 'channel_pair'})

    print('Channel pairs that have correlation value >= ' + str(threshold) + ' (' + group + ' group):')

    print(corr_count['channel_pair'][corr_count['count'] > 0].tolist())
def plot_sensors_correlation(df, threshold_value):

    """Funtion plots the the correlation plots between sensor positions for each group"""

    correlations_alcoholic = pd.pivot_table(df[df['subject identifier'] == 'a'], 

                                          values='sensor value', index='sample num', columns='sensor position').corr()



    correlations_control = pd.pivot_table(df[df['subject identifier'] == 'c'], 

                                          values='sensor value', index='sample num', columns='sensor position').corr()



    fig = plt.figure(figsize=(17,10))

    ax = fig.add_subplot(121)

    ax.set_title('Alcoholic group', fontsize=14)

    mask = np.zeros_like(correlations_alcoholic, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(correlations_alcoholic, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5})



    ax = fig.add_subplot(122)

    ax.set_title('Control group', fontsize=14)

    mask = np.zeros_like(correlations_control, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(correlations_control, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5})



    plt.suptitle('Correlation between Sensor Positions for ' + df['matching condition'].unique()[0] + ' stimulus', fontsize=16)

    plt.show()

    

    get_correlated_pairs_sample(threshold=threshold_value, correlation_df=correlations_alcoholic, group='Alcoholic')

    print('\n')

    get_correlated_pairs_sample(threshold=threshold_value, correlation_df=correlations_control, group='Control')
plot_sensors_correlation(df=S1_sample_df, threshold_value=.97)
plot_sensors_correlation(df=S2_m_sample_df, threshold_value=.97)
plot_sensors_correlation(df=S2_nm_sample_df, threshold_value=.97)
def get_correlated_pairs(stimulus, threshold, group):

    """Funtion returns the df which holds pairs of channel with high correlation for stimulus, group and threshold provided"""

    corr_pairs_dict = {}

    trial_numbers_list = EEG_data['trial number'][(EEG_data['subject identifier'] == group) & (EEG_data['matching condition'] == stimulus)].unique()

    ## create dictionary wheke keys are the pairs and values are the amount of high correlation pair

    for i in range(len(list_of_pairs)):

        temp_corr_pair = dict(zip(list_of_pairs[i], [0]))

        corr_pairs_dict.update(temp_corr_pair)



    for trial_number in trial_numbers_list:    

        correlation_df = pd.pivot_table(EEG_data[(EEG_data['subject identifier'] == group) & (EEG_data['trial number'] == trial_number)], 

                                        values='sensor value', index='sample num', columns='sensor position').corr()



        j = 0 ## by setting the j we are going just through values below the main diagonal

        for column in correlation_df.columns:

            j += 1

            for i in range(j, len(correlation_df)):

                if ((correlation_df[column][i] >= threshold) & (column != correlation_df.index[i])):

                    corr_pairs_dict[column + '-' + correlation_df.index[i]] += 1



    corr_count = pd.DataFrame(corr_pairs_dict, index=['count']).T.reset_index(drop=False).rename(columns={'index': 'channel_pair'})

    corr_count['group'] = group

    corr_count['stimulus'] = stimulus

    return(corr_count)
def compare_corr_pairs(stimulus):

    """Function creates bar chart with the ratio of correlated pairs for both groups"""

    top_control_df = corr_pairs_df[(corr_pairs_df['group'] == 'c') & (corr_pairs_df['stimulus'] == stimulus)]

    top_alcoholic_df = corr_pairs_df[(corr_pairs_df['group'] == 'a') & (corr_pairs_df['stimulus'] == stimulus)]

    top_control_pairs = top_control_df.sort_values('count', ascending=False)['channel_pair'][:20]

    top_alcoholic_pairs = top_alcoholic_df.sort_values('count', ascending=False)['channel_pair'][:20]



    merged_df = pd.DataFrame({'channel_pair': top_control_pairs.append(top_alcoholic_pairs).unique()})

    merged_df = merged_df.merge(top_control_df[['channel_pair', 'count', 'trials_count']],

                                on='channel_pair', how='left').rename(columns={'count':'count_control', 'trials_count': 'trials_count_c'})

    merged_df = merged_df.merge(top_alcoholic_df[['channel_pair', 'count', 'trials_count']],

                                on='channel_pair', how='left').rename(columns={'count':'count_alcoholic', 'trials_count': 'trials_count_a'})



    data_1 = go.Bar(x=merged_df['channel_pair'],

                    y=(merged_df['count_alcoholic']/merged_df['trials_count_a']).apply(lambda x: round(x,2)),

                    text=merged_df['count_alcoholic'],

                    name='Alcoholic Group',

                    marker=dict(color='rgb(20,140,45)'))



    data_2 = go.Bar(x=merged_df['channel_pair'],

                    y=(merged_df['count_control']/merged_df['trials_count_c']).apply(lambda x: round(x,2)),

                    text=merged_df['count_control'],

                    name='Control Group',

                    marker=dict(color='rgb(200,100,45)'))



    layout = go.Layout(title='Amount of Correlated Pairs for the whole Data Set (' + stimulus + ' stimulus)',

                       xaxis=dict(title='Channel Pairs'),

                       yaxis=dict(title='Ratio'),

                       barmode='group')



    data = [data_1, data_2]

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
corr_pairs_df = pd.DataFrame({})

stimuli_list = ['S1 obj', 'S2 match', 'S2 nomatch']

## create df that holds information of total trial amount for each subject by stimulus

size_df = EEG_data.groupby(['subject identifier', 'matching condition'])[['trial number']].nunique().reset_index(drop=False).rename(columns={'trial number':'trials_count'})



for stimulus in stimuli_list:

    corr_pairs_df = corr_pairs_df.append(get_correlated_pairs(stimulus=stimulus, threshold=.9, group='c'))

    corr_pairs_df = corr_pairs_df.append(get_correlated_pairs(stimulus=stimulus, threshold=.9, group='a'))

corr_pairs_df = corr_pairs_df.merge(size_df, left_on=['group', 'stimulus'], right_on=['subject identifier', 'matching condition'], how='left')
compare_corr_pairs(stimulus='S1 obj')
compare_corr_pairs(stimulus='S2 match')
compare_corr_pairs(stimulus='S2 nomatch')
stimulus_list = EEG_data['matching condition'].unique().tolist() ## list of stimuli

channels_list = EEG_data['channel'].unique().tolist() ## list of channels



## get the Average Sensor Values for each channel by Subject group and Stimulus

agg_df = EEG_data.groupby(['subject identifier', 'matching condition', 'sensor position'], as_index=False)[['sensor value']].mean()
def get_p_value(stimulus, sensor):

    """

    Funtion takes the stimulus parameter and channel number and returns the p-value from Mann Whitney U-test (Alcoholic vs Control).

    """

    x = EEG_data[['sensor value']][(EEG_data['subject identifier'] == 'a') & 

                                   (EEG_data['matching condition'] == stimulus) & 

                                   (EEG_data['sensor position'] == sensor)]

    y = EEG_data[['sensor value']][(EEG_data['subject identifier'] == 'c') & 

                                   (EEG_data['matching condition'] == stimulus) & 

                                   (EEG_data['sensor position'] == sensor)]

    stat, p = mannwhitneyu(x=x, 

                           y=y,

                           alternative='two-sided')

    return p
## create empty df that will hold info about the statistica test

stat_test_results = pd.DataFrame({'stimulus': [], 

                                  'sensor': [],

                                  'p_value': []})



for sensor in tqdm(EEG_data['sensor position'].unique()):

    for stimulus in EEG_data['matching condition'].unique():

        temp_df = pd.DataFrame({'stimulus': stimulus,

                                'sensor': sensor,

                                'p_value': get_p_value(stimulus=stimulus, sensor=sensor)},

                               index=[0])

        stat_test_results = stat_test_results.append(temp_df)

        

stat_test_results = stat_test_results.reset_index(drop=True)

stat_test_results['reject_null'] = stat_test_results['p_value'] <= 0.05 ## check whether we can reject null hypothesis
stat_test_results.groupby(['stimulus'])[['reject_null']].mean()
data = []

for stimulus in stimulus_list:

    trace = go.Bar(x=stat_test_results['sensor'][stat_test_results['stimulus'] == stimulus],

                    y=stat_test_results['reject_null'][stat_test_results['stimulus'] == stimulus],

                    name=stimulus)

    data.append(trace)



layout = go.Layout(title='Amount of Significant Differences for each Channel',

                   xaxis=dict(title='Sensor Position'),

                   yaxis=dict(title='Is Significant',

                              showticklabels=False),

                   barmode='stack')

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# ## df that will be needed to plot the 'sesnors' that showed the difference for 3 stimuli in different color on ML future importance plot

# color_scale_df = stat_test_results.groupby(['sensor'])[['reject_null']].sum()

# color_scale_df['color'] = ''

# color_scale_df.loc[color_scale_df['reject_null'] != 3, 'color'] = 'rgba(48,203,231,0.7)'

# color_scale_df.loc[color_scale_df['reject_null'] == 3, 'color'] = 'rgba(222,45,38,0.8)'