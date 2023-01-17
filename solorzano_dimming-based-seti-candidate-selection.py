import pandas as pd



pos_data = pd.read_csv('../input/hipparcos-region-mean-mag-change.csv', dtype={'source_id': str})

len(pos_data)
pos_data.columns
from scipy.stats import binom # Library used to calculate binomial density and standard deviation.

import numpy as np



def get_selection_fraction_results(pos_data, from_f, to_f, from_g, to_g, is_dim: bool):

    anomaly_selection_fractions = np.logspace(np.log10(from_f), np.log10(to_f), 30)

    regional_mag_change_selection_fractions = np.logspace(np.log10(from_g), np.log10(to_g), 30) 

    anomaly_values = pos_data['anomaly'].values

    if is_dim:

        anomaly_values = -anomaly_values

    pos_data_anomaly_ordering = np.argsort(anomaly_values)

    # The regional metric is sorted in descending order.

    pos_data_regional_mag_change_ordering = np.argsort(-pos_data['smooth_regional_metric'].values)    

    results_frame = pd.DataFrame(columns=['f', 'g', 'cand_count', 'expected_count', 'n_stdevs', 'ratio', 'cumul_density'])

    pos_data_len = len(pos_data)

    for asf in anomaly_selection_fractions:

        for ssf in regional_mag_change_selection_fractions:

            expected_count = pos_data_len * asf * ssf

            anomaly_count = int(round(pos_data_len * asf))

            regional_mag_change_count = int(round(pos_data_len * ssf))

            anomalous_sub_frame = pos_data.iloc[pos_data_anomaly_ordering[:anomaly_count]]

            skewed_sub_frame = pos_data.iloc[pos_data_regional_mag_change_ordering[:regional_mag_change_count]]

            source_id_in_common = set(anomalous_sub_frame['source_id']).intersection(set(skewed_sub_frame['source_id']))

            cand_count = len(source_id_in_common)

            actual_p = (anomaly_count / pos_data_len) * (regional_mag_change_count / pos_data_len)

            ratio = cand_count / expected_count

            cumul_density = binom.cdf(cand_count - 1, pos_data_len, actual_p)

            binom_sd = binom.std(pos_data_len, actual_p)

            n_stdevs = (cand_count - expected_count) / binom_sd

            results_frame.loc[len(results_frame)] = [asf, ssf, cand_count, expected_count, n_stdevs, ratio, cumul_density]

    return results_frame
STDEVS_THRESHOLD = 5.0



def shown_results(results_frame):

    return results_frame[results_frame['n_stdevs'] >= STDEVS_THRESHOLD].sort_values('ratio', ascending=False).head(10)
dim_results_frame = get_selection_fraction_results(pos_data, 0.0008, 0.008, 0.02, 0.2, is_dim=True)

shown_results(dim_results_frame)
bright_results_frame = get_selection_fraction_results(pos_data, 0.003, 0.03, 0.03, 0.3, is_dim=False)

shown_results(bright_results_frame)
def get_candidates(results_frame, pos_data, is_dim: bool):

    anomaly_values = pos_data['anomaly'].values

    if is_dim:

        anomaly_values = -anomaly_values

    pos_data_anomaly_ordering = np.argsort(anomaly_values)

    pos_data_regional_mag_change_ordering = np.argsort(-pos_data['smooth_regional_metric'].values)    

    top_count_row = results_frame[results_frame['n_stdevs'] >= STDEVS_THRESHOLD].reset_index(drop=True).sort_values('ratio', ascending=False).iloc[0]

    select_anomaly_fraction = top_count_row['f']

    select_regional_mag_change_fraction = top_count_row['g']

    pos_data_len = len(pos_data)

    anomaly_count = int(round(pos_data_len * select_anomaly_fraction))

    regional_mag_change_count = int(round(pos_data_len * select_regional_mag_change_fraction))

    anomalous_sub_frame = pos_data.iloc[pos_data_anomaly_ordering[:anomaly_count]]

    skewed_sub_frame = pos_data.iloc[pos_data_regional_mag_change_ordering[:regional_mag_change_count]]

    source_id_in_common = set(anomalous_sub_frame['source_id']).intersection(set(skewed_sub_frame['source_id']))

    candidates = pos_data[pos_data['source_id'].isin(source_id_in_common)].reset_index(drop=True)

    return candidates
dim_candidates = get_candidates(dim_results_frame, pos_data, is_dim=True)

dim_candidates['is_dim'] = True
bright_candidates = get_candidates(bright_results_frame, pos_data, is_dim=False)

bright_candidates['is_dim'] = False
len(dim_candidates)
len(bright_candidates)
import plotly.plotly as py

import plotly.offline as py

import plotly.graph_objs as go



py.init_notebook_mode(connected=False)
import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)



def get_sun():

    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])

    new_frame.loc[0] = ['sun', 0.0, 0.0, 0.0]

    return new_frame



def get_row_color(row, dim_color='green', bright_color='red', sun_color = 'blue', bstar_color = 'black'):

    if row['source_id'] == '2081900940499099136':

        return bstar_color

    elif row['source_id'] == 'sun':

        return sun_color

    elif row['is_dim']:

        return dim_color

    else:

        return bright_color

    

def plot_pos_frame(pos_frame, dim_color='green', bright_color='red', sun_color = 'blue', bstar_color = 'black'):

    star_color = [get_row_color(row, dim_color, bright_color, sun_color, bstar_color) for _, row in pos_frame.iterrows()]

    trace1 = go.Scatter3d(

        x=pos_frame['x'],

        y=pos_frame['y'],

        z=pos_frame['z'],

        mode='markers',

        text=pos_frame['source_id'],

        marker=dict(

            size=3,

            color=star_color,

            opacity=0.67

        )

    )

    scatter_data = [trace1]

    layout = go.Layout(

        margin=dict(

            l=0,

            r=0,

            b=0,

            t=0

        ),

        scene=dict(

            xaxis=dict(

                range=[-450, 450]

            ),

            yaxis=dict(

                range=[-450, 450]

            ),

            zaxis=dict(

                range=[-450, 450]

            )

        ),

    )

    fig = go.Figure(data=scatter_data, layout=layout)

    py.iplot(fig)
candidates_with_extras = pd.concat([dim_candidates, bright_candidates, pos_data[pos_data['source_id'] == '2081900940499099136'], get_sun()])

plot_pos_frame(candidates_with_extras)
dim_candidates[['source_id', 'tycho2_id', 'anomaly', 'smooth_regional_metric']].to_csv('dimming-region-dim-candidates.csv', index=False)
bright_candidates[['source_id', 'tycho2_id', 'anomaly', 'smooth_regional_metric']].to_csv('dimming-region-bright-candidates.csv', index=False)