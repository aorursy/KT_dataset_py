import pandas as pd

file_folder = '../input/'
file_name = 'stars_with_trimmed_skewness_of_anomaly.csv'
pos_data = pd.read_csv(file_folder + file_name, dtype={'source_id': str})
len(pos_data)
pos_data.columns
from scipy.stats import binom

def get_ratio_scale(n, p, confidence_level=0.9999):
    cumulative_density = 0
    max_k = None
    for k in range(0, n):
        density = binom.pmf(k, n, p)
        cumulative_density += density
        if cumulative_density >= confidence_level:
            max_k = k
            break
    if max_k is None:
        max_k = n
    expected_k = n * p
    return max_k / expected_k
get_ratio_scale(200, 0.5)
get_ratio_scale(200, 0.01)
import numpy as np

anomaly_selection_fractions = np.logspace(np.log10(0.0008), np.log10(0.008), 30)
skewness_selection_fractions = np.logspace(np.log10(0.008), np.log10(0.08), 30)
NEIGHBOR_METRIC_COLUMN = 'best_std_skewness'
pos_data_anomaly_ordering = np.argsort(pos_data['anomaly'].values)
pos_data_skewness_ordering = np.argsort(pos_data[NEIGHBOR_METRIC_COLUMN].values)
from scipy.stats import binom

results_frame = pd.DataFrame(columns=['f', 'g', 'cand_count', 'expected_count', 'ratio', 'scaled_ratio', 'cumul_density'])
pos_data_len = len(pos_data)
for asf in anomaly_selection_fractions:
    for ssf in skewness_selection_fractions:
        expected_count = pos_data_len * asf * ssf
        anomaly_count = int(round(pos_data_len * asf))
        skewness_count = int(round(pos_data_len * ssf))
        anomalous_sub_frame = pos_data.iloc[pos_data_anomaly_ordering[:anomaly_count]]
        skewed_sub_frame = pos_data.iloc[pos_data_skewness_ordering[:skewness_count]]
        source_id_in_common = set(anomalous_sub_frame['source_id']).intersection(set(skewed_sub_frame['source_id']))
        cand_count = len(source_id_in_common)
        actual_p = (anomaly_count / pos_data_len) * (skewness_count / pos_data_len)
        ratio = cand_count / expected_count
        ratio_scale = get_ratio_scale(pos_data_len, actual_p)
        if ratio_scale == 0:
            raise ValueError('p=' + str(actual_p) + ' is too small!')
        scaled_ratio = ratio / ratio_scale
        cumul_density = binom.cdf(cand_count - 1, pos_data_len, actual_p)
        results_frame.loc[len(results_frame)] = [asf, ssf, cand_count, expected_count, ratio, scaled_ratio, cumul_density]
results_frame.sort_values('scaled_ratio', ascending=False).head(10)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 5)
plt.hist(results_frame['cumul_density'], bins=100)
plt.title('Distribution of cumulative binomial density\nin search space')
plt.xlabel('Cumulative density')
plt.ylabel('Frequency')
plt.show()
top_count_row = results_frame.sort_values('scaled_ratio', ascending=False).iloc[0]
select_anomaly_fraction = top_count_row['f']
select_anomaly_fraction
select_skewness_fraction = top_count_row['g']
select_skewness_fraction
anomaly_count = int(round(pos_data_len * select_anomaly_fraction))
skewness_count = int(round(pos_data_len * select_skewness_fraction))
anomalous_sub_frame = pos_data.iloc[pos_data_anomaly_ordering[:anomaly_count]]
skewed_sub_frame = pos_data.iloc[pos_data_skewness_ordering[:skewness_count]]
source_id_in_common = set(anomalous_sub_frame['source_id']).intersection(set(skewed_sub_frame['source_id']))
candidates = pos_data[pos_data['source_id'].isin(source_id_in_common)]
len(candidates)
def get_sun():
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame.loc[0] = ['sun', 0.0, 0.0, 0.0]
    return new_frame
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
candidates_with_extras = pd.concat([candidates, pos_data[pos_data['source_id'] == '2081900940499099136'], get_sun()])
import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=False)
def plot_pos_frame(pos_frame, star_color, sun_color = 'blue', bstar_color = 'black'):    
    star_color = [(bstar_color if row['source_id'] == '2081900940499099136' else (sun_color if row['source_id'] == 'sun' else star_color)) for _, row in pos_frame.iterrows()]
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
        )
    )
    fig = go.Figure(data=scatter_data, layout=layout)
    py.iplot(fig)
plot_pos_frame(candidates_with_extras, 'red')
candidates.to_csv('clustered-bright-candidates.csv', index=False)