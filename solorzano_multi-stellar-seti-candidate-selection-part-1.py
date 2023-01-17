from scipy.stats import skew
import numpy as np

def trimmed_skewness(x):
    sorted_x = np.sort(x)
    trimmed_x = sorted_x[1:-1]
    return skew(trimmed_x)
K_VALUES = [100, 200, 300] # Number of nearest neighbors to look at
BASE_METRIC = 'anomaly' # Could be 'model_residual' 
import pandas as pd

raw_data = pd.read_csv('../input/mag-modeling-results.csv', dtype={'source_id': str})
len(raw_data)
import numpy as np

ORIG_LABEL = 'phot_g_mean_mag'

def get_position_frame(data_frame):
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame['source_id'] = data_frame['source_id'].values
    distance = data_frame['distance'].values
    latitude = np.deg2rad(data_frame['b'].values)
    longitude = np.deg2rad(data_frame['l'].values)
    new_frame['z'] = distance * np.sin(latitude)
    projection = distance * np.cos(latitude)
    new_frame['x'] = projection * np.cos(longitude)
    new_frame['y'] = projection * np.sin(longitude)
    new_frame['blend_' + ORIG_LABEL] = data_frame['blend_' + ORIG_LABEL]
    new_frame['model_residual'] = data_frame['model_residual']
    new_frame['anomaly'] = data_frame['anomaly']
    new_frame['ra'] = data_frame['ra']
    new_frame['dec'] = data_frame['dec']
    new_frame['color_index'] = data_frame['color_index']
    new_frame[ORIG_LABEL] = data_frame[ORIG_LABEL]
    new_frame['tycho2_id'] = data_frame['tycho2_id']
    return new_frame
pos_data = get_position_frame(raw_data)
raw_data = None # Discard
from sklearn.neighbors import BallTree

ball_tree = BallTree(pos_data[['x', 'y', 'z']])
from scipy.stats import skew

max_k = np.max(K_VALUES)
base_metric_series = pos_data[BASE_METRIC].values
idx_source_id = pos_data.columns.get_loc('source_id')
idx_x = pos_data.columns.get_loc('x')
idx_y = pos_data.columns.get_loc('y')
idx_z = pos_data.columns.get_loc('z')
skewness_matrix = []
row_index = 0
for row in pos_data.itertuples(index=False):
    source_id = row[idx_source_id]
    source_pos = [row[idx_x], row[idx_y], row[idx_z]]
    # Get the K+1 nearest neighbors; results sorted by distance.
    distance_matrix, index_matrix = ball_tree.query([source_pos], k=max_k + 1)
    indexes = index_matrix[0]
    distances = distance_matrix[0]
    # The closest star is the current star - leave it out.
    assert indexes[0] == row_index
    assert distances[0] == 0
    skewness_row = []
    for k in K_VALUES:
        leave_one_out_indexes = indexes[1:k + 1]
        base_metric_neighbors = base_metric_series[leave_one_out_indexes]    
        assert len(base_metric_neighbors) == k
        skewness = trimmed_skewness(base_metric_neighbors)
        skewness_row.append(skewness)
    skewness_matrix.append(skewness_row)
    row_index += 1
skewness_frame = pd.DataFrame(skewness_matrix, columns=['k' + str(k) for k in K_VALUES])
skewness_frame.head(5)
skewness_means = [np.mean(skewness_frame['k' + str(k)]) for k in K_VALUES]
skewness_means
skewness_scales = [np.std(skewness_frame['k' + str(k)]) for k in K_VALUES]
skewness_scales
for i in range(len(K_VALUES)):
    k = K_VALUES[i]
    cn = 'k' + str(k)
    skewness_frame[cn] = (skewness_frame[cn] - skewness_means[i]) / skewness_scales[i]
skewness_frame.head(5)
best_std_skewness_series = []
best_k_series = []
for row in skewness_frame.itertuples(index=False):
    idx_max = np.argmax(np.abs(row))
    best_std_skewness_series.append(row[idx_max])
    best_k_series.append(K_VALUES[idx_max])
pos_data['best_std_skewness'] = best_std_skewness_series
pos_data['skewness_k'] = best_k_series
pos_data.head(5)[['source_id', 'skewness_k', 'best_std_skewness']]
n_to_show = 2000
high_outliers = pos_data.sort_values('best_std_skewness', ascending=False).head(n_to_show)
low_outliers = pos_data.sort_values('best_std_skewness', ascending=True).head(n_to_show)
random_stars = pos_data.sample(n_to_show)
import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
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
            opacity=0.5
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
def get_sun():
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame.loc[0] = ['sun', 0.0, 0.0, 0.0]
    new_frame.reset_index(inplace=True, drop=True)
    return new_frame
bstar_frame = pos_data[pos_data['source_id'] == '2081900940499099136']
high_outliers_with_extras = pd.concat([high_outliers, bstar_frame, get_sun()])
low_outliers_with_extras = pd.concat([low_outliers, bstar_frame, get_sun()])
random_with_extras = pd.concat([random_stars, bstar_frame, get_sun()])
plot_pos_frame(high_outliers_with_extras, 'green')
plot_pos_frame(low_outliers_with_extras, 'red')
plot_pos_frame(random_with_extras, 'gray')
pos_data.to_csv('stars_with_trimmed_skewness_of_' + BASE_METRIC + '.csv', index=False)