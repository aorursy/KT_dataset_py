import pandas as pd

work_data = pd.read_csv('../input/gaia-dr2-kepler-field-plain.csv', dtype={'source_id': str})
import numpy as np

def get_position_frame(data_frame):
    new_frame = pd.DataFrame(columns=['source_id', 'x', 'y', 'z'])
    new_frame['source_id'] = data_frame['source_id'].values
    new_frame['kepid'] = data_frame['kepid'].values
    new_frame['parallax'] = data_frame['parallax'].values
    new_frame['ra'] = data_frame['ra'].values
    new_frame['dec'] = data_frame['dec'].values
    new_frame['phot_g_mean_mag'] = data_frame['phot_g_mean_mag']
    distance = 1000.0 / data_frame['parallax'].values
    latitude = np.deg2rad(data_frame['b'].values)
    longitude = np.deg2rad(data_frame['l'].values)
    new_frame['z'] = distance * np.sin(latitude)
    projection = distance * np.cos(latitude)
    new_frame['x'] = projection * np.cos(longitude)
    new_frame['y'] = projection * np.sin(longitude)
    return new_frame
work_data = get_position_frame(work_data)
from sklearn.neighbors import BallTree

def get_bstar_neighbors(data_pool, pos_columns, k = 100):
    ball_tree = BallTree(work_data[pos_columns])
    source_pos = work_data.loc[work_data['source_id'] == '2081900940499099136', pos_columns]
    # Get the K+1 nearest neighbors; results sorted by distance.
    distance_matrix, index_matrix = ball_tree.query(source_pos, k=k + 1)
    indexes = index_matrix[0]
    distances = distance_matrix[0]
    # The closest star is the current star - leave it out.
    assert work_data.iloc[indexes[0]]['source_id'] == '2081900940499099136'
    assert distances[0] == 0
    neighbors = work_data.iloc[indexes[1:k + 1]]
    return pd.DataFrame({
        'source_id': neighbors['source_id'].values, 
        'kepid': neighbors['kepid'].values, 
        'distance_to_target': distances[1:k + 1],
        'parallax': neighbors['parallax'],
        'phot_g_mean_mag': neighbors['phot_g_mean_mag']})
space_neighbors = get_bstar_neighbors(work_data, ['x', 'y', 'z'], 100)
not_space_neighbors = work_data[~work_data['source_id'].isin(set(space_neighbors['source_id']))]
sky_neighbors = get_bstar_neighbors(not_space_neighbors, ['ra', 'dec'], 100)
np.random.seed(2018100001)
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

def get_matched_control_group(data_pool, exp_group, matching_columns, sample_fraction=0.5, k=15, n_factor=1):
    # Only a sample of the dataset is used to come up with a matched control group
    data_for_search = data_pool.sample(int(len(data_pool) * sample_fraction)).reset_index(drop=True)
    # Set of source_id's that are already in the matched control group
    source_id_set = set()
    
    # Scale matching columns
    scaler = StandardScaler()
    raw_x = data_for_search[matching_columns]
    scaled_x = scaler.fit_transform(raw_x)

    # Find nearest neighbor of each sample using sklearn's BallTree implementation.
    ball_tree = BallTree(scaled_x)
    data_source_id = data_for_search['source_id']
    shuffled_exp_group = exp_group.sample(frac=1)    
    idx_source_id = shuffled_exp_group.columns.get_loc('source_id')
    matching_col_indexes = [shuffled_exp_group.columns.get_loc(cn) for cn in matching_columns]
    results = pd.DataFrame(columns=data_for_search.columns)
    for row in shuffled_exp_group.itertuples(index=False):
        raw_source_pos = [row[i] for i in matching_col_indexes]
        source_pos = scaler.transform([raw_source_pos])
        _, index_matrix = ball_tree.query(source_pos, k=k)
        assert len(index_matrix) == 1
        indexes = index_matrix[0]
        count = 0
        for i in range(k):
            data_index = indexes[i]
            source_id = data_source_id[data_index]
            if source_id not in source_id_set:
                source_id_set.add(source_id)
                results.loc[len(results)] = data_for_search.iloc[data_index]
                count += 1
                if count >= n_factor:
                    break
    return results
prev_id_set = set(space_neighbors['source_id']).union(set(sky_neighbors['source_id']))
work_data_remaining = work_data[~work_data['source_id'].isin(prev_id_set)]
control_group = get_matched_control_group(work_data_remaining, space_neighbors, ['parallax', 'phot_g_mean_mag'])
SAVED_COLUMNS = ['source_id', 'kepid', 'distance_to_target']
space_neighbors[SAVED_COLUMNS].to_csv('100-space-neighbors-of-bstar.csv', index=False)
sky_neighbors[SAVED_COLUMNS].to_csv('100-sky-neighbors-of-bstar.csv', index=False)
control_group[['source_id', 'kepid']].to_csv('100-matched-controls-for-neighbors-of-bstar.csv', index=False)