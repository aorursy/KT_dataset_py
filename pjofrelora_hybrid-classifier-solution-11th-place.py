# Import libraries to store data

import pandas as pd

import numpy as np

# Import libraries to visualize data

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

import seaborn as sns

from ipywidgets import interact, interact_manual

# Import libraries to process data

import tsfresh

from scipy.signal import welch

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# Import libraries to classify data and score results

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, f1_score

import xgboost as xgb

# Import libraries used in functions and for feedback

import time

import random

import warnings

warnings.filterwarnings("ignore") # Setting values on a Pandas DataFrame 

# sometimes throws errors, so I'll silence the warnings.



# The below has to be set because matplotlib and XGB crash 

# by causing a duplicate copy of a library to be run. It's a 

# known issue, unfortunately.

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
f_train = pd.read_csv('../input/X_train.csv')

t_train = pd.read_csv('../input/y_train.csv')

test = pd.read_csv('../input/X_test.csv')

train = pd.merge(t_train, f_train, how = 'outer', left_on = 'series_id', right_on = 'series_id')
train.head()
train.groupby('series_id').mean().describe()
test.groupby('series_id').mean().describe()
surfaces = list(train.surface.unique())

meta = list(train.columns[0:5])

position = list(train.columns[5:9])

motion = list(train.columns[9:])

features = position+motion
@interact

def plot_series(surface = surfaces, series_number = list(range(1,10))):

    subset = train[train.surface == surface]

    g = subset.groupby('series_id')

    series_id = list(g.groups.keys())[series_number]

    plt.figure(figsize = (10,24))

    signal = g.get_group(series_id)

    for i,feature in enumerate(features):

        ax = plt.subplot(5,2,i+1)

        ax.plot(signal[feature])

        ax.set_title(feature)

        ax.set_xlabel('Measurement Number')

        ax.set_ylabel('Magnitude')
@interact

def plot_overlapping_motion_graphs(feature = motion, number = (10,100,10)):

    colors = ['grey','red','orange','yellow','green','blue','indigo','pink','brown']

    plt.figure(figsize = (12,8))

    plt.xlabel('Measurement Number')

    plt.ylabel('Magnitude')

    plt.title('Overlapping Motion Signals Sorted by Color')

    for surface, color in zip(surfaces,colors):

        series_ids = train.loc[train.surface==surface].series_id.unique()

        for i, series in enumerate(series_ids[0:number]):

            if i == 0:

                plt.plot(train[train.series_id==series][feature].values, 

                     alpha = 0.05, 

                     color = color,

                     label = surface)

            else:

                plt.plot(train[train.series_id==series][feature].values, 

                     alpha = 0.05, 

                     color = color)

    leg = plt.legend()

    for lh in leg.legendHandles: 

        lh.set_alpha(1)
def RMS(signal):

    signal = np.array(signal)

    return np.sqrt((signal**2).mean())
@interact

def plot_RMS_histograms(feature = motion):

    plt.figure(figsize = (12,8))

    df = pd.DataFrame()

    g = train.groupby('series_id')

    df['Surface'] = g.max()['surface']

    df['RMS'] = g[feature].apply(RMS)

    for surface in surfaces:

        sns.distplot(df.loc[df.Surface==surface]['RMS'], 

                     hist = False, 

                     label = surface.title(),)

    plt.legend()

    plt.title('RMS Histograms for One Feature by Surface')
@interact

def plot_psd_overview(feature = motion, segment_size = (4,6)):

    plt.figure(figsize = (12,8))

    nperseg = 2**segment_size

    for surface in surfaces:

        freq,psd = welch(train[train.surface==surface][feature], 

                         nperseg = nperseg, 

                         return_onesided = True,

                         scaling = 'spectrum')

        plt.plot(psd,label = surface)

    plt.legend()

    plt.title('PSD Estimate by Surface (grouped data)')

    plt.xlabel('Frequency (1/sample)')

    plt.ylabel('Magnitude of Signal')
def psd(signal):

    _,PSD = welch(signal, nperseg = 128, scaling = 'spectrum')

    return PSD
@interact

def plot_all_psds(feature = motion, surface = surfaces):

    plt.figure(figsize = (12,8))

    max_value = []

    # Plot PSDs of every other surface in grey first

    g = train[train.surface != surface].groupby('series_id')

    for PSD in g[feature].apply(psd):

        max_value.append(max(PSD))

        plt.plot(PSD, color = 'gray', alpha = 0.05)

    # Plot PSDs of surface in red

    g = train[train.surface == surface].groupby('series_id')

    for PSD in g[feature].apply(psd):

        max_value.append(max(PSD))

        plt.plot(PSD, color = 'red', alpha = 0.075)

    

    ylim = np.quantile(max_value, 0.99)

    plt.ylim(0,ylim)

    plt.title('Overlapped PSDs for one motion; \none surface (red) compared to all surfaces (grey)')

    plt.xlabel('Frequency')

    plt.ylabel('Magnitude at Frequency')
g = train.groupby(by = 'group_id')

g67 = g.get_group(67)

g67_g = g67.groupby('series_id')

f, axes = plt.subplots(2, 2, figsize = (12,8))

axs = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

orientations = ['orientation_X','orientation_Y','orientation_Z','orientation_W']

for name, group in g67_g:

    for ax,orientation in zip(axs,orientations):

        group[orientation].plot(label = name, ax=ax, title = orientation)
def find_matches(current_signal, df, in_front = True, threshold = 0.01):

    ## Returns a dataframe of the series ids, sums, and position (front/end) of the likely matches for the given front and end values.

    orientations = ['orientation_X','orientation_Y','orientation_Z','orientation_W']

    selection = ['series_id'] + orientations

    if in_front:

        end = current_signal.iloc[127][selection]

        front = df.loc[df.measurement_number ==0][selection]

    else:

        front = current_signal.iloc[0][selection]

        end = df.loc[df.measurement_number ==127][selection]

    compare = (front-end).abs()

    if in_front:

        compare.series_id = front.series_id

    else:

        compare.series_id = end.series_id

    compare['sums'] = compare[orientations].sum(axis=1)

    compare['in_front'] = in_front

    compare.drop(labels = orientations, axis = 1, inplace = True)

    compare.sort_values(by = 'sums', inplace = True)

    if any(compare.sums < threshold):

        return compare.loc[compare.sums < threshold], True

    else:

        return 0,False
def continuity(current_signal, compare_signal, in_front = True, merge_threshold = 4, slope_threshold = 6):

    orientations = ['orientation_X','orientation_Y','orientation_Z','orientation_W']

    select = ['series_id'] + orientations

    if in_front:

        end = current_signal[select]

        front = compare_signal[select]

    else:

        front = current_signal[select]

        end = compare_signal[select]

    stitched = pd.concat([end,front])

    diff = stitched.diff()

    

    # Check for continuity based on difference at merge point

    mean = diff[orientations].iloc[120:127].mean()

    std = diff[orientations].iloc[120:127].std()

    test = (diff[orientations].iloc[128]-mean)/std

    if any(test.abs() > merge_threshold):

        return False

    

    # Check for continuity based on slopes (t-test of means)

    slope_front = diff[orientations].iloc[120:127].mean()

    std_front = diff[orientations].iloc[120:127].std()

    var_front = std_front.apply(np.square)

    slope_end = diff[orientations].iloc[129:136].mean()

    std_end = diff[orientations].iloc[129:136].std()

    var_end = std_end.apply(np.square)

    t_value = (slope_front-slope_end)/(var_front/8+var_end/8).apply(np.sqrt)

    if any(t_value.abs() > slope_threshold):

        return False

    

    return True
def group_series(df, match_threshold = 0.01, continuity_thresholds = (4,6)):

    df_c = df.copy()

    # Create variables

    lookup_dictionary = {}

    new_group_id = -1

    l_init = len(df_c)

    t_start = time.time()

    # Cycle through df

    while len(df_c):

        # Give feedback with regard to end time

        t_elapsed = time.time()

        t_diff = t_elapsed-t_start

        l_now = len(df_c)

        frac_done = 1-(l_now/l_init)

        if frac_done > 0:

            ETA = t_diff/frac_done-t_diff

        else:

            ETA = 0

        message = 'Grouping Progress:{0:1.2f}%\tETA:{1:1.2f}min'.format(frac_done*100,ETA/60 )

        print(message, end='\r')

        # Add new group to lookup dictionary

        new_group_id +=1

        lookup_dictionary[new_group_id] = []

        # Start by selecting first available signal

        start_series = df_c.series_id.unique()[0]

        start_signal = df_c.loc[df_c.series_id==start_series]

        current_signal = start_signal

        # Add series to lookup dictionary

        lookup_dictionary[new_group_id].append(start_series)

        # Delete entry from df

        indices =  start_signal.index

        df_c.drop(indices,inplace = True)

        # Prepare to expand forwards and backwards

        expand_forward = True

        expand_backward = True

        # Expand signal forward first

        while (expand_forward) & (len(df_c)>0):

            candidates, match = find_matches(current_signal, 

                                             df_c, 

                                             in_front = True,

                                             threshold = match_threshold)

            continuity_tracker = []

            if match:

                for _,candidate in candidates.iterrows():

                    compare_signal = df_c.loc[df_c.series_id == candidate.series_id]

                    if continuity(current_signal, 

                                  compare_signal, 

                                  in_front = candidate.in_front,

                                  merge_threshold = continuity_thresholds[0],

                                  slope_threshold = continuity_thresholds[1]):

                        current_series = candidate.series_id

                        lookup_dictionary[new_group_id].append(current_series)

                        current_signal = df_c.loc[df_c.series_id==current_series]

                        df_c.drop(current_signal.index, inplace = True)

                        continuity_tracker.append(True)

                        break

                    else:

                        continuity_tracker.append(False)

                if any(continuity_tracker):

                    continue

                else: # If nothing passed the continuity check then stop expanding forward

                    expand_forward = False

            else: # If no candidates, then stop expanding forward

                expand_forward = False

            # Clean variable space

            del candidates,match

        # Expand signal backwards

        current_signal = start_signal

        while (expand_backward) & (len(df_c)>0):

            candidates, match = find_matches(current_signal, 

                                             df_c, 

                                             in_front = False,

                                             threshold = match_threshold)

            continuity_tracker = []

            if match:

                for _,candidate in candidates.iterrows():

                    compare_signal = df_c.loc[df_c.series_id == candidate.series_id]

                    if continuity(current_signal, 

                                  compare_signal, 

                                  in_front = candidate.in_front,

                                  merge_threshold = continuity_thresholds[0],

                                  slope_threshold = continuity_thresholds[1]):

                        current_series = candidate.series_id

                        lookup_dictionary[new_group_id].insert(0,current_series)

                        current_signal = df_c.loc[df_c.series_id==current_series]

                        df_c.drop(current_signal.index, inplace = True)

                        continuity_tracker.append(True)

                        break

                    else:

                        continuity_tracker.append(False)

                if any(continuity_tracker):

                    continue

                else: # If nothing passed the continuity check then stop expanding forward

                    expand_backward = False

            else: # If no candidates, then stop expanding forward

                expand_backward = False

            # Clean variable space

            del candidates, match

    print("Grouping Done                             ")

    print("Total Groups: {}".format(new_group_id))

    return lookup_dictionary
def stitch_series(df,ordered_series_array):

    osa = ordered_series_array

    g = df.groupby('series_id')

    sub_dfs = []

    for series in osa:

        temp_df = g.get_group(series)

        sub_dfs.append(temp_df)

    df_out = pd.concat(sub_dfs)

    df_out.measurement_number = list(range(0,len(df_out)))

    return df_out
def recast_df(df, match_threshold = 0.01, continuity_thresholds = (4,6)):

    lookup_dict = group_series(df, 

                               match_threshold=match_threshold, 

                               continuity_thresholds=continuity_thresholds)

    sub_dfs = []

    for key,group in lookup_dict.items():

        temp_df = stitch_series(df,group)

        temp_df['new_group_id'] = key

        sub_dfs.append(temp_df)

    stitched = pd.concat(sub_dfs)

    return stitched
stitched_train = recast_df(train, match_threshold=0.05, continuity_thresholds=(6,9))
stitched_train.tail()
@interact

def plot_stitched_signals(surface = surfaces, group = (1,10)):

    sub_stitched = stitched_train.loc[stitched_train.surface==surface]

    g = sub_stitched.groupby('new_group_id')

    group_id = list(g.groups.keys())[group]

    plt.figure(figsize = (12,18))

    signal = g.get_group(group_id)

    for i,feature in enumerate(features):

        ax = plt.subplot(5,2,i+1)

        ax.plot(signal['measurement_number'], signal[feature])

        ax.set_title(feature)

        ax.set_xlabel('Measurement Number')

        ax.set_ylabel('Magnitude')
g = stitched_train.groupby('new_group_id')

surface_count = []

sample_count = []

for name, group in g:

    surfaces = group.surface.unique()

    surface_count.append(len(surfaces))

    sample_count.append(len(group))

print('Max Surface Count:{0}'.format(max(surface_count)))

print('Min Sample Count:{0}'.format(min(sample_count)))

print('Max Sample Count:{0}'.format(max(sample_count)))
g = stitched_train.groupby('group_id')

sample_count = []

for name, group in g:

    surfaces = group.surface.unique()

    surface_count.append(len(surfaces))

    sample_count.append(len(group))

print('Max sample count in original groups:{0}'.format(max(sample_count)))
# Drop unnecessary columns from train, and add new columns to keep track of train/test

train.drop(['row_id','group_id'], axis = 1, inplace = True)

train['orig_id'] = train.series_id

train['test'] = 0

# Drop unnecessary columns from test, and add new columns as above

test.drop(['row_id'], axis = 1, inplace = True)

test['surface'] = 'no_surface'

test['orig_id'] = test.series_id

test['test'] = 1

test.series_id += train.series_id.max()+1
all_data = pd.concat([train,test], sort=False, ignore_index=True)
all_stitched = recast_df(all_data, match_threshold=0.05, continuity_thresholds=(9,9))
g = all_stitched.groupby('new_group_id')

surface_count = []

sample_count = []

for name, group in g:

    surfaces = group.surface.unique()

    surface_count.append(len(surfaces))

    sample_count.append(len(group))

print('Max Surface Count:{0}'.format(max(surface_count)))

print('Min Sample Count:{0}'.format(min(sample_count)))

print('Max Sample Count:{0}'.format(max(sample_count)))
g = all_stitched.groupby('new_group_id')

new_dfs = []

error_surfaces = []

error_group_ids = []

matched = 0

unmatched = 0

unmatched_ids = []

for name, group in g:

    surfaces = list(group.surface.unique())

    # If a group contains one known surface and a total of two surfaces, then reclassify the unknown data as appropriate

    if (len(surfaces)==2) & ('no_surface' in surfaces):

        surfaces.remove('no_surface')

        group['surface'] = surfaces[0]

        new_dfs.append(group)

        matched +=1

    # If a group contains two known surfaces then store it as a group "in error"    

    elif (len(surfaces)==2) & ('no_surface' not in surfaces):

        error_surfaces.append(surfaces)

        new_dfs.append(group)

    # If a group has three surfaces or more than store it as a group "in error"    

    elif (len(surfaces)>2):

        error_surfaces.append(surfaces)

        error_group_ids.append(name)

        new_dfs.append(group)

    # Else, increase the unmatched counter by one if there's only one surface and it's unknown

    else:

        new_dfs.append(group)

        if surfaces[0] == 'no_surface':

            unmatched += 1

            unmatched_ids.append(name)

# Make a new dataframe from reclassified data

sorted_data = pd.concat(new_dfs)

sorted_data.reset_index(inplace=True, drop = True)
print('Number of unmatched groups:{0}'.format(unmatched))
error_surfaces
error_group_ids
# Some groups take >2 mins to plot because of the number of available samples

@interact

def plot_grouped_data(group_id = (1,90)):

    g = all_stitched.groupby('new_group_id')

    plt.figure(figsize = (12,30))

    signal = g.get_group(group_id)

    for i,feature in enumerate(features):

        ax = plt.subplot(5,2,i+1)

        sns.lineplot(x = 'measurement_number',

                     y = feature,

                     data = signal, 

                     hue='surface', 

                     ax=ax);

        ax.set_title(feature)

        ax.set_xlabel('Measurement Number')

        ax.set_ylabel('Magnitude')
def reassign_surfaces(new_surface_list, group_id, df):

    # Pull group from dataframe first

    error_group = df[sorted_data.new_group_id == group_id]

    reassign = {}

    start = 1

    end = 1

    segment = 0

    escape = 0 

    while escape == 0:

        # Find segments that are represented by one surface (do this recursively)

        temp_dict = {}

        surface = error_group.surface.iloc[0]

        start = error_group.first_valid_index()

        if len(error_group.surface.unique()) > 1:

            end = error_group.loc[error_group.surface != surface].first_valid_index()

        else:

            end = error_group.last_valid_index()

            escape = 1

        error_group.drop(range(start,end),inplace = True)

        

        #Store Data

        if surface == 'no_surface':

            temp_dict['surface'] = new_surface_list[segment]

            temp_dict['start'] = start

            temp_dict['end'] = end

            reassign[segment] = temp_dict

            segment += 1

    # Use data to reassign surfaces

    for key,value in reassign.items():

        df.iloc[value['start']:value['end'],1] = value['surface']

    

    fixed_group = df[sorted_data.new_group_id == group_id]

    plt.figure(figsize = (12,4))

    sns.lineplot(x = 'measurement_number',

                     y = 'orientation_X',

                     data = fixed_group, 

                     hue='surface');
surface_2 = ['concrete','wood','wood']

reassign_surfaces(surface_2,2,sorted_data)
surface_9 = ['concrete', 'concrete']

reassign_surfaces(surface_9, 9, sorted_data)
output = sorted_data[sorted_data.test == 1][['orig_id','surface']].groupby('orig_id').max().reset_index()

output.to_csv(path_or_buf = 'preClassification.csv', 

              header = ['series_id','surface'], 

              index = False)

estimated_score = sum(output.surface!='no_surface')/len(output)
real_score = 0.55*0.6878 + 0.45*0.8175

print('Estimated Score: {0:1.3f}'.format(estimated_score))

print('Real Score: {0:1.3f}'.format(real_score))

print('Accuracy: {0:1.3f}'.format(real_score/estimated_score))
extract_from = sorted_data.drop(['surface','test','new_group_id', 'orig_id']+orientations, axis = 1)

extract_from.head()
params = {'abs_energy':None,

          'absolute_sum_of_changes':None,

          'agg_autocorrelation':[{'f_agg':'var','maxlag':32}],

          'change_quantiles':[{'ql':0.25,'qh':0.75,'isabs':True, 'f_agg':'mean'},

                             {'ql':0.25,'qh':0.75,'isabs':True, 'f_agg':'std'}],

          'cid_ce':[{'normalize':True},{'normalize':False}],

          'fft_aggregated':[{'aggtype': 'centroid'},

                            {'aggtype': 'variance'},

                            {'aggtype': 'skew'},

                            {'aggtype': 'kurtosis'}],

          'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}],

          'standard_deviation': None,

          'variance': None,

          'skewness': None,

          'kurtosis': None,

          'maximum': None,

          'minimum': None,

          'sample_entropy':None,

          'mean_abs_change':None,

          'sum_values':None,

          'quantile': [{'q': 0.1},

                       {'q': 0.2},

                       {'q': 0.3},

                       {'q': 0.4},

                       {'q': 0.6},

                       {'q': 0.7},

                       {'q': 0.8},

                       {'q': 0.9}],

          'large_standard_deviation': [{'r': 0.25},{'r':0.35}],

          'fft_coefficient': [{'coeff': 0, 'attr': 'real'},

                              {'coeff': 1, 'attr': 'real'},

                              {'coeff': 2, 'attr': 'real'},

                              {'coeff': 3, 'attr': 'real'},

                              {'coeff': 4, 'attr': 'real'},

                              {'coeff': 5, 'attr': 'real'},

                              {'coeff': 6, 'attr': 'real'},

                              {'coeff': 7, 'attr': 'real'},

                              {'coeff': 8, 'attr': 'real'},

                              {'coeff': 9, 'attr': 'real'},

                              {'coeff': 10, 'attr': 'real'},

                              {'coeff': 11, 'attr': 'real'},

                              {'coeff': 12, 'attr': 'real'},

                              {'coeff': 13, 'attr': 'real'},

                              {'coeff': 14, 'attr': 'real'},

                              {'coeff': 15, 'attr': 'real'},

                              {'coeff': 16, 'attr': 'real'},

                              {'coeff': 17, 'attr': 'real'},

                              {'coeff': 18, 'attr': 'real'},

                              {'coeff': 19, 'attr': 'real'},

                              {'coeff': 20, 'attr': 'real'},

                              {'coeff': 21, 'attr': 'real'},

                              {'coeff': 22, 'attr': 'real'},

                              {'coeff': 23, 'attr': 'real'},

                              {'coeff': 24, 'attr': 'real'},

                              {'coeff': 25, 'attr': 'real'},

                              {'coeff': 26, 'attr': 'real'},

                              {'coeff': 27, 'attr': 'real'},

                              {'coeff': 28, 'attr': 'real'},

                              {'coeff': 29, 'attr': 'real'},

                              {'coeff': 30, 'attr': 'real'},

                              {'coeff': 31, 'attr': 'real'},

                              {'coeff': 32, 'attr': 'real'},

                              {'coeff': 33, 'attr': 'real'},

                              {'coeff': 34, 'attr': 'real'},

                              {'coeff': 35, 'attr': 'real'},

                              {'coeff': 36, 'attr': 'real'},

                              {'coeff': 37, 'attr': 'real'},

                              {'coeff': 38, 'attr': 'real'},

                              {'coeff': 39, 'attr': 'real'},

                              {'coeff': 40, 'attr': 'real'},

                              {'coeff': 41, 'attr': 'real'},

                              {'coeff': 42, 'attr': 'real'},

                              {'coeff': 43, 'attr': 'real'},

                              {'coeff': 44, 'attr': 'real'},

                              {'coeff': 45, 'attr': 'real'},

                              {'coeff': 46, 'attr': 'real'},

                              {'coeff': 47, 'attr': 'real'},

                              {'coeff': 48, 'attr': 'real'},

                              {'coeff': 49, 'attr': 'real'},

                              {'coeff': 50, 'attr': 'real'},

                              {'coeff': 51, 'attr': 'real'},

                              {'coeff': 52, 'attr': 'real'},

                              {'coeff': 53, 'attr': 'real'},

                              {'coeff': 54, 'attr': 'real'},

                              {'coeff': 55, 'attr': 'real'},

                              {'coeff': 56, 'attr': 'real'},

                              {'coeff': 57, 'attr': 'real'},

                              {'coeff': 58, 'attr': 'real'},

                              {'coeff': 59, 'attr': 'real'},

                              {'coeff': 60, 'attr': 'real'},

                              {'coeff': 61, 'attr': 'real'},

                              {'coeff': 62, 'attr': 'real'},

                              {'coeff': 63, 'attr': 'real'},

                              {'coeff': 64, 'attr': 'real'}],

          

         }
extracted_features = tsfresh.extract_features(extract_from,

                                              column_id='series_id',

                                              column_sort='measurement_number',

                                              n_jobs = 2, 

                                              default_fc_parameters = params)
efc = extracted_features.copy()

tsfresh.utilities.dataframe_functions.impute(efc)

y = all_stitched.groupby('series_id').max()['surface']

matched_y = y[y!='no_surface']

matched_ef = efc.iloc[matched_y.index]
features_filtered = tsfresh.select_features(matched_ef, matched_y)
feature_columns = features_filtered.columns
metadata = all_stitched.groupby('series_id').max()[['surface','test','new_group_id', 'orig_id']]
filtered_extracted = metadata.join(extracted_features[feature_columns])
train_filtered = filtered_extracted[filtered_extracted.test == 0]

test_filtered = filtered_extracted[filtered_extracted.test == 1]
X_filtered = train_filtered.drop(labels = ['surface','test','new_group_id', 'orig_id'], 

                                 axis = 1)

y_filtered = train_filtered.surface



X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_filtered, y_filtered, test_size=0.25, random_state=42, stratify = y_filtered)
scaler_f = StandardScaler()

X_f_train_scaled = scaler_f.fit_transform(X_f_train)

X_f_test_scaled = scaler_f.transform(X_f_test)
RFC = RandomForestClassifier(n_estimators=50, random_state = 42, n_jobs=-1)

RFC.fit(X_f_train, y_f_train)

y_f_test_RFC = RFC.predict(X_f_test)

print(classification_report(y_f_test, y_f_test_RFC))
NB = GaussianNB()

NB.fit(X_f_train_scaled, y_f_train)

y_test_NB = NB.predict(X_f_test_scaled)

print(classification_report(y_f_test, y_test_NB))
C_range = np.logspace(-4, 0, 4)

param_grid = dict(C=C_range)

SVM = GridSearchCV(SVC(kernel = 'linear'), 

                    param_grid=param_grid, 

                    n_jobs=-1, 

                    pre_dispatch=2, 

                    cv = 3)

SVM.fit(X_f_train_scaled, y_f_train)

y_test_svc = SVM.predict(X_f_test_scaled)

print(classification_report(y_f_test, y_test_svc))
KNN = KNeighborsClassifier(10)

KNN.fit(X_f_train_scaled,y_f_train)

y_test_knn = KNN.predict(X_f_test_scaled)

print(classification_report(y_f_test, y_test_knn))
DT = DecisionTreeClassifier()

DT.fit(X_f_train_scaled,y_f_train)

y_test_dt = DT.predict(X_f_test_scaled)

print(classification_report(y_f_test, y_test_dt))
param_grid = {

    'max_depth': [3,4,5,6],  # the maximum depth of each tree

    'min_child_weight':np.linspace(0.8,1.2,4),

    'gamma': np.linspace(0,0.2,4),

}
# The below parameters were found using GridSearchCV, using the parameter search basis above.

XGB = xgb.sklearn.XGBClassifier(learning_rate = 0.025,

                                objective = 'multi:softmax',

                                n_estimators = 150,

                                max_depth = 5,

                                min_child_weight = 1.2,

                                subsample=0.8,

                                colsample_bytree = 0.8,

                                gamma = 0.066,

                                n_jobs = 4,

                                nthreads = 1,

                                silent = True,

                                seed = 42)

XGB.fit(X_f_train_scaled,y_f_train)

y_test_XGB = XGB.predict(X_f_test_scaled)

print(classification_report(y_f_test, y_test_svc))
def get_predictions(filtered_data, classifiers, scaler):

    warnings.filterwarnings("ignore")

    

    df = filtered_data[['surface','test','new_group_id', 'orig_id']]

    X = filtered_data.drop(labels = ['surface','test','new_group_id','orig_id'], 

                                 axis = 1).values

    X_scaled = scaler.transform(X)

    for name, classifier_dict in classifiers.items():

        clf = classifier_dict['classifier']

        if classifier_dict['scale']:

            prediction = clf.predict(X_scaled)

        else:

            prediction = clf.predict(X)

        df[name] = prediction

    warnings.resetwarnings()

    return df
classifiers = {'RFC':{'classifier':RFC,'scale': False},

               'NB':{'classifier':NB,'scale':True},

               'SVM':{'classifier':SVM,'scale':True},

               'KNN':{'classifier':KNN,'scale':True},

               'DT':{'classifier':DT,'scale':True},

               'XGB':{'classifier':XGB,'scale':True}}
multi_predictions = get_predictions(filtered_data=train_filtered,

                              classifiers=classifiers, 

                              scaler=scaler_f)
def classifier_ensemble(predictions, classifier_weights, groupby = 'new_group_id'):

    warnings.filterwarnings("ignore")

    df = predictions[['test','new_group_id','orig_id']]

    df['surface'] = 'None'

    g = predictions.groupby(groupby)

    for name, group in g:

        index = group.index

        weighted_value_counts = []

        for classifier,weight in classifier_weights.items():

            value_counts = group[classifier].value_counts()

            weighted_value_counts.append(value_counts*weight)

        all_counts = pd.DataFrame(weighted_value_counts).fillna(value=0)

        surface = all_counts.sum().sort_values(ascending=False).index[0]

        df.loc[index,'surface'] = surface

    warnings.resetwarnings()

    return df
classifier_weights = {classifier:1 for classifier in classifiers.keys()}

del classifier_weights['NB'] # Naive Bayes classification wasn't great, so I won't include it

ensemble_prediction = classifier_ensemble(predictions=multi_predictions,

                                 classifier_weights=classifier_weights)
print(classification_report(train_filtered.surface, ensemble_prediction.surface))
def CEParamSearch(y_true, multi_predictions, classifier_names, 

                  groupby = 'new_group_id', generations = 20, 

                  seed = 1, mutation_rate = 0.35):  

    # Generate random weights first

    np.random.seed(seed)

    weights = np.random.rand(len(classifiers))

    classifier_weights = {classifier:weight for classifier,weight in zip(classifier_names,weights)}

    

    # Classify the data and calculate a score with the weights

    ensemble_prediction = classifier_ensemble(multi_predictions,classifier_weights,groupby=groupby)

    y_pred = ensemble_prediction.surface

    score = sum(y_pred == y_true)/len(y_pred)

    

    # Keep track of score and time

    scores = [score]

    start = time.time()

    

    

    # Hunt for a better solution

    gene = random.choice(classifier_names)

    orig_mut_rate = mutation_rate

    for generation in range(1,generations+1):

        # Make new weights using small mutation

        new_weights = classifier_weights

        mutation = (np.random.rand(1)-0.5)*mutation_rate

        

        new_weights[gene] += mutation

        #Constrain weights to [0,1]

        if new_weights[gene] < 0:

            new_weights[gene] = 0

        if new_weights[gene] > 1:

            new_weights[gene] = 1

            

        # Classify data and obtain new score

        ensemble_prediction = classifier_ensemble(multi_predictions,new_weights,groupby=groupby)

        y_pred = ensemble_prediction.surface

        new_score = sum(y_pred == y_true)/len(y_pred)

        change = new_score - score

        # Select fittest weights

        if change < 0: # If new score is worse change the gene, reestablish mutation rate

            scores.append(score)

            gene = random.choice(classifier_names)

            mutation_rate = orig_mut_rate

        elif change == 0: # Mutate, but don't change gene or the mutation rate

            scores.append(score)

            classifier_weights = new_weights

        elif change > 0: # Mutate and change mutation rate, but don't change the gene

            mutation_rate /= 0.75

            score = new_score

            scores.append(score)

            classifier_weights = new_weights

        

        # Update User

        progress = generation/generations

        time_elapsed = (time.time()-start)

        ETA = (time_elapsed/progress-time_elapsed)/60

        print('Score:{0:1.5f}\t|\tChange:{1:1.5f}\t|\tTime Remaining: {2:1.2f} min'.format(score,change,ETA), end = '\r')

    

    plt.plot(scores)

    return classifier_weights
classifier_names = list(classifier_weights.keys())
classifier_weights = CEParamSearch(train_filtered.surface,

                                   multi_predictions,

                                   classifier_names, 

                                   generations = 25,

                                   mutation_rate = 0.5,

                                   seed = 40)
classifier_weights = {'RFC': 1,

 'SVM': 1,

 'KNN': 0,

 'DT': 0.22006725,

 'XGB': 0.38377276}
multi_predictions_test = get_predictions(test_filtered,classifiers,scaler_f)
ensemble_prediction_test = classifier_ensemble(predictions=multi_predictions_test,

                                 classifier_weights=classifier_weights)
output = ensemble_prediction_test[['orig_id','surface']].reset_index().drop('series_id', axis =1)

output.to_csv(path_or_buf = 'Ensemble_Classified.csv', 

              header = ['series_id','surface'], 

              index = False)
metadata = sorted_data.groupby('series_id').max()[['surface','test','new_group_id', 'orig_id']]

filtered_extracted_matched = metadata.join(extracted_features[feature_columns])

unmatched_filtered = filtered_extracted_matched[filtered_extracted_matched.surface == 'no_surface']
multi_predictions_unmatched = get_predictions(unmatched_filtered, classifiers, scaler_f)

ensemble_prediction_unmatched = classifier_ensemble(multi_predictions_unmatched, classifier_weights)

unmatched_output = ensemble_prediction_unmatched[['orig_id','surface']].reset_index().drop('series_id', axis =1)
matched_output = metadata[(metadata.test == 1)&(metadata.surface != 'no_surface')][['orig_id','surface']].reset_index().drop('series_id', axis =1)
os.chdir("/kaggle/working/")
output = pd.concat([matched_output,unmatched_output])

output.to_csv('Hybrid_Classified.csv', 

              header = ['series_id','surface'], 

              index = False)