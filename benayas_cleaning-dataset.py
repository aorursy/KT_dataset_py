import pandas as pd

import numpy as np

from ast import literal_eval

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import sklearn as sk

from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_squared_error

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from functools import partial

import category_encoders as ce

import scipy as sp

from scipy.stats import ks_2samp, ttest_ind, kstest

import os

import json as json

from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, anneal

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")



pd.set_option('display.max_colwidth', 200)

pd.set_option('display.max_columns', None)

pd.set_option('display.min_rows', 100)

pd.set_option('display.max_rows', 200)



kaggle = True

generate_or_load = 'generate'

dirname = '/kaggle/input/data-science-bowl-2019/' if kaggle == True else ''
def read_dics_event(df, fields):

    event_df = read_dics(df)   

    keep = [x for x in fields if x in event_df.columns]

    event_df = event_df[keep]

    return pd.concat([df.reset_index(drop=True), event_df.reset_index(drop=True)], axis=1)



def read_dics(df):

    df['event_data'] = df['event_data'].str.replace('false', 'False')

    df['event_data'] = df['event_data'].str.replace('true', 'True')

    return pd.DataFrame( [ eval(x) for x in df['event_data'] ] )



def get_accuracy_group(completed, errors=0):

    if completed == False:

        return 0

    if errors == 0:

        return 3

    if errors == 1:

        return 2

    return 1



def unfold(df):

    return pd.concat([df.drop(['event_data','event_count','event_code','game_time'], axis=1).reset_index(drop=True), read_dics(df).reset_index(drop=True)], axis=1)



def get_categorical_index(df, cat_cols):

    cat_features_index = np.where(df.columns.isin(cat_cols))[0].tolist()

    return cat_features_index



def pca_num(pca, required=99.0):

    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    n = 0

    for i in range(0,len(var)):

        if var[i] >= 99:

            n = i

            break

    return n+1



def remove_correlation(df, threshold = 0.99, verbose=False):

    columns = df.columns.copy()

    counter = 0

    to_remove = []

    for a in tqdm(range(0,len(columns))):

        for b in range(a,len(columns)):

            if a == b:

                if df[columns[b]].std() == 0:

                    counter += 1

                    to_remove.append(columns[b])

                    if verbose==True:

                        print('{}: {} vs {} : Correlation= {}'.format(counter, columns[a], columns[b], 'Constant'))

                continue

            if columns[a] not in to_remove and columns[b] not in to_remove:

                c = np.corrcoef(df[columns[a]], df[columns[b]])[0][1]

                if c > threshold:

                    counter += 1

                    to_remove.append(columns[b])

                    if verbose==True:

                        print('{}: {} vs {} : Correlation= {}'.format(counter, columns[a], columns[b], c))



    df = df.drop(columns=to_remove)

    print("New shape after remove correlations {}".format(df.shape))

    return df



def expand_features(X, categorical):

    if not isinstance(X, pd.DataFrame):

        df = pd.DataFrame(data=X, index=[ i for i in range(0,X.shape[0])], columns=[ 'C'+str(i) for i in range(0,X.shape[1])] )

    else:

        df = X

    cols = df.columns.copy()

    for a in tqdm(cols):

        for b in cols:

            if a == b:

                continue

            if ((a in categorical) or (b in categorical)):

                continue

            df[a+'_'+b+'_1'] = (df[a] - df[b]) / df[b]

            df[a+'_'+b+'_2'] = (df[a] + df[b]) / df[b]



    print("New shape after feature expansion {}".format(df.shape))

    return df



def remove_different_features(df, test_start, categorical=[], threshold=0.5, verbose=False):

    columns = df.columns

    df_train = df[:test_start]

    df_test = df[test_start:]

    counter = 0

    to_remove = []

    for c in columns:

        if c in categorical:

            continue

        if np.issubdtype(df[c].dtype, np.number):

            o = overlapping(df_train[c], df_test[c], n_bins=1000)

            if o < threshold: # the samples are not from the same distribution

                if verbose == True:

                    print("Feature {} overlaps in : {}".format(c,o))

                to_remove.append(c)

    

    df = df.drop(columns=to_remove)

    print("New shape after removing different distribution features {}".format(df.shape))

    return df



def overlapping(xtr, xts, n_bins = 100):

    xtr = xtr.sort_values()

    xts = xts.sort_values()

    step = (max(xtr.max(), xts.max()) - min(xtr.min(), xts.min())) / n_bins

    if step == 0.0:

        return 1.0

    bins_tr = [0 for i in range(0,n_bins)]

    bins_ts = [0 for i in range(0,n_bins)]

    for x in xtr:

        index = min(int(np.floor(x/step)),n_bins-1)

        bins_tr[index] = bins_tr[index] + 1

    for x in xts:

        index = min(int(np.floor(x/step)),n_bins-1)

        bins_ts[index] = bins_ts[index] + 1

    bins_tr = np.array(bins_tr) / len(xtr)

    bins_ts = np.array(bins_ts) / len(xts)

    overlapping = 0

    for i in range(0,len(bins_tr)):

        overlapping = overlapping + min(bins_tr[i], bins_ts[i])

    return overlapping



def divide_data_for_models(train, test, multikid_ids):

    train_multikid = train[train['installation_id'].isin(multikid_ids)]

    train = train[~train['installation_id'].isin(multikid_ids)]

    test_multikid = test[test['installation_id'].isin(multikid_ids)]

    test = test[~test['installation_id'].isin(multikid_ids)]

    

    print("Shape of train set {}".format(train.shape))

    print("Shape of test set {}".format(test.shape))

    

    print("Shape of train multikid set {}".format(train_multikid.shape))

    print("Shape of test multikid set {}".format(test_multikid.shape))

    

    return train, test, train_multikid, test_multikid



def divide_data_for_models2(df):

    #df = df_all.drop('set', axis=1)

    train_multikid = df[df['cluster']==0]

    train = df[df['cluster']==1]

    test_multikid = df[df['cluster']==2]

    test = df[df['cluster']==3]

    

    print("Shape of train set {}".format(train.shape))

    print("Shape of test set {}".format(test.shape))

    

    print("Shape of train multikid set {}".format(train_multikid.shape))

    print("Shape of test multikid set {}".format(test_multikid.shape))

    

    return train, test, train_multikid, test_multikid
# Load

raw_train = pd.read_csv(os.path.join(dirname, 'train.csv'), dtype={"event_code": np.int16, "event_count": np.int16, "game_time": np.int32})

raw_test = pd.read_csv(os.path.join(dirname, 'test.csv'), dtype={"event_code": np.int16, "event_count": np.int16, "game_time": np.int32})

specs = pd.read_csv(os.path.join(dirname, 'specs.csv'))

specs_list = specs['event_id']



# Remove installations without assessments

raw_train = raw_train[raw_train['installation_id'].isin( raw_train[raw_train['type']=='Assessment']['installation_id'])]

raw_train['timestamp'] = pd.to_datetime(raw_train['timestamp'])

raw_test['timestamp'] = pd.to_datetime(raw_test['timestamp'])



# Convert certain variables to categoricals

raw_train.head()
raw_train.dtypes
print('Raw train data contains {0} rows'.format(len(raw_train)))

print('Raw test data contains {0} rows'.format(len(raw_test)))
# Function to prepare data

def prepare_data(df, specs, is_test=False):

    fields_event_data = ['exit_type', 'level', 'round', 'correct']

    assessments = df[df['type']=='Assessment']['title'].unique()

    types = df['type'].unique() # Game, assessment, clip , activity

    worlds = df['world'].unique()

    titles = df['title'].unique()

    extracted_data = []

    cumulative_all = {}

    df = df.sort_values(by=['installation_id','game_session','timestamp'])



    # For each game_session

    for i, g_installation_id in tqdm(df.groupby('installation_id')):

        

        g_installation_id = read_dics_event(g_installation_id, fields_event_data)

        if 'level' in g_installation_id:

            g_installation_id['level'] = g_installation_id['level'].fillna(0).astype(int)

        

        if 'round' in g_installation_id:

            g_installation_id['round'] = g_installation_id['round'].fillna(0).astype(int)

        

         # Cumulative dict initialization

        cumulative = {'total_cum_sessions':int(0),

                      'total_cum_events_count':int(0), 

                      'total_cum_events_id':dict(),                     

                      'total_cum_events_code':dict(), 

                      'durations':[]}

        for x in assessments:

            cumulative['incorrect_'+str(x)] = 0

            cumulative['correct_'+str(x)] = 0

        for x in types:

            cumulative[x] = 0

            cumulative['durations_'+x] = []

        for x in worlds:

            cumulative[x] = 0

            cumulative['durations_'+x] = []

            cumulative[x+'_max_level'] = 0

            cumulative[x+'_max_round'] = 0

        for x in titles:

            cumulative[x] = 0

            cumulative['durations_'+x] = []

            

        cumulative['exit_type'] = 0    

        cumulative['max_level'] = 0 

        cumulative['max_round'] = 0 

        



            

        # Sort by timestamp so for testing the last row is the one to be predicted

        g_installation_id = g_installation_id.sort_values(by='timestamp')

        

        # Separe the last row if this is to build a test dataset

        if is_test == True:

            to_predict = g_installation_id[-1:]

            g_installation_id = g_installation_id[:-1]     

        

        # Process each game_session

        for s, df_unfold in g_installation_id.groupby('game_session'):

            # Unfold game_data

            #df_unfold = unfold(df_unfold)



            features = process_session(i, s, df_unfold.sort_values(by='timestamp'), cumulative, types, worlds, titles, False)

            if features:

                extracted_data.append(features)

                

        if is_test == True:

            extracted_data.append(process_session(i, to_predict.iloc[0]['game_session'], to_predict, cumulative, types, worlds, titles, is_test))

            

    # Create dataframe from the extracted data

    df_all = pd.DataFrame(extracted_data).fillna(0)

        

    # Complete the missing columns from specs

    '''for c in specs:

        if not c in df_all.columns:

            df_all[c] = 0

        df_all[c] = df_all[c].astype(np.int16)'''

        

    # Sort columns

    df_all = df_all.reindex(sorted(df_all.columns), axis=1)

        

    df_all['accuracy_group'] = df_all['accuracy_group'].astype(np.int16)

    df_all.columns = [ x.replace('!','').replace(' ','_').replace(',','') for x in df_all.columns]

    return df_all

            

def process_session(installation_id, game_session, df, cumulative, types, worlds, titles, is_test=False):

    # If not an assessment, nothing to be returned, just cumulate

    if df.iloc[0]['type'] != 'Assessment':

        cumulate(df,cumulative)

        return

    

    # Features is the dict to be returned

    features = {}

    # Basic ID features

    features['installation_id'] = installation_id

    features['game_session'] = game_session

    

    # Session data (Categorical)

    features['title'] = df.iloc[0]['title']

    features['world'] = df.iloc[0]['world']

    features['hour'] = df.iloc[-1]['timestamp'].hour

    features['weekday'] = df.iloc[-1]['timestamp'].weekday()

    features['month'] = df.iloc[-1]['timestamp'].month

    

    # Game Session cumulatives

    features['total_cum_sessions'] = cumulative['total_cum_sessions']

    features['log_total_cum_sessions'] = np.log(1+cumulative['total_cum_sessions'])

    features['total_cum_sessions_world'] = cumulative[features['world']]

    features['total_cum_sessions_title'] = cumulative[features['title']]

    

    # Events cumulatives

    #features.update(cumulative['total_cum_events_id'])

    features.update(cumulative['total_cum_events_code'])

    features['total_cum_events_count'] = cumulative['total_cum_events_count']

    features['log_total_cum_events_counts'] = np.log(1+cumulative['total_cum_events_count'])

    features['distinct_event_codes'] = len(np.unique(list(cumulative['total_cum_events_code'].keys())))

      

    # Game time duration cumulatives

    features['log_total_cum_time'] = np.log(1+np.array(cumulative['durations']).sum())

    features['total_cum_time'] = np.array(cumulative['durations']).sum()

    features['total_avg_time'] = np.array(cumulative['durations']).mean()

    features['total_std_time'] = np.array(cumulative['durations']).std()

    features['total_cum_time_world'] = np.array(cumulative['durations_'+features['world']]).sum()

    features['total_avg_time_world'] = np.array(cumulative['durations_'+features['world']]).mean()

    features['total_std_time_world'] = np.array(cumulative['durations_'+features['world']]).std()

    for x in types:

        features[x] = cumulative[x]

        features['log_total_cum_time_'+x] = np.log(1+np.array(cumulative['durations_'+x]).sum())

        features['total_cum_time_'+x] = np.array(cumulative['durations_'+x]).sum()

        features['total_avg_time_'+x] = np.array(cumulative['durations_'+x]).mean()

        features['total_std_time_'+x] = np.array(cumulative['durations_'+x]).std()

    for x in worlds:

        features[x] = cumulative[x]

        features['log_total_cum_time_'+x] = np.log(1+np.array(cumulative['durations_'+x]).sum())

        features['total_cum_time_'+x] = np.array(cumulative['durations_'+x]).sum()

        features['total_avg_time_'+x] = np.array(cumulative['durations_'+x]).mean()

        features['total_std_time_'+x] = np.array(cumulative['durations_'+x]).std()

        features[x+'_max_level'] = cumulative[x+'_max_level']

        features[x+'_max_round'] = cumulative[x+'_max_round']

    for x in titles:

        features[x] = cumulative[x]

        features['log_total_cum_time_'+x] = np.log(1+np.array(cumulative['durations_'+x]).sum())

        features['total_cum_time_'+x] = np.array(cumulative['durations_'+x]).sum()

        features['total_avg_time_'+x] = np.array(cumulative['durations_'+x]).mean()

        features['total_std_time_'+x] = np.array(cumulative['durations_'+x]).std()

        

    features['exit_type'] = cumulative['exit_type']

    features['max_level'] = cumulative['max_level']

    features['max_round'] = cumulative['max_round']



        

    # Cumulate

    cumulate(df,cumulative)

    

    

    # If this is a sample to be predicted

    if (is_test == True) & (len(df)==1) & (df.iloc[0]['event_code']==2000):

        features['prev_errors'] = cumulate_errors(cumulative, df.iloc[0]['title'], 0)

        features['prev_correct'] = cumulate_correct(cumulative, df.iloc[0]['title'], 0)

        features['accuracy_group'] = -1 # this is a flag

        return features

        

    # Calculate label

    df_assessment = df[ ((df['event_code']==4100)&(~df['title'].str.startswith('Bird'))) | ((df['event_code']==4110)&(df['title'].str.startswith('Bird'))) ]

    if len(df_assessment) == 0:

        return

    for t, g_title in df_assessment.groupby('title'):

        if True in g_title['correct'].unique():

            correct_attempt = g_title[g_title['correct'] == True].iloc[0]

            errors = len(g_title.loc[:correct_attempt.name]) -1

            features['prev_errors'] = cumulate_errors(cumulative, t, errors)

            features['prev_correct'] = cumulate_correct(cumulative, t, 1)

            features['accuracy_group'] = get_accuracy_group(True, errors)

        else:

            features['prev_errors'] = cumulate_errors(cumulative, t, len(g_title)) 

            features['prev_correct'] = cumulate_correct(cumulative, t, 0)

            features['accuracy_group'] = get_accuracy_group(False)  

    return features



# Cumulate data for the same installation id        

def cumulate(df,cumulative):

    

    # Type of game session

    type_session = df.iloc[-1]['type']

    cumulative[type_session] = cumulative[type_session] + 1

    

    # World of game session

    world_session = df.iloc[-1]['world']

    cumulative[world_session] = cumulative[world_session] + 1

    

    # Title of game session

    title_session = df.iloc[-1]['title']

    cumulative[title_session] = cumulative[title_session] + 1

    

    # Game time duration

    if len(df) > 1:

        duration = df.iloc[-1]['game_time']/1000

        cumulative['durations'] = cumulative['durations'] + [duration]

        cumulative['durations_'+type_session] = cumulative['durations_'+type_session] + [duration]

        cumulative['durations_'+world_session] = cumulative['durations_'+world_session] + [duration]

        cumulative['durations_'+title_session] = cumulative['durations_'+title_session] + [duration]

    else:

        cumulative['durations'] = cumulative['durations'] + [0.0]



    # Game sessions

    cumulative['total_cum_sessions'] = cumulative['total_cum_sessions'] + 1

    

    # Event count

    cumulative['total_cum_events_count'] = cumulative['total_cum_events_count'] + df.iloc[-1]['event_count']

    

    # Events ID

    '''events = cumulative['total_cum_events_id']

    for x in df['event_id']:

        if x in events:

            events[x] = events[x] + 1

        else:

            events[x] = 1

    cumulative['total_cum_events_id'] = events'''

    

    # Events code

    events_code = cumulative['total_cum_events_code']

    for e in df['event_code']:

        x = str(e)

        if x in events_code:

            events_code[x] = events_code[x] + 1

        else:

            events_code[x] = 1

    cumulative['total_cum_events_code'] = events_code

    

    # Exit type correct

    if 'exit_type' in df.columns:

        cumulative['exit_type'] = cumulative['exit_type'] + df['exit_type'].count()

     

    # Level

    if 'level' in df.columns:

        cumulative['max_level'] = max(cumulative['max_level'], max(df['level']))

        cumulative[world_session+'_max_level'] =  max(cumulative[world_session+'_max_level'], df['level'].max())

    

    # Round

    if 'round' in df.columns:

        cumulative['max_round'] = max(cumulative['max_round'], max(df['round']))

        cumulative[world_session+'_max_round'] =  max(cumulative[world_session+'_max_round'], df['round'].max())





# Cumulate the number of errors for each assessment type

def cumulate_errors(cumulative,title,errors):

    t = 'incorrect_'+title

    previous = cumulative[t]

    cumulative[t] = previous + errors

    return previous



# Cumulate the number of errors for each assessment type

def cumulate_correct(cumulative,title,correct):

    t = 'correct_'+title

    previous = cumulative[t]

    cumulative[t] = previous + correct

    return previous

# Train Data

if (generate_or_load == 'generate') | (kaggle == True):

    train = prepare_data(raw_train, specs_list)

    if kaggle == False:

        train.to_csv('new_train.csv',index=False)

        print('Saved to {0}'.format('new_train.csv'))        

else:

    train = pd.read_csv('new_train.csv')

    print('Data loaded from to {0}'.format('new_train.csv'))

print("Train's shape is {0}".format(train.shape))
# Consistency check

temp_df = train[['installation_id','game_session','accuracy_group']]

merged = temp_df.merge(pd.read_csv(os.path.join(dirname, 'train_labels.csv')), on=['installation_id','game_session'], how='left')

merged['OK'] = merged['accuracy_group_x'] == merged['accuracy_group_y']

print('Generated values not in train_label {0}'.format(str(len(merged[pd.isnull(merged['accuracy_group_y'])]))))

limpios = merged.dropna(subset=['accuracy_group_y'], axis=0)

print('Generated values with different accuracy_group {0}'.format(str(len(limpios[limpios['OK']==False]))))
# Test Data

if (generate_or_load == 'generate') | (kaggle == True):

    test = prepare_data(raw_test, specs_list, is_test=True)

    if kaggle == False:

        test.to_csv('new_test.csv',index=False)

        print('Saved to {0}'.format('new_test.csv'))       

else:

    test = pd.read_csv('new_test.csv')

    print('Data loaded from to {0}'.format('new_test.csv'))

print("Test's shape is {0}".format(test.shape))
assert (list(train.columns) == list(test.columns))

categorical = ['hour', 'month', 'weekday','world','title']

useless = ['installation_id', 'game_session']
# From the X_test dataset just 1000 are to be predicted

# The rest will join the X_train dataset

train = pd.concat([train,test[test['accuracy_group']!=-1]], ignore_index=True)

test = test[test['accuracy_group']==-1]

test = test.drop('accuracy_group',axis=1)

print("Train's shape is {0}".format(train.shape))

print("Test's shape is {0}".format(test.shape))
# Encode certain features



# concatenate train and test data

temp_df = pd.concat([train, test])

# encode

encoder = ce.ordinal.OrdinalEncoder(cols = ['world','title'])

temp_df = encoder.fit_transform(temp_df)

# dataset

train = temp_df.iloc[:len(train),:]

test = temp_df.iloc[len(train):,:]

train.head()
# Divide the data to try to find installation id's with multiple kids

cum_time_mean = train['total_cum_time'].mean()

cum_time_std = train['total_cum_time'].std()

cum_sessions_mean = train['total_cum_sessions'].mean()

cum_sessions_std = train['total_cum_sessions'].std()



mask_train = ((train['total_cum_time']>=cum_time_mean+1*cum_time_std)|(train['total_cum_sessions']>=cum_sessions_mean+1*cum_sessions_std))

mask_test = ((test['total_cum_time']>=cum_time_mean+1*cum_time_std)|(test['total_cum_sessions']>=cum_sessions_mean+1*cum_sessions_std))



df_0 = train[mask_train]

df_0['cluster'] = 0

df_1 = train[~mask_train]

df_1['cluster'] = 1

df_2 = test[mask_test]

df_2['cluster'] = 2

df_3 = test[~mask_test]

df_3['cluster'] = 3

df_mask = pd.concat([df_0,df_1,df_2,df_3], ignore_index=True)

fig, ax = plt.subplots(figsize=(20,10))

ax = sns.scatterplot(x="total_cum_time", y="total_cum_sessions", hue="cluster", palette=sns.color_palette("husl", 4), data=df_mask)

print(df_mask['cluster'].value_counts())
# Group the data by installation id, and keeping the last sample of each

df_installations = pd.concat([train, test]).groupby('installation_id').max()

df_installations = df_installations.reset_index()

fig, ax = plt.subplots(figsize=(20,10))

df_installations['cluster'] = df_installations.apply(lambda x: 'outsiders' if ((x['total_cum_time']>=cum_time_mean+2*cum_time_std)|(x['total_cum_sessions']>=cum_sessions_mean+2*cum_sessions_std)) else 'normal', axis=1 )

ax = sns.scatterplot(x="total_cum_time", y="total_cum_sessions", hue="cluster", data=df_installations)
df_0 = train

df_0['set'] = 'train'

df_1 = test

df_1['set'] = 'test'



df_all = pd.concat([df_0, df_1])

df_all = df_all.merge(df_installations[['installation_id','cluster']], on='installation_id', how='left')

fig, ax = plt.subplots(figsize=(20,10))

ax = sns.scatterplot(x="total_cum_time", y="total_cum_sessions", hue="cluster",style="set", data=df_all)

print(df_all['cluster'].value_counts())
# This set we have to keep

df_installation_test = df_all[(df_all['cluster']=='outsiders')&(df_all['installation_id'].isin(test['installation_id']))]

fig, ax = plt.subplots(figsize=(20,10))

ax = sns.scatterplot(x="total_cum_time", y="total_cum_sessions", hue="set",style="cluster", data=df_installation_test)

print(df_all['cluster'].value_counts())
df_remove = df_all[(df_all['cluster']=='outsiders')&(~df_all['installation_id'].isin(test['installation_id']) )] #Installations ID not in test

fig, ax = plt.subplots(figsize=(20,10))

ax = sns.scatterplot(x="total_cum_time", y="total_cum_sessions", data=df_remove)

print(df_remove['cluster'].value_counts())
df_mask = df_mask[(~df_mask['installation_id'].isin(df_remove['installation_id']) )]

df_mask.describe()
fig, ax = plt.subplots(figsize=(20,10))

ax = sns.scatterplot(x="total_cum_time", y="total_cum_sessions", hue="cluster", palette=sns.color_palette("husl", 4), data=df_mask)

print(df_mask['cluster'].value_counts())
def feature_pipeline(train, test, categorical):

    temp_df = pd.concat([train.drop('accuracy_group',axis=1), test.drop('accuracy_group',axis=1)])

    temp_df = feature_reduction(temp_df, categorical)

    

    # Scaling numerical features

    '''scaler = MinMaxScaler()

    mask = [c for c in temp_df.columns if c not in categorical]

    scaler.fit(temp_df[mask])

    temp_df[mask] = scaler.transform(temp_df[mask])

    

    # Features expansion       

    temp_df = expand_features(temp_df, categorical)'''

    

    # Divide into train and test again

    X_train = temp_df[:len(train)]

    X_test = temp_df[len(train):]

    print("Train's shape is {0}".format(X_train.shape))

    print("Test's shape is {0}".format(X_test.shape))

    return X_train, X_test



def feature_reduction(df, categorical):

    df = remove_different_features(df, len(train), categorical = categorical, threshold=0.5)

    df = remove_correlation(df)

    return df
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

from sklearn import preprocessing

%matplotlib inline



# Divide datasets

train_normal, test_normal, train_multikid, test_multikid = divide_data_for_models2(df_mask)



print('Train and Test set (normal)')

y_train = train_normal['accuracy_group']

X_train, X_test = feature_pipeline(train_normal.drop(useless, axis=1), test_normal.drop(useless, axis=1), categorical)



print('Train and Test set (multikid)')

y_train_multikid = train_multikid['accuracy_group']

X_train_multikid, X_test_multikid = feature_pipeline(train_multikid.drop(useless, axis=1), test_multikid.drop(useless, axis=1), categorical)

import catboost as cb

import xgboost as xgb

import lightgbm as lgb
best_n_cb = 1747

best_cb = {'bagging_temperature': 0.7907291047410311,

 'border_count': 35,

 'depth': 6,

 'l2_leaf_reg': 6.697930677527237,

 'learning_rate': 0.01586403145995819,

 'random_state': 1,

 'random_strength': 2.76835390842624,

 'loss_function': 'RMSE',

 'one_hot_max_size': 31,

 'eval_metric': 'RMSE',

 'od_type': 'Iter',

 'od_wait': 20}



best_n_cb_mk = 77

best_cb_mk = {'bagging_temperature': 0.5640162442730875,

 'border_count': 60,

 'depth': 4,

 'l2_leaf_reg': 19.707986433630612,

 'learning_rate': 0.1286788580745031,

 'random_state': 2,

 'random_strength': 1.542224177415203,

 'loss_function': 'RMSE',

 'one_hot_max_size': 31,

 'eval_metric': 'RMSE',

 'od_type': 'Iter',

 'od_wait': 20}
best_xgb = {'alpha': 18.659799567977437,

 'bagging_temperature': 0.6414551219335474,

 'colsample_bytree': 0.5098810850299129,

 'gamma': 0.1,

 'lambda': 12.181369439828163,

 'learning_rate': 0.02025268651064681,

 'max_depth': 6,

 'min_child_weight': 8.0,

 'seed': 0,

 'subsample': 0.7993323980076127,

 'loss_function': 'rmse',

 'eval_metric': 'rmse'}

best_n_xgb = 614



best_xgb_mk = {'alpha': 11.433288052255918,

 'bagging_temperature': 0.33942590071593925,

 'colsample_bytree': 0.9979068963608676,

 'gamma': 0.30000000000000004,

 'lambda': 2.668300231808874,

 'learning_rate': 0.028968044125941405,

 'max_depth': 2,

 'min_child_weight': 6.0,

 'seed': 0,

 'subsample': 0.6144708440137876,

 'loss_function': 'rmse',

 'eval_metric': 'rmse'}

best_n_xgb_mk = 211
best_n_lgb = 1338

best_lgb = {'bagging_temperature': 0.9611498217282982,

 'colsample_bytree': 0.6771533510903894,

 'gamma': 0.30000000000000004,

 'lambda_l1': 9.189847609240765,

 'lambda_l2': 2.728003044833198,

 'learning_rate': 0.010244282681882995,

 'max_depth': 4,

 'min_child_weight': 2.0,

 'num_leaves': 2,

 'seed': 1,

 'subsample': 0.3414568373784109,

 'loss_function': 'rmse',

 'eval_metric': 'rmse'}



best_n_lgb_mk = 1107

best_lgb_mk = {'bagging_temperature': 0.7960888097194333,

 'colsample_bytree': 0.6813552383711141,

 'gamma': 0.4,

 'lambda_l1': 11.856321953925471,

 'lambda_l2': 9.609965355663928,

 'learning_rate': 0.010167529279903848,

 'max_depth': 10,

 'min_child_weight': 8.0,

 'num_leaves': 3,

 'seed': 3,

 'subsample': 0.20000221570501625,

 'loss_function': 'rmse',

 'eval_metric': 'rmse'}
class OptRounder(object):

    def __init__(self):

        self.res_ = []

        self.coef_ = []

        

    def get_res(self):

        return self.res_

    

    # objective function

    def func(self, coef, X, y):

        kappa = cohen_kappa_score(self.bincut(coef, X), y, weights='quadratic')

        return -kappa



    def bincut(self, coef, X):

        return pd.cut(X,

                      [-np.inf] + list(np.sort(coef)) + [np.inf],

                      labels = [0, 1, 2, 3])

        

    def fit(self, X, y):

        pfunc = partial(self.func, X=X, y=y)

        self.res_ = sp.optimize.minimize(fun = pfunc,           # objective func

                                         x0 = [0.9, 1.8, 2.3],  # initial coef

                                         method='nelder-mead')  # solver

        self.coef_ = self.res_.x

        

    def predict(self, X, coef):

        return self.bincut(coef, X)

    

def voting(x):

    votes = [0,0,0,0]

    for i,v in x.iteritems():

        votes[v] = votes[v] + 1

    return np.asarray(votes).argmax()
from sklearn.feature_selection import SelectFromModel

class TrainModels():   

    def __init__(self, cb_data=[], xgb_data=[], lgb_data=[]):

        self.cb_data = cb_data

        self.xgb_data = xgb_data

        self.lgb_data = lgb_data

        

        self.training_data = {'CB': cb_data, 'XGB': xgb_data, 'LGB': lgb_data}

        

        self.cb = []

        self.xgb = []

        self.lgb = []

        

        self.models = {'CB': [], 'XGB': [], 'LGB': []}

        

        self.fselection = {}

        

    def _get_categorical_index(self, df, cat_cols):

        cat_features_index = np.where(df.columns.isin(cat_cols))[0].tolist()

        return cat_features_index

    

    def _get_mean_coeff(self):

        coeffs = []

        m = self.cb + self.xgb + self.lgb

        for x in m:

            res = x[1].get_res().x

            coeffs.append({'C0':res[0], 'C1':res[1], 'C2':res[2], 'kappa':-(x[1].get_res().fun)})

            

        df = pd.DataFrame(coeffs)

        coeffs =[]

        for c in df.columns[:-1]:

            coeffs.append(np.average(np.array(df[c]), weights=np.array(df['kappa'])))

            

        return coeffs

    

    def _build_predict(self, df, installation_id):

        try:

            df_new = installation_id.reset_index(drop=True)

            for c in df.columns:

                df_new[c] = df[c].astype(int)

            return df_new

        except:

            return df

    

    def _train(self,library,model,X,y,categorical):

        rounder = OptRounder()

        if library == 'CB':

            m=cb.train(dtrain=cb.Pool(data=X, label=y, cat_features=self._get_categorical_index(X, categorical)), params=model[0], num_boost_round=model[1], verbose_eval=False)

            rounder.fit(m.predict(X), y)

        if library == 'XGB':

            m=xgb.train(params=model[0], dtrain=xgb.DMatrix(X, y), num_boost_round=model[1], verbose_eval =False)

            rounder.fit(m.predict(xgb.DMatrix(X)), y)

        if library == 'LGB':

            m=lgb.train(params=model[0], train_set=lgb.Dataset(X, label=y, categorical_feature = self._get_categorical_index(X, categorical), free_raw_data = False), num_boost_round=model[1], verbose_eval =False)         

            rounder.fit(m.predict(X), y)     

        return (m, rounder)

    

    def fit(self, X, y, categorical=[], folds=1, feature_selection=-1):

        self.rounders = {}

        self._fit('CB', X, y, categorical, folds, feature_selection)

        self._fit('XGB', X, y, categorical, folds, feature_selection)

        self._fit('LGB', X, y, categorical, folds, feature_selection)

    

    def _fit(self, library, X, y, categorical=[], folds=1, feature_selection=-1):

        print("Training " + library + " models")

        for model in self.training_data[library]:

            if feature_selection > 0:

                print("\tFeature Selection for " + library + " models")

                fselection = select_features(library)

                fselection.fit(self._train(library, model, X, y, categorical)[0], feature_selection)

                self.fselection[library] = fselection

            X_data = self.fselection[library].transform(X) if library in self.fselection else X

            if folds > 1:

                print("\tTraining with " + str(folds) + " folds")

                kf = KFold(n_splits=folds, random_state=7)                       

                for train_index, test_index in kf.split(X):

                    self.models[library].append(self._train(library, model, X_data.iloc[train_index], y.iloc[train_index], categorical))

            else:

                self.models[library].append(self._train(library, model, X_data, y, categorical))

    

    def predict(self, X, mean=False, installation_id=None):

        df = pd.DataFrame()

        

        for i in range(0,len(self.models['CB'])):

            X_data = self.fselection['CB'].transform(X) if 'CB' in self.fselection else X

            df['cb'+str(i)] = self.models['CB'][i].predict(X_data)

            

        for i in range(0,len(self.models['XGB'])):

            X_data = self.fselection['XGB'].transform(X) if 'XGB' in self.fselection else X

            df['xgb'+str(i)] = self.models['XGB'][i].predict(xgb.DMatrix(X_data))

            

        for i in range(0,len(self.models['LGB'])):

            X_data = self.fselection['LGB'].transform(X) if 'LGB' in self.fselection else X

            df['lgb'+str(i)] = self.models['LGB'][i].predict(X_data)

            

        if mean == True:

            df['mean'] = df.mean(axis=1)

            

        return self._build_predict(df, installation_id) 

        

    def predict_class(self, X, individual_rounder=True, mean_function=None, installation_id=None):

        if individual_rounder == False:

            coeffs = self._get_mean_coeff()

        

        df = pd.DataFrame()

        for i in range(0,len(self.models['CB'])):

            X_data = self.fselection['CB'].transform(X) if 'CB' in self.fselection else X

            df['cb_class'+str(i)] = self.models['CB'][i][1].predict(self.models['CB'][i][0].predict(X_data), self.models['CB'][i][1].get_res().x if individual_rounder==True else coeffs).astype(int)

            

        for i in range(0,len(self.models['XGB'])):

            X_data = self.fselection['XGB'].transform(X) if 'XGB' in self.fselection else X

            df['xgb_class'+str(i)] = self.models['XGB'][i][1].predict(self.models['XGB'][i][0].predict(xgb.DMatrix(X_data)), self.models['XGB'][i][1].get_res().x if individual_rounder==True else coeffs).astype(int)

            

        for i in range(0,len(self.models['LGB'])):

            X_data = self.fselection['LGB'].transform(X) if 'LGB' in self.fselection else X

            df['lgb_class'+str(i)] = self.models['LGB'][i][1].predict(self.models['LGB'][i][0].predict(X_data), self.models['LGB'][i][1].get_res().x if individual_rounder==True else coeffs).astype(int)

            

        if not mean_function is None:

            df['mean_class'] = df.apply(mean_function, axis = 1)

            

        return self._build_predict(df, installation_id)

    



def select_features(library):

    if library == 'CB':

        return SelectFeaturesCB()

    if library == 'XGB':

        return SelectFeaturesXGB()

    if library == 'LGB':

        return SelectFeaturesLGB()

    

class SelectFeatures():

    def transform(self,df):

        return df[self.features]

    

class SelectFeaturesCB(SelectFeatures):        

    def fit(self,model, threshold):

        lista = model.get_feature_importance(prettified=True)

        lista['Importances'] = lista['Importances']/100

        lista['cum'] = lista['Importances'].cumsum()

        self.features = list(lista[lista['cum']<threshold]['Feature Id'])

        print('\tNumber of selected features for CB: ' + str(len(self.features)))

        

class SelectFeaturesXGB(SelectFeatures):

    def fit(self,model, threshold):

        lista = list(model.get_score(importance_type='gain').items())

        suma = sum([x[1] for x in lista])

        lista = [(x[0],x[1]/suma) for x in lista]

        lista.sort(key = lambda x: x[1], reverse=True)

        v = 0

        i = 0

        while v < threshold:

            v = v + lista[i][1]

            i = i+1

        lista = [x[0] for x in lista]

        self.features = lista[:i-1]

        print('\tNumber of selected features for XGB: ' + str(len(self.features)))



class SelectFeaturesLGB(SelectFeatures):

    def fit(self,model, threshold):

        names = model.feature_name()

        imp = model.feature_importance(importance_type='gain')

        lista = [(names[i], imp[i]/imp.sum()) for i in range(len(names))]

        lista.sort(key = lambda x: x[1], reverse=True)

        v = 0

        i = 0

        while v < threshold:

            v = v + lista[i][1]

            i = i+1

        lista = [x[0] for x in lista]

        self.features = lista[:i-1]

        print('\tNumber of selected features for LGB: ' + str(len(self.features)))
m_normal = TrainModels(cb_data=[(best_cb, best_n_cb)], xgb_data=[(best_xgb, best_n_xgb)], lgb_data=[(best_lgb, best_n_lgb)])

m_normal.fit(X_train, y_train, categorical, folds=5, feature_selection=0.9)

df_preds_train_normal = m_normal.predict_class(X_train, mean_function=voting, installation_id=train_normal[useless] )

df_preds_train_normal.plot(title= 'Prediction - Training - Normal', subplots=True, layout=(4, 5), figsize=(16, 16), kind='hist')
m_multikid = TrainModels(cb_data=[(best_cb_mk, best_n_cb_mk)], xgb_data=[(best_xgb_mk, best_n_xgb_mk)], lgb_data=[(best_lgb_mk, best_n_lgb_mk)])

m_multikid.fit(X_train_multikid, y_train_multikid, categorical, folds=5, feature_selection=0.9)

df_preds_train_mk = m_multikid.predict_class(X_train_multikid, mean_function=voting, installation_id=train_multikid[useless] )

df_preds_train_mk.plot(title= 'Prediction - Training - Multikid', subplots=True, layout=(4, 5), figsize=(16, 16), kind='hist')
df_preds_train = pd.concat([train_normal[useless+['accuracy_group']], train_multikid[useless+['accuracy_group']]])

df_preds_train['y'] = df_preds_train['accuracy_group']

df_preds_train = df_preds_train.drop('accuracy_group', axis=1)

df_preds_train = df_preds_train.merge( pd.concat([df_preds_train_normal, df_preds_train_mk]), on=useless, how='left' )



best_kappa = 0

for a in df_preds_train_normal.columns[3:]:

    for b in df_preds_train_mk.columns[3:]:

        df = pd.concat([df_preds_train_normal[useless], df_preds_train_mk[useless] ], ignore_index=True)

        df['c'] = pd.concat([ df_preds_train_normal[a], df_preds_train_mk[b] ], ignore_index=True)

        df = df_preds_train.merge( df, on=useless, how='left' )

        kappa = cohen_kappa_score( df['c'], df_preds_train['y'], weights='quadratic')

        if kappa > best_kappa:

            best_kappa = kappa

            df_preds_train['best'] = df['c']

            col_a = a

            col_b = b





for c in df_preds_train.columns[3:]:

    print('%s => Kappa: %.3f, RMSE: %.3f, Accuracy: %.3f' % (c, cohen_kappa_score(df_preds_train[c], df_preds_train['y'], weights='quadratic'), 

                                                                     np.sqrt(mean_squared_error( df_preds_train['y'], df_preds_train[c])),

                                                                     accuracy_score( df_preds_train['y'], df_preds_train[c]) ) )
df_preds_test_normal = m_normal.predict_class(X_test, mean_function=voting, installation_id=test_normal[useless] )

df_preds_test_normal['best'] = df_preds_test_normal[col_a]

df_preds_test_mk = m_multikid.predict_class(X_test_multikid, mean_function=voting, installation_id=test_multikid[useless] )

df_preds_test_mk['best'] = df_preds_test_mk[col_b]





df_preds_test = test[useless].reset_index(drop=True)

df_preds_test = df_preds_test.merge( pd.concat([df_preds_test_normal, df_preds_test_mk]), on=useless, how='left' )

df_preds_test.plot(title= 'Prediction - Test', subplots=True, layout=(4, 5), figsize=(16, 16), kind='hist')
pd.DataFrame(df_preds_test[c].value_counts()/1000 for c in df_preds_test.columns[2:]).append(train['accuracy_group'].astype(int).value_counts()/len(train))
submission = df_preds_test[['installation_id']]

submission['accuracy_group'] = df_preds_test['best']

submission.to_csv('submission.csv', index=None)