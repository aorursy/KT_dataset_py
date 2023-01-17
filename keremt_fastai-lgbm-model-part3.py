from fastai.core import *

Path.read_csv = lambda o: pd.read_csv(o)

input_path = Path("/kaggle/input/data-science-bowl-2019") # kaggle

# input_path = Path("data/") # local

pd.options.display.max_columns=200

pd.options.display.max_rows=200

input_path.ls()
# precomputed train df

train_with_features_part3 = pd.read_feather("../input/dsbowlfengpart3/train_with_features_part3.fth") #kaggle

# train_with_features_part3 = pd.read_feather("output/dsbowl-feng-part3/train_with_features_part3.fth") #local
train_with_features_part3.head()
# sample_subdf = (input_path/'sample_submission.csv').read_csv()

# specs_df = (input_path/"specs.csv").read_csv()

# train_labels_df = (input_path/"train_labels.csv").read_csv()

# train_df = (input_path/"train.csv").read_csv()

test_df = (input_path/"test.csv").read_csv()
test_df.shape
# for c in ['event_id', 'type', 'title', 'world', 'event_code']: print(c, set(test_df[c]).difference(set(train_df[c])))
from fastai.tabular import *

import types



stats = ["median","mean","sum","min","max"]

UNIQUE_COL_VALS = pickle.load(open("../input/dsbowlfengpart3/UNIQUE_COL_VALS.pkl", "rb")) #kaggle

# UNIQUE_COL_VALS = pickle.load(open("output/dsbowl-feng-part3/UNIQUE_COL_VALS.pkl", "rb")) #local
for k in UNIQUE_COL_VALS.__dict__.keys(): print(k, len(UNIQUE_COL_VALS.__dict__[k]))
def array_output(f):

    def inner(*args, **kwargs): return array(listify(f(*args, **kwargs))).flatten()

    return inner



feature_funcs = []



@array_output

def time_elapsed_since_hist_begin(df):

    "total time passed until assessment begin"

    return df['timestampElapsed'].max() - df['timestampElapsed'].min()



feature_funcs.append(time_elapsed_since_hist_begin)



@array_output

def time_elapsed_since_each(df, types, dfcol):

    "time since last occurence of each types, if type not seen then time since history begin"

    types = UNIQUE_COL_VALS.__dict__[types]

    last_elapsed = df['timestampElapsed'].max()

    _d = dict(df.iloc[:-1].groupby(dfcol)['timestampElapsed'].max())

    return [last_elapsed - _d[t] if t in _d else time_elapsed_since_hist_begin(df)[0] for t in types]



feature_funcs.append(partial(time_elapsed_since_each, types="media_types", dfcol="type"))

feature_funcs.append(partial(time_elapsed_since_each, types="titles", dfcol="title"))

feature_funcs.append(partial(time_elapsed_since_each, types="event_ids", dfcol="event_id"))

feature_funcs.append(partial(time_elapsed_since_each, types="worlds", dfcol="world"))

feature_funcs.append(partial(time_elapsed_since_each, types="event_codes", dfcol="event_code"))



@array_output

def countfreqhist(df, types, dfcol, freq=False):

    "count or freq of types until assessment begin"

    types = UNIQUE_COL_VALS.__dict__[types]

    _d = dict(df[dfcol].value_counts(normalize=(True if freq else False)))

    return [_d[t] if t in _d else 0 for t in types]



feature_funcs.append(partial(countfreqhist, types="media_types", dfcol="type", freq=False))

feature_funcs.append(partial(countfreqhist, types="media_types", dfcol="type", freq=True))



feature_funcs.append(partial(countfreqhist, types="titles", dfcol="title", freq=False))

feature_funcs.append(partial(countfreqhist, types="titles", dfcol="title", freq=True))



feature_funcs.append(partial(countfreqhist, types="event_ids", dfcol="event_id", freq=False))

feature_funcs.append(partial(countfreqhist, types="event_ids", dfcol="event_id", freq=True))



feature_funcs.append(partial(countfreqhist, types="worlds", dfcol="world", freq=False))

feature_funcs.append(partial(countfreqhist, types="worlds", dfcol="world", freq=True))



feature_funcs.append(partial(countfreqhist, types="event_codes", dfcol="event_code", freq=False))

feature_funcs.append(partial(countfreqhist, types="event_codes", dfcol="event_code", freq=True))



@array_output

def overall_event_count_stats(df):

    "overall event count stats until assessment begin"

    return df['event_count'].agg(stats)

feature_funcs.append(overall_event_count_stats)



@array_output

def event_count_stats_each(df, types, dfcol):

    "event count stats per media types until assessment begin, all zeros if media type missing for user"

    types = UNIQUE_COL_VALS.__dict__[types]

    _stats_df = df.groupby(dfcol)['event_count'].agg(stats)

    _d = dict(zip(_stats_df.reset_index()[dfcol].values, _stats_df.values))

    return [_d[t] if t in _d else np.zeros(len(stats)) for t in types]

feature_funcs.append(partial(event_count_stats_each, types="media_types", dfcol="type"))

feature_funcs.append(partial(event_count_stats_each, types="titles", dfcol="title"))

feature_funcs.append(partial(event_count_stats_each, types="event_ids", dfcol="event_id"))

feature_funcs.append(partial(event_count_stats_each, types="worlds", dfcol="world"))

feature_funcs.append(partial(event_count_stats_each, types="event_codes", dfcol="event_code"))



@array_output

def overall_session_game_time_stats(df):

    "overall session game time stats until assessment begin"

    return df['game_time'].agg(stats)

feature_funcs.append(overall_session_game_time_stats)



@array_output

def session_game_time_stats_each(df, types, dfcol):

    "session game time stats per media types until assessment begin, all zeros if missing for user"

    types = UNIQUE_COL_VALS.__dict__[types]

    _stats_df = df.groupby(dfcol)['game_time'].agg(stats)

    _d = dict(zip(_stats_df.reset_index()[dfcol].values, _stats_df.values))

    return [_d[t] if t in _d else np.zeros(len(stats)) for t in types]

feature_funcs.append(partial(session_game_time_stats_each, types="media_types", dfcol="type"))

feature_funcs.append(partial(session_game_time_stats_each, types="titles", dfcol="title"))

feature_funcs.append(partial(session_game_time_stats_each, types="event_ids", dfcol="event_id"))

feature_funcs.append(partial(session_game_time_stats_each, types="worlds", dfcol="world"))

feature_funcs.append(partial(session_game_time_stats_each, types="event_codes", dfcol="event_code"))



len(feature_funcs)
def get_test_assessment_start_idxs(df): 

    return list(df.sort_values("timestamp")

                  .query("type == 'Assessment' & event_code == 2000")

                  .groupby("installation_id").tail(1).index)



def get_sorted_user_df(df, ins_id):

    "extract sorted data for a given installation id and add datetime features"

    _df = df[df.installation_id == ins_id].sort_values("timestamp").reset_index(drop=True)

    add_datepart(_df, "timestamp", time=True)

    return _df



def get_test_feats_row(idx, i):

    "get all faeatures by an installation start idx"

    df = test_df

    ins_id = df.loc[idx, "installation_id"]

    _df = get_sorted_user_df(df, ins_id)

    assessment_row = _df.iloc[-1]

    row_feats = np.concatenate([f(_df) for f in feature_funcs])

    feat_row = pd.Series(row_feats, index=[f"static_feat{i}"for i in range(len(row_feats))])

    row = pd.concat([assessment_row, feat_row])

    return row
# # testit with single assessment row

# start_idxs = get_test_assessment_start_idxs(test_df)

# row = get_test_feats_row(start_idxs[0], 0)
# Feature Engineering part 1

start_idxs = get_test_assessment_start_idxs(test_df)

res = parallel(partial(get_test_feats_row), start_idxs)
colnames = res[0].index

test_with_features_df = pd.DataFrame(np.vstack(res).tolist(), columns=colnames)
del res; gc.collect()
test_with_features_df.shape
def target_encoding_stats_dict(df, by, targetcol):

    "get target encoding stats dict, by:[stats]"

    _stats_df = df.groupby(by)[targetcol].agg(stats)   

    _d = dict(zip(_stats_df.reset_index()[by].values, _stats_df.values))

    return _d
def _value_counts(o, freq=False): return dict(pd.value_counts(o, normalize=freq))

def countfreqhist_dict(df, by, targetcol, types, freq=False):

    "count or freq histogram dict for categorical targets"

    types = UNIQUE_COL_VALS.__dict__[types]

    _hist_df = df.groupby(by)[targetcol].agg(partial(_value_counts, freq=freq))

    _d = dict(zip(_hist_df.index, _hist_df.values))

    for k in _d: _d[k] = array([_d[k][t] for t in types]) 

    return _d
f1 = partial(target_encoding_stats_dict, by="title", targetcol="num_incorrect")

f2 = partial(target_encoding_stats_dict, by="title", targetcol="num_correct")

f3 = partial(target_encoding_stats_dict, by="title", targetcol="accuracy")

f4 = partial(target_encoding_stats_dict, by="world", targetcol="num_incorrect")

f5 = partial(target_encoding_stats_dict, by="world", targetcol="num_correct")

f6 = partial(target_encoding_stats_dict, by="world", targetcol="accuracy")

f7 = partial(countfreqhist_dict, by="title", targetcol="accuracy_group", types="accuracy_groups",freq=False)

f8 = partial(countfreqhist_dict, by="title", targetcol="accuracy_group", types="accuracy_groups",freq=True)

f9 = partial(countfreqhist_dict, by="world", targetcol="accuracy_group", types="accuracy_groups",freq=False)

f10 = partial(countfreqhist_dict, by="world", targetcol="accuracy_group", types="accuracy_groups",freq=True)
# Feature Engineering part 2

_idxs = test_with_features_df.index

feat1 = np.stack(test_with_features_df['title'].map(f1(train_with_features_part3)).values)

feat2 = np.stack(test_with_features_df['title'].map(f2(train_with_features_part3)).values)

feat3 = np.stack(test_with_features_df['title'].map(f3(train_with_features_part3)).values)

feat4 = np.stack(test_with_features_df['world'].map(f4(train_with_features_part3)).values)

feat5 = np.stack(test_with_features_df['world'].map(f5(train_with_features_part3)).values)

feat6 = np.stack(test_with_features_df['world'].map(f6(train_with_features_part3)).values)

feat7 = np.stack(test_with_features_df['title'].map(f7(train_with_features_part3)).values)

feat8 = np.stack(test_with_features_df['title'].map(f8(train_with_features_part3)).values)

feat9 = np.stack(test_with_features_df['world'].map(f9(train_with_features_part3)).values)

feat10 = np.stack(test_with_features_df['world'].map(f10(train_with_features_part3)).values)
# create dataframe with same index to merge later

_test_feats_df = pd.DataFrame(np.hstack([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10]).tolist(), index=_idxs)

_test_feats_df.columns = [f"targenc_feat{i}"for i in range(_test_feats_df.shape[1])]
test_with_features_df = pd.concat([test_with_features_df, _test_feats_df],1)
test_with_features_df.shape
del _test_feats_df; gc.collect()
# check to see train and test have same features

num_test_feats = [c for c in test_with_features_df.columns if c.startswith("static")]

num_train_feats = [c for c in train_with_features_part3.columns if c.startswith("static")]

assert num_train_feats == num_test_feats

# check to see train and test have same features

num_test_feats = [c for c in test_with_features_df.columns if c.startswith("targenc")]

num_train_feats = [c for c in train_with_features_part3.columns if c.startswith("targenc")]

assert num_train_feats == num_test_feats
def save_pickle(obj,fn): pickle.dump(obj, open(fn, "wb"))

def load_pickle(fn): return pickle.load(open(fn, "rb"))
activities_map = load_pickle('../input/dsbowlfengpart3/activities_map.pkl') #kaggle

activities_world = load_pickle('../input/dsbowlfengpart3/activities_world.pkl') #kaggle



# activities_map = load_pickle('output/dsbowl-feng-part3/activities_map.pkl') #local

# activities_world = load_pickle('output/dsbowl-feng-part3/activities_world.pkl') #local
def encode_title(test):

    # encode title    

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    test['title'] = test['title'].map(activities_map)

    test['title'] = test['title'].fillna(np.max(test['title'])+1).astype(int)

    

    test['world'] = test['world'].map(activities_world)

    test['world'] = test['world'].fillna(np.max(test['world'])+1).astype(int)

    

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return test
# load same mappings used in creating train part3 data

# kaggle

assess_titles = load_pickle('../input/dsbowlfengpart3/assess_titles.pkl')

list_of_event_code = load_pickle('../input/dsbowlfengpart3/list_of_event_code.pkl')

list_of_event_id = load_pickle('../input/dsbowlfengpart3/list_of_event_id.pkl')

activities_labels = load_pickle('../input/dsbowlfengpart3/activities_labels.pkl')

all_title_event_code = load_pickle('../input/dsbowlfengpart3/all_title_event_code.pkl')

activities_labels = load_pickle('../input/dsbowlfengpart3/activities_labels.pkl')

win_code = load_pickle('../input/dsbowlfengpart3/win_code.pkl')



# assess_titles = load_pickle('output/dsbowl-feng-part3/assess_titles.pkl')

# list_of_event_code = load_pickle('output/dsbowl-feng-part3/list_of_event_code.pkl')

# list_of_event_id = load_pickle('output/dsbowl-feng-part3/list_of_event_id.pkl')

# activities_labels = load_pickle('output/dsbowl-feng-part3/activities_labels.pkl')

# all_title_event_code = load_pickle('output/dsbowl-feng-part3/all_title_event_code.pkl')

# activities_labels = load_pickle('output/dsbowl-feng-part3/activities_labels.pkl')

# win_code = load_pickle('output/dsbowl-feng-part3/win_code.pkl')
# this is the function that convert the raw data into processed features

def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''    

    # Constants and parameters declaration

    last_activity = 0

    

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}



    

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    

    # accuracies for each title in assess_titles

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}



    # init counter dicts

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

        

    # last features

    sessions_count = 0

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):



        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels.get(session_title, None)

                    

            

        # for each assessment, and only this kind of session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            _event_code = win_code.get(session_title, 4100)

            all_attempts = session.query(f'event_code == {_event_code}')

           

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            

            # copy a dict to use as feature template, it's initialized with some items: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            features['installation_session_count'] = sessions_count

            

            # also store game_session for merge with other datasets            

            features['game_session'] = session['game_session'].values[0] 

            

            variety_features = [('var_event_code', event_code_count),

                               ('var_event_id', event_id_count),

                               ('var_title', title_count),

                               ('var_title_event_code', title_event_code_count)]

            

            for name, dict_counts in variety_features:

                arr = np.array(list(dict_counts.values()))

                features[name] = np.count_nonzero(arr)

                 

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

                features['duration_std'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

                features['duration_std'] = np.std(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            if session_title_text:

                last_accuracy_title['acc_' + session_title_text] = accuracy

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            if accuracy == 0: features['accuracy_group'] = 0

            elif accuracy == 1: features['accuracy_group'] = 3

            elif accuracy == 0.5: features['accuracy_group'] = 2

            else: features['accuracy_group'] = 1

            features.update(accuracy_groups)

            accuracy_groups[features['accuracy_group']] += 1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set: all_assessments.append(features)

            elif true_attempts+false_attempts > 0: all_assessments.append(features)

                

            counter += 1

        

        sessions_count += 1

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

                num_of_session_count = Counter(session[col])

                for k in num_of_session_count.keys():

                    x = k

                    if col == 'title': 

                        if k in activities_labels: x = activities_labels[k]

                        else: x = None

                    if (x in counter) and (x is not None): counter[x] += num_of_session_count[k]

                return counter

            

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set: return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments
def get_train_and_test(test):

    from tqdm import tqdm

    compiled_test = []

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_test = pd.DataFrame(compiled_test)

    return reduce_test
# encode tet

del test_df; gc.collect()

test_df = (input_path/"test.csv").read_csv()

test = encode_title(test_df)
# tranform function to get the train and test set

_test_feats_df = get_train_and_test(test)
_test_feats_df.head()
# kernel feature columns

target_cols, cat_cols, id_cols = ['accuracy', 'accuracy_group'], ['session_title'], ['installation_id', 'game_session']

feature_cols = [c for c in _test_feats_df.columns if c not in (target_cols + cat_cols + id_cols)]
feature_cols[-10:]
# extra column for identification or category

extra_cols = target_cols + cat_cols + id_cols

test_extra_cols = [c for c in extra_cols if 'accuracy' not in c]
# create feng part 3 test data

_test_kernel_feats_df = _test_feats_df[feature_cols]

_test_kernel_feats_df.columns = [f"kernel_feat{i}" for i in range(len(_test_kernel_feats_df.columns))]

_test_kernel_feats_df[test_extra_cols] = _test_feats_df[test_extra_cols];



# merge new features to test features df

test_with_features_df = test_with_features_df.merge(_test_kernel_feats_df, on=['installation_id', 'game_session'])
del _test_feats_df, _test_kernel_feats_df

gc.collect()
test_with_features_df.head(3)
train_with_features_part3.head(3)
assert len([c for c in test_with_features_df.columns if "kernel" in c]) == len([c for c in train_with_features_part3.columns if "kernel" in c])
cat_names = ['title','world','timestampMonth','timestampWeek','timestampDay','timestampDayofweek',

             'timestampDayofyear','timestampHour'] + ['session_title']

cont_names = [c for c in train_with_features_part3.columns if c.startswith("static")]

cont_names += [c for c in train_with_features_part3.columns if c.startswith("targenc")]

cont_names += [c for c in train_with_features_part3.columns if c.startswith("kernel")]

# cont_names += [c for c in train_with_features_part3.columns if c.startswith("kernel")][-7:] # accuracy feats can't be used
feature_names = cont_names + cat_names
len(feature_names)
drop_cols = []

# drop columns with 1 unique value in train

for c in feature_names:

    if len(np.unique(train_with_features_part3[c])) == 1:

        drop_cols.append(c)
len(drop_cols)
# drop columns that have single value in all training data

train_with_features_part3 = train_with_features_part3.drop(drop_cols, 1)

test_with_features_df = test_with_features_df.drop(drop_cols, 1)
train_with_features_part3.shape, test_with_features_df.shape
# check nan values

assert not any(test_with_features_df.isna().sum() > 0)

assert not any(train_with_features_part3.isna().sum() > 0)
cat_names = ['title','world','timestampMonth','timestampWeek','timestampDay','timestampDayofweek',

             'timestampDayofyear','timestampHour'] + ['session_title']

cont_names = [c for c in train_with_features_part3.columns if c.startswith("static")]

cont_names += [c for c in train_with_features_part3.columns if c.startswith("targenc")]

cont_names += [c for c in train_with_features_part3.columns if c.startswith("kernel")]
len(cat_names), len(cont_names)
# for c in cont_names: if not any(train_with_features_part3[c] != train_with_features_part3['accuracy_group']): print(c)
from fastai.tabular import *

from fastai.callbacks import *

from fastai.metrics import RegMetrics

from sklearn.metrics import cohen_kappa_score
test_with_features_df.shape, train_with_features_part3.shape
set(train_with_features_part3.columns).difference(set(test_with_features_df.columns))
# load CV installation_ids

trn_val_ids = pickle.load(open("../input/dsbowlfengpart3/CV_installation_ids.pkl", "rb")) #kaggle

# trn_val_ids = pickle.load(open("output/dsbowl-feng-part3/CV_installation_ids.pkl", "rb")) #local
# label distribution for training fold to be used in metric

train_labels_dist = (train_with_features_part3['accuracy_group'].value_counts(normalize=True))

q = np.cumsum([train_labels_dist[i] for i in range(3)]); q
# metric

def convert_preds_with_search(preds, targs): 

    "soft accuracy 0-1 preds to accuracy groups by optimized search thresholds"

    pass



def convert_preds(preds, q):

    "soft accuracy 0-1 preds to accuracy groups given quantiles 'q'"

    preds = preds.view(-1)

    targ_thresh = np.quantile(preds, q)

    hard_preds = torch.zeros_like(preds)

    hard_preds[preds <= targ_thresh[0]] = 0

    hard_preds[(preds > targ_thresh[0]) & (preds <= targ_thresh[1])] = 1

    hard_preds[(preds > targ_thresh[1]) & (preds <= targ_thresh[2])] = 2

    hard_preds[preds > targ_thresh[2]] = 3

    return hard_preds



def convert_targs(targs):

    "convert accuracy to accuracy group for targs"

    targs = targs.view(-1)

    hard_targs = torch.zeros_like(targs)

    hard_targs[targs == 0] = 0

    hard_targs[(targs > 0) & (targs < 0.5)] = 1

    hard_targs[targs == 0.5] = 2

    hard_targs[targs == 1] = 3

    return hard_targs

# assert not any(convert_targs(tensor(train_labels_df['accuracy'])) != tensor(train_labels_df['accuracy_group']))



class KappaScoreRegression(RegMetrics):

    "uses accuracy as target"

    def __init__(self): pass

    def on_epoch_end(self, last_metrics, **kwargs):

        preds = convert_preds(self.preds, q=q)

        targs = convert_targs(self.targs)

        qwk = cohen_kappa_score(preds, targs, weights="quadratic")

        return add_metrics(last_metrics, qwk)

    

class KappaScoreRegressionv2(RegMetrics):

    "uses accuracy_group as target"

    def __init__(self): pass

    def on_epoch_end(self, last_metrics, **kwargs):

        "convert preds and calc qwk"

        preds = convert_preds(self.preds, q=q)

        targs = self.targs

        qwk = cohen_kappa_score(preds, targs, weights="quadratic")

        return add_metrics(last_metrics, qwk)
def get_learner_data(foldidx, bs=64, dep_var="accuracy"):

    # pick trn-val installation ids

    trn_ids, val_ids = trn_val_ids[foldidx]

    train_idx = (train_with_features_part3[train_with_features_part3.installation_id.isin(trn_ids)].index)

    valid_idx = (train_with_features_part3[train_with_features_part3.installation_id.isin(val_ids)].index)



    # get data

    procs = [FillMissing, Categorify, Normalize]

    data = TabularDataBunch.from_df(bs=bs, path=".", df=train_with_features_part3, dep_var=dep_var, 

                                    valid_idx=valid_idx, procs=procs, cat_names=cat_names, cont_names=cont_names)



    data.add_test(TabularList.from_df(test_with_features_df, cat_names=cat_names, cont_names=cont_names));

    return data
def fit_learner(data):

    learner = tabular_learner(data, [256,256], y_range=(0.,1), ps=0.5)

    early_cb = EarlyStoppingCallback(learner, monitor="kappa_score_regression", mode="max", patience=5)

    save_cb = SaveModelCallback(learner, monitor="kappa_score_regression", mode="max", name=f"bestmodel")

    cbs = [early_cb, save_cb]

    learner.metrics = [KappaScoreRegression()]

    learner.fit_one_cycle(10, 1e-3, callbacks=cbs)

    return learner
n_folds = 5

learner_preds = 0

learner_cv_scores = [] 

for i in range(n_folds):

    data = get_learner_data(i)

    learner = fit_learner(data)

    _learner_preds, _ = learner.get_preds(DatasetType.Test)

    learner_preds += to_np(_learner_preds.view(-1)) / n_folds

    learner_cv_scores.append(learner.validate()[-1])
np.mean(learner_cv_scores), np.std(learner_cv_scores)
plt.hist(learner_preds)
learner_preds
import lightgbm as lgb
train_with_features_part3[cat_names].head()
# need to encode string type for xgboost

title2codes = {v:k for k,v in enumerate(np.unique(train_with_features_part3['title']))}

world2codes = {v:k for k,v in enumerate(np.unique(train_with_features_part3['world']))}

train_with_features_part3['title'] = train_with_features_part3['title'].map(title2codes)

train_with_features_part3['world'] = train_with_features_part3['world'].map(world2codes)

test_with_features_df['title'] = test_with_features_df['title'].map(title2codes)

test_with_features_df['world'] = test_with_features_df['world'].map(world2codes)
def convert_preds_np(preds, q):

    "soft accuracy 0-1 preds to accuracy groups given quantiles 'q'"

    preds = preds.flatten()

    targ_thresh = np.quantile(preds, q)

    hard_preds = np.zeros_like(preds)

    hard_preds[preds <= targ_thresh[0]] = 0

    hard_preds[(preds > targ_thresh[0]) & (preds <= targ_thresh[1])] = 1

    hard_preds[(preds > targ_thresh[1]) & (preds <= targ_thresh[2])] = 2

    hard_preds[preds > targ_thresh[2]] = 3

    return hard_preds



def convert_targs_np(targs):

    "convert accuracy to accuracy group for targs"

    targs = targs.flatten()

    hard_targs = np.zeros_like(targs)

    hard_targs[targs == 0] = 0

    hard_targs[(targs > 0) & (targs < 0.5)] = 1

    hard_targs[targs == 0.5] = 2

    hard_targs[targs == 1] = 3

    return hard_targs

    

def kappa_score_regression(preds, targs):

    preds = convert_preds_np(preds, q=q)

    targs = convert_targs_np(targs.get_label())

    qwk = cohen_kappa_score(preds, targs, weights="quadratic")

    return "kappa_regression", qwk, True



def kappa_score_regression_v2(preds, targs):

    preds = convert_preds_np(preds, q=q)

    targs = targs.get_label().flatten()

    qwk = cohen_kappa_score(preds, targs, weights="quadratic")

    return "kappa_regression", qwk, True
features = cat_names + cont_names

dep_var = "accuracy"
def get_lgb_data(foldidx):

    trn_ids, val_ids = trn_val_ids[foldidx]

    train_idx = (train_with_features_part3[train_with_features_part3.installation_id.isin(trn_ids)].index)

    valid_idx = (train_with_features_part3[train_with_features_part3.installation_id.isin(val_ids)].index)

    x_train = train_with_features_part3.loc[train_idx, features]

    y_train = train_with_features_part3.loc[train_idx, dep_var]

    x_val = train_with_features_part3.loc[valid_idx, features]

    y_val = train_with_features_part3.loc[valid_idx, dep_var]

    train_set = lgb.Dataset(x_train, y_train, categorical_feature=cat_names)

    val_set = lgb.Dataset(x_val, y_val, categorical_feature=cat_names)

    test_set = lgb.Dataset(test_with_features_df[features])

    return train_set, val_set, test_set
def fit_lgb_model(train_set, val_set, test_set):

    model = lgb.train(params, train_set, valid_sets=[train_set, val_set], feval=kappa_score_regression, verbose_eval=100)

    return model
params = {'n_estimators':5000,

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',

            'subsample': 0.75,

            'subsample_freq': 1,

            'learning_rate': 0.01,

            'feature_fraction': 0.9,

            'max_depth': 15,

            'lambda_l1': 1,  

            'lambda_l2': 1,

            'verbose': 100,

            'early_stopping_rounds': 100

            }
n_folds = 5

lgb_preds = 0

lgb_cv_scores = [] 

for i in range(n_folds):

    data = get_lgb_data(i)

    lgb_model = fit_lgb_model(*data)

    _lgb_preds = lgb_model.predict(test_with_features_df[features])

    lgb_preds += _lgb_preds / n_folds

    score = lgb_model.best_score['valid_1']['kappa_regression']

    lgb_cv_scores.append(score)
np.mean(lgb_cv_scores), np.std(lgb_cv_scores)
# combine preds

# _preds = learner_preds*0.3 + lgb_preds*0.7

_preds = learner_preds*0.3 + lgb_preds*0.7
plt.hist(_preds)
# label distribution for training fold to be used in metric

train_labels_dist = (train_with_features_part3['accuracy_group'].value_counts(normalize=True))

q = np.cumsum([train_labels_dist[i] for i in range(3)])

# get accuracy groups

preds = convert_preds_np(_preds, q=q)
Counter(preds)
# get installation ids for test set

test_ids = test_with_features_df['installation_id'].values; len(test_ids)
# generate installation_id : pred dict

test_preds_dict = dict(zip(test_ids, preds)); len(test_preds_dict)
# # create submission

sample_subdf = (input_path/'sample_submission.csv').read_csv()
sample_subdf['accuracy_group'] = sample_subdf.installation_id.map(test_preds_dict).astype(int)
sample_subdf.to_csv("submission.csv", index=False)
sample_subdf.head()