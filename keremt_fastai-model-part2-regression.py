from fastai.core import *

Path.read_csv = lambda o: pd.read_csv(o)

input_path = Path("/kaggle/input/data-science-bowl-2019")

pd.options.display.max_columns=200

pd.options.display.max_rows=200

input_path.ls()
train_with_features_part2 = pd.read_feather("../input/dsbowl-feng-part2/train_with_features_part2.fth")
sample_subdf = (input_path/'sample_submission.csv').read_csv()

# specs_df = (input_path/"specs.csv").read_csv()

# train_labels_df = (input_path/"train_labels.csv").read_csv()

# train_df = (input_path/"train.csv").read_csv()

test_df = (input_path/"test.csv").read_csv()
sample_subdf.shape, test_df.shape, train_with_features_part2.shape
from fastai.tabular import *

import types



stats = ["median","mean","sum","min","max"]

UNIQUE_COL_VALS = pickle.load(open("../input/dsbowl-feng-part2/UNIQUE_COL_VALS.pkl", "rb"))
for k in UNIQUE_COL_VALS.__dict__.keys():

    print(k, len(UNIQUE_COL_VALS.__dict__[k]))
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
def get_sorted_user_df(df, ins_id):

    "extract sorted data for a given installation id and add datetime features"

    _df = df[df.installation_id == ins_id].sort_values("timestamp").reset_index(drop=True)

    add_datepart(_df, "timestamp", time=True)

    return _df



def get_test_assessment_start_idxs(df): 

    return list(df.sort_values("timestamp")

                  .query("type == 'Assessment' & event_code == 2000")

                  .groupby("installation_id").tail(1).index)



def get_test_feats_row(idx, i):

    "get all faeatures by an installation start idx"

    ins_id = test_df.loc[idx, "installation_id"]

    _df = get_sorted_user_df(test_df, ins_id)

    assessment_row = _df.iloc[-1]

    row_feats = np.concatenate([f(_df) for f in feature_funcs])

    feat_row = pd.Series(row_feats, index=[f"static_feat{i}"for i in range(len(row_feats))])

    row = pd.concat([assessment_row, feat_row])

    return row
# Feature Engineering part 1

start_idxs = get_test_assessment_start_idxs(test_df)

res = parallel(get_test_feats_row, start_idxs)

test_with_features_df = pd.concat(res,1).T
# Feature Engineering part 2

_idxs = test_with_features_df.index

feat1 = np.stack(test_with_features_df['title'].map(f1(train_with_features_part2)).values)

feat2 = np.stack(test_with_features_df['title'].map(f2(train_with_features_part2)).values)

feat3 = np.stack(test_with_features_df['title'].map(f3(train_with_features_part2)).values)

feat4 = np.stack(test_with_features_df['world'].map(f4(train_with_features_part2)).values)

feat5 = np.stack(test_with_features_df['world'].map(f5(train_with_features_part2)).values)

feat6 = np.stack(test_with_features_df['world'].map(f6(train_with_features_part2)).values)

feat7 = np.stack(test_with_features_df['title'].map(f7(train_with_features_part2)).values)

feat8 = np.stack(test_with_features_df['title'].map(f8(train_with_features_part2)).values)

feat9 = np.stack(test_with_features_df['world'].map(f9(train_with_features_part2)).values)

feat10 = np.stack(test_with_features_df['world'].map(f10(train_with_features_part2)).values)

# create dataframe with same index for later merge

_test_feats = np.hstack([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10])

_test_feats_df = pd.DataFrame(_test_feats, index=_idxs)

_test_feats_df.columns = [f"targenc_feat{i}"for i in range(_test_feats_df.shape[1])]
test_with_features_part2 = pd.concat([test_with_features_df, _test_feats_df],1)
# check to see train and test have same features

num_test_feats = [c for c in test_with_features_part2.columns if c.startswith("static")]

num_train_feats = [c for c in train_with_features_part2.columns if c.startswith("static")]

assert num_train_feats == num_test_feats

# check to see train and test have same features

num_test_feats = [c for c in test_with_features_part2.columns if c.startswith("targenc")]

num_train_feats = [c for c in train_with_features_part2.columns if c.startswith("targenc")]

assert num_train_feats == num_test_feats
from fastai.tabular import *
train_with_features_part2.shape, test_with_features_part2.shape
# create validation set - split by installation_id

np.random.seed(42)

valid_ids = (np.random.choice(train_with_features_part2.installation_id.unique(),

                              int(len(train_with_features_part2)*0.05)))

valid_idx = (train_with_features_part2[train_with_features_part2.installation_id.isin(valid_ids)].index); valid_idx
# get data

cat_names = ['title','world','timestampMonth','timestampWeek','timestampDay','timestampDayofweek','timestampDayofyear','timestampHour']

cont_names = [c for c in train_with_features_part2.columns if c.startswith("static")]

cont_names += [c for c in train_with_features_part2.columns if c.startswith("targenc")]



procs = [FillMissing, Categorify, Normalize]

data = TabularDataBunch.from_df(path=".", df=train_with_features_part2, dep_var="accuracy", 

                                valid_idx=valid_idx, procs=procs, cat_names=cat_names, cont_names=cont_names)



data.add_test(TabularList.from_df(test_with_features_part2, cat_names=cat_names, cont_names=cont_names));
# fit

learner = tabular_learner(data, [256,256], y_range=(0.,1.), ps=0.6)

learner.fit_one_cycle(10, 3e-3)
from sklearn.metrics import cohen_kappa_score
coefs=array([0.25,0.50,0.75])

def soft2hard(o):

    if o < coefs[0]: return 0

    elif o < coefs[1]: return 1

    elif o < coefs[2]: return 2

    else: return 3
# get valid preds

preds, targs = learner.get_preds()
# get accuracy_group for preds and targs

_preds = array([soft2hard(o.item()) for o in preds])

_targs = array(train_with_features_part2.iloc[valid_idx]['accuracy_group'].values)
# see validation score

cohen_kappa_score(_targs, _preds, weights="quadratic")
# get test preds

preds,targs=learner.get_preds(DatasetType.Test)

_preds = array([soft2hard(o.item()) for o in preds])
Counter(_preds)
# get installation ids for test set

test_ids = test_with_features_part2['installation_id'].values; len(test_ids)
# generate installation_id : pred dict

test_preds_dict = dict(zip(test_ids, _preds)); len(test_preds_dict)
# create submission

sample_subdf['accuracy_group'] = sample_subdf.installation_id.map(test_preds_dict)

sample_subdf['accuracy_group'] = sample_subdf['accuracy_group'].fillna(3)

sample_subdf.to_csv("submission.csv", index=False)