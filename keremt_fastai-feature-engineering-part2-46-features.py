from fastai.core import *

Path.read_csv = lambda o: pd.read_csv(o)

input_path = Path("/kaggle/input/data-science-bowl-2019")

pd.options.display.max_columns=200

pd.options.display.max_rows=200

input_path.ls()
sample_subdf = (input_path/'sample_submission.csv').read_csv()

specs_df = (input_path/"specs.csv").read_csv()

train_df = (input_path/"train.csv").read_csv()

train_labels_df = (input_path/"train_labels.csv").read_csv()

test_df = (input_path/"test.csv").read_csv()
assert set(train_df.installation_id).intersection(set(test_df.installation_id)) == set()
train_with_features_part1 = pd.read_feather("../input/dsbowl2019-feng-part1/train_with_features_part1.fth")
train_with_features_part1.shape, test_df.shape, train_labels_df.shape
train_with_features_part1.head()
test_df.head()
# there shouldn't be any common installation ids between test and train 

assert set(train_df.installation_id).intersection(set(test_df.installation_id)) == set()
from fastai.tabular import *

import types



stats = ["median","mean","sum","min","max"]

UNIQUE_COL_VALS = pickle.load(open("../input/dsbowl2019-feng-part1/UNIQUE_COL_VALS.pkl", "rb"))

list(UNIQUE_COL_VALS.__dict__.keys())
# add accuracy_group unique vals

UNIQUE_COL_VALS.__dict__['accuracy_groups'] = np.unique(train_with_features_part1.accuracy_group)

UNIQUE_COL_VALS.accuracy_groups

pickle.dump(UNIQUE_COL_VALS, open( "UNIQUE_COL_VALS.pkl", "wb" ))
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
countfreqhist_dict(train_with_features_part1, "title", "accuracy_group", "accuracy_groups")
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
from sklearn.model_selection import KFold

# create cross-validated indexes

unique_ins_ids = np.unique(train_with_features_part1.installation_id)

train_val_idxs = KFold(5, random_state=42).split(unique_ins_ids)
feature_dfs = [] # collect computed _val_feats_dfs here

for train_idxs, val_idxs  in train_val_idxs:

    # get train and val dfs

    train_ins_ids, val_ins_ids = unique_ins_ids[train_idxs], unique_ins_ids[val_idxs]

    _train_df = train_with_features_part1[train_with_features_part1.installation_id.isin(train_ins_ids)]

    _val_df = train_with_features_part1[train_with_features_part1.installation_id.isin(val_ins_ids)]

    assert (_train_df.shape[0] + _val_df.shape[0]) == train_with_features_part1.shape[0]

    # compute features for val df

    _idxs = _val_df['title'].map(f1(_train_df)).index

    feat1 = np.stack(_val_df['title'].map(f1(_train_df)).values)

    feat2 = np.stack(_val_df['title'].map(f2(_train_df)).values)

    feat3 = np.stack(_val_df['title'].map(f3(_train_df)).values)

    feat4 = np.stack(_val_df['world'].map(f4(_train_df)).values)

    feat5 = np.stack(_val_df['world'].map(f5(_train_df)).values)

    feat6 = np.stack(_val_df['world'].map(f6(_train_df)).values)

    feat7 = np.stack(_val_df['title'].map(f7(_train_df)).values)

    feat8 = np.stack(_val_df['title'].map(f8(_train_df)).values)

    feat9 = np.stack(_val_df['world'].map(f9(_train_df)).values)

    feat10 = np.stack(_val_df['world'].map(f10(_train_df)).values)

    # create dataframe with same index for later merge

    _val_feats = np.hstack([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10])

    _val_feats_df = pd.DataFrame(_val_feats, index=_idxs)

    _val_feats_df.columns = [f"targenc_feat{i}"for i in range(_val_feats_df.shape[1])]

    feature_dfs.append(_val_feats_df)
train_feature_df = pd.concat(feature_dfs, 0)
train_with_features_part2 = pd.concat([train_with_features_part1, train_feature_df],1)
train_with_features_part2.to_feather("train_with_features_part2.fth")