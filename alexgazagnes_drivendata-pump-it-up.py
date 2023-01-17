#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 

########################################################
########################################################
#       Driven Data - Pump it up 
########################################################
########################################################

# author  : Alexandre GAZAGNES
# date    : 18/09/2018
# version : v3

import os
print(os.listdir("../input"))
########################################################
#       Import
########################################################
# built in
import os, logging, sys, time, random, inspect
from math import ceil
import itertools as it
from collections import OrderedDict, Iterable

# data management
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
########################################################
# logging and warnings
########################################################
# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)
l = logging.INFO
logging.basicConfig(level=l, format="%(levelname)s : %(message)s")
info = logging.info

# import warnings
# warnings.filterwarnings('ignore')
########################################################
#       Graph settings
########################################################
%matplotlib inline
sns.set()
########################################################
#       Filepaths
########################################################
DATA_FOLDER     = "../input/"
TRAIN_FEATURES  = "training_set_values.csv"
TRAIN_TARGETS   = "training_set_labels.csv"
TEST_FEATURES   = "test_set_values.csv"

# test
os.path.isfile(DATA_FOLDER+TRAIN_FEATURES)
########################################################
#       Consts
########################################################
TEST_SIZE   = 0.33
CV          = 10
N_JOBS      = 6
OOB         = True   
WARM_START  = True 
SCORING     = "accuracy"
########################################################
#       DataFrame        
########################################################
def init_train_df(data, features, targets):
    """as we have features and targets in 2 csv files, we need to concat them
    we will also recast target 
    """
    
    # init df features and targets
    train_features = pd.read_csv(str(data+features), index_col=0)
    train_targets = pd.read_csv(str(data+targets), index_col=0)

    # sort if needed
    train_features.sort_index(ascending=True, inplace=True)
    train_targets.sort_index(ascending=True, inplace=True)

    # control same index to concat
    assert list(train_features.index) == list(train_targets.index)

    # concat with rename taget columns in "target"
    train_df = train_features.copy()
    train_df["target"] = train_targets.values

    # just cast our target 
    target_dict = { 'non functional'         :0, 
                    'functional'             :2, 
                    'functional needs repair':1}
    train_df["target"] = train_df.target.map(target_dict)


    # control same shape, no info loss
    assert train_df.shape[1] == (train_targets.shape[1] + train_features.shape[1])
    assert len(train_df) == len(train_features) == len(train_targets)

    return train_df

####

train_df = init_train_df(DATA_FOLDER, TRAIN_FEATURES, TRAIN_TARGETS)
train_df.head()
def init_test_df(data, features) : 
    """cf init train_df"""

    return pd.read_csv(str(data+features), index_col=0)

####

test_df = init_test_df(DATA_FOLDER, TEST_FEATURES)
test_df.head()
# control same features
assert list(test_df.columns) == list(train_df.drop("target", axis=1).columns)
def init_global_df(train_df, test_df, keep_target=True) :
    """very usefull to have noth train and test to be able to handle nan, 
    outliers etc etc """

    # checks
    if not (isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame)) :
        raise TypeError("train_df and test_df have to be pd.DataFrame objects")

    if "target" in test_df.columns : raise ValueError("targets in test_df")
    
    if (keep_target) and not ("target" in train_df.columns) : 
        raise ValueError("no targets in train_df")
    
    # keep target
    if not keep_target : _train_df = train_df.drop("target", axis=1)
    else               : _train_df = train_df.copy()

    # df
    df = _train_df.copy()
    df = df.append(test_df)

    # control shape 
    assert len(df) == (len(train_df)+ len(test_df))
    assert len(df.index) == len(df.index.unique())

    return df

####

df = init_global_df(train_df, test_df)
df.head()
# local save 
DF          = df.copy()
TRAIN_DF    = train_df.copy()
TEST_DF     = test_df.copy()
########################################################
#       EXPLORE
########################################################
def explore(df) : 
    """premier print histoire de..."""

    txt = ""
    txt += str("Dim :\n------------------\n\n{}\n\n\n".format(df.ndim))
    txt += str("Shape :\n------------------\n\n{}\n\n\n".format(df.shape))
    txt += str("Columns :\n------------------\n\n{}\n\n\n".format(df.columns))
    txt += str("Feat types :\n------------------\n\n{}\n\n\n".format(df.dtypes))
    txt += str("Index :\n------------------\n\n{}\n\n\n".format(df.index))
    txt += str("Head :\n------------------\n\n{}\n\n\n".format(df.head(3)))
    txt += str("Tail :\n------------------\n\n{}\n\n\n".format(df.tail(3)))
    txt += str("Describe :\n------------------\n\n{}\n\n\n".format(df.describe()))
    txt += str("info :\n------------------\n\n{}\n\n\n".format(df.info()))

    return txt

####

print(explore(df))
def study_unique(df, first_elems=0) : 
    """etudions les uniques pour avoir nos features continus et discrets
    reardons le nb, le % et des exemples de valeurs"""

    # uniques val and %
    uniques = [len(df[feat].unique()) for feat in df.columns]
    uniques = pd.DataFrame(uniques, index=df.columns, columns=["uniques_count"])

    l = len(df)
    perc = (uniques.uniques_count/l).round(3)
    uniques["uniques perc"] = pd.Series(perc, index=df.columns)
    
    uniques["dtypes"] = df.dtypes

    if first_elems : 
        # first elems
        N = int(first_elems)
        give_n_first = lambda x : [str(df.loc[i+1, x]) for i in range(N)]
        first_elems = [";".join(give_n_first(feat)) for feat in df.columns]
        uniques["first_elems"] = pd.Series(first_elems, index=df.columns)

    uniques = uniques.sort_values("uniques_count", axis=0)

    return uniques

####

study_unique(df,first_elems=0)
def give_uniques(df, thres=20) : 
    """regardons à la volée les différentes valeurs de nos features
    on ne garde que celles avec un nb unique en dessous d'un certain seuil""" 

    good_cols = [i for i in df.columns if len(df[i].unique()) <= thres]
    res = [list(df[feat].unique()) for feat in good_cols]
    
    return pd.Series(res, index=good_cols)

####

give_uniques(df, 30)
def delete_useless_feat(df) : 
    """detete features with just one unique value"""

    bad_cols = [feat for feat in df.columns if len(df[feat].unique())==1]
    return df.drop(bad_cols, axis=1)

####

df          = delete_useless_feat(df)
train_df    = delete_useless_feat(train_df)
test_df     = delete_useless_feat(test_df)
def handle_inconsistent_values(df) : 
    """detect fake np.nan with value as "unknown", year == 0 for exemple"""
    
    # unknown
    _df = df.copy()
    bad_feat = [feat for feat in _df.columns if "unknown" in _df[feat].unique()]
    for feat in bad_feat : 
        _df.loc[_df[feat] == "unknown", feat] = np.nan
    
    # year = 0
    feat = "construction_year"
    _df.loc[_df[feat] <1 , feat] = np.nan

    # latitude, longitude
    _df.loc[_df.longitude == 0.0, "longitude"] = np.nan
    _df.loc[_df.latitude > -0.1, "latitude"] = np.nan
    
    return _df

####

df          = handle_inconsistent_values(df)
train_df    = handle_inconsistent_values(train_df)
test_df     = handle_inconsistent_values(test_df)
# control
give_uniques(df, 30)
# control 
print(df.construction_year.describe())
print()
print(df.latitude.describe())
print()
print(df.longitude.describe())
def nan_any_cols(df) : 
    """y a t il des feature avec que des nan? 
    sont-il egalement distribués?"""
    
    l   = len(df)
    nas = [round(100 * sum(df[feat].isna()) / l, 3) for feat in df.columns]
    nas = pd.Series(nas, index=df.columns, name="% of nan by col")
    nas = nas.sort_values()
    nas = nas.loc[nas>0]

    return nas

#### 

nan_cols = nan_any_cols(df)
nan_cols
def nan_any_rows(df) : 
    """y a t il des lignes avec que des nan?
    sont-il egalement distribués"""

    if "target" in df.columns : _df = df.drop("target", axis=1)
    else                      : _df = df.copy()

    l   = len(df.columns)
    _df = _df.T
    nas = [round(np.sum(_df[feat].isna())/l,2) for feat in _df.columns]
    nas = pd.Series(nas, index=_df.columns, name="% of nan by row")
    nas = nas.sort_values()
    nas = nas.loc[nas>0]

    return nas

####

nan_rows = nan_any_rows(df)
nan_rows.head()
nan_rows.value_counts(normalize=True).sort_index()
def cast_times_series(df) : 

    df["date_recorded"] = pd.to_datetime(df.date_recorded)
    # do not take care about construction_date --> int/float
    return df

####

df          = cast_times_series(df)
train_df    = cast_times_series(train_df)
test_df     = cast_times_series(test_df)
# control 
df.dtypes.head()
def define_feat_type(df) : 

    num, cat = list(), list()

    for feat in df.columns : 
        try : 
            i = float(df.loc[2, feat])
            num.append(feat)
        except : 
            cat.append(feat)

    return num, cat

####

num, cat = define_feat_type(df)
print(num)
print(cat)
def feat_sup_classes() : 

    #                              type,  sup_class 
    feat_dict = {
     'amount_tsh'               : ("num", "waterpoint"),
     'construction_year'        : ("num", "installer"),
     'district_code'            : ("cat_num_unord", "location"),
     'gps_height'               : ("num", "location"),
     'latitude'                 : ("num", "location"),
     'longitude'                : ("num", "location"),
     'num_private'              : ("num", "other"),
     'permit'                   : ("bool", "other"), 
     'population'               : ("num", "other"),
     'public_meeting'           : ("bool", "other"),
     'region_code'              : ("cat_num_unord", "location"),
     'target'                   : ("cat_num_unord", "target"),
     'basin'                    : ("cat_str_unord", "location"),
     'date_recorded'            : ("cat_str_ord", "location",), # should be num !!!
     'extraction_type'          : ("cat_str_unord", "extraction"), 
     'extraction_type_class'    : ("cat_str_unord", "extraction"),
     'extraction_type_group'    : ("cat_str_unord", "extraction"), # same as extraction_type
     'funder'                   : ("cat_str_unord", "other"),       # far away from target???
     'installer'                : ("cat_str_unord", "installer"),
     'lga'                      : ("cat_str_unord", "location"),
     'management'               : ("cat_str_unord", "management"),
     'management_group'         : ("cat_str_unord", "management"),
     'payment'                  : ("cat_str_ord", "payment"),  # should be num!!!
     'payment_type'             : ("cat_str_ord", "payment"),  # should be num!!!
     'quality_group'            : ("cat_str_unord", "quality"),  # ??? ord OR non ORD
     'quantity'                 : ("cat_str_ord", "quantity"), # should be num!!!
     'quantity_group'           : ("cat_str_ord", "quantity"),
     'region'                   : ("cat_num_unord", "location"),
     'scheme_management'        : ("cat_str_unord", "sheme"),
     'scheme_name'              : ("cat_str_unord", "sheme"), # ??? very usefull ??? yes if one of them very correlated !! 
     'source'                   : ("cat_str_unord", "source"), # maybe ord
     'source_class'             : ("cat_str_unord", "source"),
     'source_type'              : ("cat_str_unord", "source"),
     'subvillage'               : ("cat_str_unord", "location"),
     'ward'                     : ("cat_str_unord", "location"),
     'water_quality'            : ("cat_str_unord", "quality"), # should be ordered
     'waterpoint_type'          : ("cat_str_unord", "waterpoint"),
     'waterpoint_type_group'    : ("cat_str_unord", "waterpoint"),
     'wpt_name'                 : ("cat_str_unord", "waterpoint"), # ??? very usefull ??? yes if one of them very correlated !! 
    }

    return feat_dict

####

feat_dict = feat_sup_classes()
cats      =  pd.Series([j[0] for i, j in feat_dict.items()]).value_counts()
sup_class =  pd.Series([j[1] for i, j in feat_dict.items()]).value_counts()
print(cats)
print(sup_class)
# conclusion 
# we can easily say that our nb of features will fall from 40 to 10/15 max (before one hot encoding)
# without any loss of information
########################################################
#       VISUALIZE
########################################################
# target

def graph_target(df) : 

    target = df.target.dropna()

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    target_freq = target.value_counts(normalize=True)
    sns.barplot(target_freq.index, target_freq.values, ax=ax)
    ax.set_title("target distribution")
    ax.set_ylabel("%")
    ax.set_xlabel("target")
    
####

graph_target(df)
# we have 0 1 and 2, what about 1 becomes 0 or 2 ? 
target_dist =   """ 0.0    0.384 
                    2.0    0.543
                    1.0    0.073   """

def build_target_normal_dist(df, target="target", step=300) : 
    
    target = df[target]
    target = target.dropna()
    
    cols = sorted(list(target.unique()))
    idxs = np.arange(0, len(df)-(step+2), step).astype(np.uint32)
    # info(len(idxs))
    # info(idxs[:10])
    # info(cols)
    
    target_dist = list()
    for i in idxs : 
        _target = target.iloc[i:i+step]
        dist = _target.value_counts(normalize=True).sort_index(ascending=True)
        dist = pd.Series(dist.values, name=i)
        target_dist.append(dist)
            
    target_dist = pd.DataFrame(target_dist, columns=cols)
    
    for i in target_dist.columns :
        plt.hist(target_dist[i])
    plt.legend(target_dist.columns)
    
    print(target_dist.describe())
        
    return target_dist
    
####

target_dist = build_target_normal_dist(train_df, step=300)
target_dist.head()
def graph_both_target(_df, step=300) : 

    target = _df.target.dropna()

    fig, ax = plt.subplots(1,2, figsize=(20,10))
    
    target_freq = target.value_counts(normalize=True)
    sns.barplot(target_freq.index, target_freq.values, ax=ax[0])
    ax[0].set_title("target distribution")
    ax[0].set_ylabel("%")
    ax[0].set_xlabel("target")
    
    cols = sorted(list(target.unique()))
    idxs = np.arange(0, len(_df)-(step+2), step).astype(np.uint32)
    
    target_dist = pd.DataFrame(columns=cols)
    for i in idxs : 
        _target = target.iloc[i:i+step]
        dist = _target.value_counts(normalize=True).sort_index(ascending=True)
        dist = pd.Series(dist.values, name=i)
        target_dist = target_dist.append(dist)
    
    for i in target_dist.columns :
        plt.hist(target_dist[i])
    plt.legend(target_dist.columns)
    
    print(target_dist.describe())
    
    return target_dist.describe()
    
####

graph_both_target(train_df) 
def binarize_target(_df, how): 
    """how -->  "drop_one"
                "one_become_zero"
                "one_become_two"
                "50_50_random
                "59_41_random"
                "50_50_ml" """
    
    _df = _df.loc[_df.target.notna(), :]
        
    if how == "drop_one" :
        idxs = list(shuffle(_df.loc[_df.target != 1, :].index))
        _df = _df.loc[idxs, :]
        _df.loc[_df.target == 2, : ] = 1
        
    elif how == "one_become_zero" : 
        _df["target"] = _df["target"].apply(lambda i : 0 if i <2 else 1) 
    
    elif how == "one_become_two" :         
        _df["target"] = _df["target"].apply(lambda i : 0 if i <1 else 1) 
    
    elif how == "50_50_random" :
        idxs = list(shuffle(_df.loc[_df.target == 1, :].index))
        l, n = len(idxs), int(len(idxs)*0.5)
        idxs_1, idxs_2 = idxs[0:n],idxs[n:]
        _df.loc[idxs_1, "target"], _df.loc[idxs_2, "target"] = 0,1
        _df.loc[_df.target == 2, "target"] = 1
        
    elif how == "59_41_random" :
        idxs = list(shuffle(_df.loc[_df.target == 1, :].index))
        l, n = len(idxs), int(len(idxs)*0.413)
        idxs_1, idxs_2 = idxs[0:n],idxs[n:]
        _df.loc[idxs_1, "target"], _df.loc[idxs_2, "target"] = 0,1
        _df.loc[_df.target == 2, "target"] = 1
        
    elif how == "50_50_ml" : 
        raise ValueError("Not implemented")
    
    else : 
        raise AttributeError("please check doc")
    
    # print(_df.target.value_counts(normalize=True))

    return _df


####

train_df_normal          = train_df.copy() 
train_df_drop_one        = binarize_target(train_df, "drop_one")
train_df_one_become_zero = binarize_target(train_df, "one_become_zero")
train_df_one_become_two  = binarize_target(train_df, "one_become_two")
train_df_50_50_random    = binarize_target(train_df, "50_50_random") 
train_df_59_41_random    = binarize_target(train_df, "59_41_random") 

train_df_pairs = (    ("normal",train_df_normal), 
                      ("drop_one", train_df_drop_one),
                      ("one_become_zero",train_df_one_become_zero),
                      ("one_become_two",train_df_one_become_two),
                      ("50_50_random", train_df_50_50_random),
                      ("59_41_random", train_df_59_41_random)   )


results = pd.DataFrame(columns=(0,1,2))
for name, new_train in train_df_pairs : 
    
    dist = new_train["target"].value_counts(normalize=True).round(3)
    dist.name = name
    results = results.append(dist)
             
results
graph_both_target(train_df_drop_one) 
graph_both_target(train_df_one_become_zero)
graph_both_target(train_df_one_become_two)
graph_both_target(train_df_50_50_random)
graph_both_target(train_df_59_41_random)
# lattitude, longitude

def scatter_geo(df) : 
    """regardons notre target sur la carte"""

    if df.target.isna().any() : 
        raise ValueError("no np.nan target in visualization")

    colors_dict = { 0:"red", 
                    2:"green", 
                    1:"orange" }
    colors = df.target.map(colors_dict)

    #    latitude, longitude
    # NO = -1.179141, 28.224149
    # SE = -11.593991, 41.721660
    # lat_lim = -11.593991, -1.179141
    # lon_lim = 28.224149,  41.721660

    lat_lim = df.latitude.min() -0.3,  df.latitude.max() +0.3
    lon_lim = df.longitude.min() -0.3, df.longitude.max() +0.3

    info(lon_lim)
    info(lat_lim)
    
    fig, ax = plt.subplots(1,1, figsize=(20,20))
    ax.scatter(df.longitude, df.latitude, c=colors, marker=".")
    ax.set_xlim(lon_lim)
    ax.set_xlabel("longitude")
    ax.set_ylim(lat_lim)
    ax.set_ylabel("latitude")
    

####

scatter_geo(train_df)
# conclusion
# not a global correlation between loc an target, KNN or SVM models will be fun to test
# regarding to the geographical cords we can see some local strongly correlated points, 
# maybe it will be good to identify and to handle these points (with result clip methods on a KNN or SVM algo for ex)
# do not forget other loc features as region_code for ex to identify very specific points
# height

def graph_height_1(df) : 
    
    gps_height = df.gps_height.dropna().sort_values()
    # info(gps_height.describe())
    # info(gps_height.iloc[:10])
    # info(gps_height.iloc[-10:])
    sns.distplot(df.gps_height.dropna())

####

graph_height_1(df)
def graph_height_2(df) : 
    
    _df = df.copy()
    _df = _df[["gps_height", "target"]]
    _df = _df.dropna(axis=0, how="any")

    g = sns.FacetGrid(_df, col="target", margin_titles=True)
    g.map(plt.hist, "gps_height", color="steelblue")
    
####

graph_height_2(train_df)
def graph_height_3(df) : 

    _df = df.copy()
    _df = _df[["gps_height", "target"]]
    _df = _df.dropna(axis=0, how="any")

    facet = sns.FacetGrid(_df, hue="target", aspect=4)
    facet.map(sns.kdeplot,"gps_height",shade= True)
    facet.set(xlim=(0, df["gps_height"].max()))
    facet.add_legend()
    plt.title("gps by target : normalized distribution")

####

graph_height_3(df)  
def graph_height_4(df) :

    # create a clean sub df
    _df = df.copy()
    _df = _df[["gps_height", "target"]]
    _df = _df.dropna(axis=0, how="any")
    
    # transform our feature
    gps_height = _df.gps_height
    _min, _max = gps_height.min()-1, gps_height.max()+1
    _range = range(_min, _max, 50)
    info(len(_range))
    _gps_height = pd.cut(gps_height, bins=len(_range), labels=_range)
    _df["_gps_height"] = _gps_height
    _df = _df.drop("gps_height", axis=1)
    
    # create our result dataframe 
    result = pd.DataFrame(columns=sorted(_df.target.unique()))
    for i in _df["_gps_height"].unique() : 
        ser = pd.Series(_df.loc[_df["_gps_height"]==i, "target"].value_counts(normalize=True), name=i)
        result = result.append(ser)

    # drop na, and and sort
    result = result.dropna(axis=0, how="any")
    result = result.sort_index(ascending=True)
    result.index = result.index.astype(np.int64)

    # print
    fig, ax = plt.subplots(2,1, figsize=(20,10))
    result.plot(kind='bar', stacked=True, ax=ax[0])
    plt.xlabel("gps height")
    plt.ylabel("%")
    
    result.plot(ax=ax[1])
    plt.xlabel("gps heigh")
    plt.ylabel("%")
    plt.suptitle("target distribution by gps height")
    
####

graph_height_4(train_df)
# conclusion
# ok there is a corelation between height and target
# but ...
# it not a so logical correlation, maybe high points are young ?
# then we need a hih level function to manage all these graph operations
def clean_df_for_graph(df, feat) : 
    
    _df = df.copy()
    _df = _df[["target", feat]]
    _df = _df.dropna(axis=0, how="any")

    return _df

def generic_feat_distribution(df, feat) : 

    _df = clean_df_for_graph(df, feat)

    fig, ax = plt.subplots(1,2, figsize=(20, 5))

    sns.distplot(_df[feat], ax=ax[0])
    ax[0].set_xlabel(feat)
    ax[0].set_ylabel("%")
    ax[0].set_title("{} distribution for all".format(feat))

    for i in sorted(_df.target.unique()) : 
        sub_df = _df.loc[_df.target == i, :] 
        sns.distplot(sub_df[feat], ax=ax[1])
    ax[1].set_xlabel(feat)
    ax[1].set_ylabel("%")
    ax[1].set_title("{} distribution by target".format(feat))
    ax[1].legend(sorted(_df.target.unique()))

    # g = sns.FacetGrid(_df, col="target", margin_titles=True)
    # g.map(plt.hist, feat, color="steelblue")

####

generic_feat_distribution(df,"gps_height")
# ok we have something good for distribution, 
# and for traget? 
def feature_cut(_df, feat, bins=50) : 

    _min, _max =  _df[feat].min()-1,  _df[feat].max()+1
    _range = range(int(_min), int(_max), bins)
    # info(len(_range))
    _df[feat] = pd.cut(_df[feat], bins=len(_range), labels=_range)

    return _df


def target_rate_by_feat(_df, _feat, sort_method="index") : 

    result = pd.DataFrame(columns=sorted(_df.target.unique()))
    for i in _df[_feat].unique() : 
        ser = pd.Series(_df.loc[_df[_feat]==i, "target"].value_counts(normalize=True), name=i)
        result = result.append(ser)
        
    result["vol"] = _df[_feat].value_counts(normalize=True)
    result = result.dropna(axis=0, how="any")
    
    if sort_method == "index"   : result = result.sort_index(ascending=True)
    elif sort_method == "target": result = result.sort_values(0, ascending=True)
    else                        : pass

    return result


def graph_target_rate(result, df, feat, sort_method="index") : 

    fig, ax = plt.subplots(3,1, figsize=(20,10))
    _result = result.drop("vol", axis=1)
    _result.plot(kind='bar', stacked=True, ax=ax[0])
    ax[0].set_xlabel(feat)
    ax[0].set_ylabel("%")
    
    _result.plot(ax=ax[1])
    ax[1].set_xlabel(feat)
    ax[1].set_ylabel("%")

    _hist = result["vol"] 
    _hist.plot(kind="bar", stacked=False, ax=ax[2], color="steelblue")
    #sns.barplot(_hist.index, _hist.values, color="red")

    ax[2].set_xlabel(feat)
    ax[2].set_ylabel("vol")

    plt.suptitle("target distribution by {}".format(feat))


def generic_graph_target_by_feat(df, feat, 
                                     bins=50,  
                                     sort_method="index", 
                                     cast_index=None,
                                     meth=None,) : 

    # clean df
    _df = clean_df_for_graph(df, feat)
    
    # cast if needed 
    if  cast_index : _df[feat] = _df[feat].astype(cast_index)
    
    # transform if needed 
    if meth : _df[feat] = _df[feat].apply(meth)

    # cast if needed 
    if  cast_index : _df[feat] = _df[feat].astype(cast_index)

    # cut if needed
    if bins :  _df =feature_cut(_df, feat, bins=bins)
        
    result = target_rate_by_feat(_df, feat, sort_method=sort_method)
    graph_target_rate(result, _df, feat, sort_method=sort_method)

    return result
####

results = generic_graph_target_by_feat(df, "gps_height", 50, "index", None, None)
results.head()
# construction_year

generic_feat_distribution(df,"construction_year")
results = generic_graph_target_by_feat(train_df, "construction_year", 0, "index", np.uint32)
# conclusion
# not so easy... even there is a global correlation between year and target, it seems to be false for last 5 years
# amount_tsh

amount_tsh = df.amount_tsh
sns.distplot(amount_tsh)
amount_tsh = amount_tsh[amount_tsh!=0.0]
sns.distplot(amount_tsh)
amount_tsh = np.log10(amount_tsh)
sns.distplot(amount_tsh)
# ok so we need to use a transformation function for this feature  
transf = lambda i : -1 if i == 0 else np.log10(i)
results = generic_graph_target_by_feat(df=train_df, feat="amount_tsh", bins=0, 
                            sort_method="index",cast_index=np.int64, meth=transf)

transf = lambda i : np.log1p(i)
results = generic_graph_target_by_feat(df=train_df, feat="amount_tsh", 
                    bins=0, sort_method="index",cast_index=np.int64, meth=transf)

# Conclusion : amount_tsh could de interpreted as binary feat : 0 or not 0.0
# it could also be interpreted as a categorical ordered feature, results will be better than 
# in a classic continuous numercial feature...  
def explore_feat(df, feat, n=10, prec=2) : 

    def _print(txt, title=None) : 
        print(title)
        print(txt)
        print()
    
    nas = round(100 * sum(df[feat].isna()) / len(df[feat]), prec)
    _print(nas, "nas %")
    
    _feat = df[feat].dropna()
    
    _print(_feat.describe(), "describe")
    _print(_feat.dtype, "dtype")
    _print(len(_feat.unique()), "nb of uniques")
    _print(list(_feat.unique()[:n]), "n first unique values")
    _print(list(_feat[:n]), "n first elem")
    _print(list(_feat.sort_values()[:n]), "n first sorted elem")
    _print(list(_feat.sort_values()[-n:]), "n last sorted elem")
    
####

explore_feat(df, "num_private")
num_private = df.num_private
sns.distplot(num_private)
num_private = num_private[num_private>0.0]
sns.distplot(num_private)
num_private = num_private[num_private<160.0]
sns.distplot(num_private)
# conclusion : very difficult to decide
# drop this feature???
# quantity 

explore_feat(df, "quantity")
_df = df.copy()
_df = _df.loc[_df.quantity.notna(), :]
quant_dict = {'dry':0, 'enough':3, 'insufficient':1, 'seasonal':2}
_df["quantity"] = _df.quantity.map(quant_dict)

####

generic_feat_distribution(_df,"quantity")
results = generic_graph_target_by_feat(df=_df, feat="quantity", 
                    bins=0, sort_method="index")
# conclusion
# we could cast quantity as numercial ordered variable 
# very good correlation by the way
# we should be able to industrialize our method :) 
# payement 

feat = "payment"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
results
# how to say if a feature is very "special", 
# let's consider our normal target distribution, for 0 
target_normal_dist = graph_both_target(train_df) 
target_0_mean, target_0_std, target_0_med = target_normal_dist.loc["mean", 0].round(3), target_normal_dist.loc["std", 0].round(3), target_normal_dist.loc["50%", 0].round(3)
target_0_q1, target_0_q3 = target_normal_dist.loc["25%", 0].round(3), target_normal_dist.loc["75%", 0].round(3)
target_0_IQ = target_0_q3 - target_0_q1
print(target_0_mean, target_0_std)
print(target_0_q1, target_0_q3)
# say if one val is > K std x mean
K = 1.96
_min_mean, _max_mean = (target_0_mean - (K * target_0_std)).round(3),  (target_0_mean + (K * target_0_std)).round(3)

K = 1.5 
_min_med, _max_med = (target_0_q1 - (K * target_0_IQ)).round(3), (target_0_q3 + (K * target_0_IQ)).round(3) 

print(_min_mean, _max_mean)
print(_min_med, _max_med)
print(results)
def find_valid_one_hot_features(_results, _min=0.317, _max=0.45) : 
    
    ser = _results[0.0]
    good_feats = [str(i) for i in ser.index if ((ser[i]>=_max) or (ser[i]<=_min))]
    
    return good_feats

####

good_feats = find_valid_one_hot_features(results)
good_feats
# conclusion
# extraction_type

feat = "extraction_type"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
good_feats = find_valid_one_hot_features(results)
good_feats
# extraction_type keep outliers as one hot 
# payment

feat = "payment"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
good_feats = find_valid_one_hot_features(results)
good_feats
# payement keep outliers as one hot
# population

feat = "population"
explore_feat(df, feat)
generic_feat_distribution(df, feat)
_df = df.copy()
_df = _df.loc[_df.population.notna(), :]

transf = lambda i : np.log1p(i)
_df["population"] = _df["population"].apply(transf)
generic_feat_distribution(_df, "population")
results = generic_graph_target_by_feat(df=train_df, feat="population", 
                    bins=0, sort_method="index",cast_index=np.int64, meth=transf)
_df = df.copy()
_df = _df.loc[_df.population.notna(), :]

transf = lambda i : -1 if (i == 0) else np.log10(i)
_df["population"] = _df["population"].apply(transf)
generic_feat_distribution(_df, "population")
results = generic_graph_target_by_feat(df=train_df, feat="population", 
                    bins=0, sort_method="index",cast_index=np.int64, meth=transf)
good_feats = find_valid_one_hot_features(results)
good_feats
# population, keep as cat with log1p or log10 transform
# date recorded 
# water_quality

feat = "water_quality"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
good_feats = find_valid_one_hot_features(results)
good_feats
# water_quality keep as one hot or cat 
# source_type

feat = "source_type"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
good_feats = find_valid_one_hot_features(results)
good_feats
# source_type keep as cat, regroup or one hot outliers
# waterpoint_type

feat = "waterpoint_type"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
good_feats = find_valid_one_hot_features(results)
good_feats
# waterpoint_type : keep as cat variable 
# regroup them by target to reduce nb
# management_group

feat = "management_group"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
print(results)
good_feats = find_valid_one_hot_features(results)
good_feats
# management_group : drop
# scheme_management

feat = "scheme_management"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
print(results)
good_feats = find_valid_one_hot_features(results)
good_feats
# scheme management keep as one hot encoding outliers
# installer

feat = "installer"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
# installer drop
# funder

feat = "funder"
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
# funder : drop
# permit

feat = 'permit'
explore_feat(df,feat)
n = len(df[feat].unique()) ; info(n) ; 
if n <30 : results = generic_graph_target_by_feat(df, feat, bins=0, sort_method="index")
print(results)
good_feats = find_valid_one_hot_features(results)
good_feats
# permit : drop
########################################################
#       DUMMY/NAIVE MODELS
########################################################
def classification_rate(y_te, y_pre) : 

    return 100 * (y_te == y_pre).mean().round(4)
def dummy_model(df) : 
    """voyons le dummy modele pour avoir notre accuracy score le plus 
    faible..."""

    if df.target.isna().any() : raise ValueError("no np.nan target")

    X, y = pd.concat([df.longitude, df.latitude], axis=1), df.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE)
    dumm = DummyClassifier()
    dumm.fit(X_tr, y_tr)
    y_pred = dumm.predict(X_te)
    acc = classification_rate(y_te, y_pred)

    return acc

####

dummy_acc = dummy_model(train_df)
dummy_acc
def binarize_all(_df) : 

    train_df_normal          = _df.copy() 
    train_df_drop_one        = binarize_target(_df, "drop_one")
    train_df_one_become_zero = binarize_target(_df, "one_become_zero")
    train_df_one_become_two  = binarize_target(_df, "one_become_two")
    train_df_50_50_random    = binarize_target(_df, "50_50_random") 
    train_df_59_41_random    = binarize_target(_df, "59_41_random") 

    train_df_pairs = (    ("normal",train_df_normal), 
                          ("drop_one", train_df_drop_one),
                          ("one_become_zero",train_df_one_become_zero),
                          ("one_become_two",train_df_one_become_two),
                          ("50_50_random", train_df_50_50_random),
                          ("59_41_random", train_df_59_41_random)   )
    
    return train_df_pairs

def parse_model_for_target_binarize(_df, model) : 
    
    train_df_pairs = binarize_all(_df)

    for name, new_train in train_df_pairs : 
        acc = model(new_train)
        print(name)
        print("score = {:.3f}".format(acc))
        print()

####

parse_model_for_target_binarize(train_df, dummy_model)  
def naive_model_geo(df, N=15) : 
    """essayons à la volée un modele naif basé sur la loc"""

    if df.target.isna().any() : 
        raise ValueError("no np.nan target in visualization")

    _df = df[["longitude", "latitude", "target"]]
    _df =_df.dropna(axis=0, how="any")

    X, y = pd.concat([_df["longitude"], _df["latitude"]], axis=1), _df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE)
    
    knn = KNeighborsClassifier()
    params = {"n_neighbors": list(range(1,N))}
    grid = GridSearchCV(knn, params, cv=CV, n_jobs=N_JOBS, scoring=SCORING)
    grid.fit(X_tr, y_tr)
    info(grid.best_score_)
    info(grid.best_params_)

    y_pred = grid.predict(X_te)
    acc = classification_rate(y_te, y_pred)

    return acc

####

geo_acc = naive_model_geo(train_df)
geo_acc
parse_model_for_target_binarize(train_df, naive_model_geo)
def naive_model_year(df) : 
    """essayons à la volée un modele naif basé sur la loc"""

    if df.target.isna().any() : 
        raise ValueError("no np.nan target in visualization")

    _df = handle_inconsistent_values(df)
    _df = clean_df_for_graph(_df, "construction_year").astype(np.float32)


    X, y = _df.drop("target", axis=1), _df.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE)

    lr = LogisticRegression()
    params = {}
    grid = GridSearchCV(lr, params, cv=CV, n_jobs=N_JOBS, scoring=SCORING)
    grid.fit(X_tr, y_tr)

    info(grid.best_score_)
    info(grid.best_params_)

    y_pred = grid.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    return acc

####

year_acc = naive_model_year(train_df)
parse_model_for_target_binarize(train_df, naive_model_year)
def naive_model_quantity(df) : 
    """essayons à la volée un modele naif basé sur la loc"""

    if df.target.isna().any() : 
        raise ValueError("no np.nan target in visualization")

    _df = handle_inconsistent_values(df)
    _df = clean_df_for_graph(_df, "quantity").astype(np.float32)

    X, y = _df.drop("target", axis=1), _df.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE)

    lr = LogisticRegression()
    params = {}
    grid = GridSearchCV(lr, params, cv=CV, n_jobs=N_JOBS, scoring=SCORING)
    grid.fit(X_tr, y_tr)

    info(grid.best_score_)
    info(grid.best_params_)

    y_pred = grid.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    return acc
good_train_df = train_df.loc[train_df.quantity.notna(), :]

quantity_dict = {'enough':3.0, 'insufficient':1.0, 'dry':0.0, 'seasonal':2.0}
good_train_df["quantity"] = good_train_df["quantity"].map(quant_dict)
info(good_train_df["quantity"].unique())

quantity_acc = naive_model_quantity(good_train_df)
quantity_acc
parse_model_for_target_binarize(good_train_df, naive_model_quantity)









