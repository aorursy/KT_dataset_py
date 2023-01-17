import pandas as pd
import glob

import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm.autonotebook import tqdm

import seaborn as sns
import scipy.special
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import holoviews as hv
hv.extension('bokeh')

from bokeh.plotting import figure, show

import pickle as pkl
import os
import json

from io import StringIO

import pandas as pd
from sklearn.tree import export_graphviz
from IPython.display import display

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import holoviews as hv
hv.extension('bokeh')
from bokeh.plotting import figure, show

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%output backend='bokeh'
%load_ext autoreload
%autoreload 2
def plot_distribution(X_trues, X_falses, title=''):
    """
    Plot the distribution of values a single feature across two classes.
        Example:
            - Feature: height
            - Class: is_psagotnik
    
    """
    a = hv.Distribution(X_trues, label='True')
    b = hv.Distribution(X_falses, label='False')
    display((a * b).options(width=800, height=400, title=title))


def plot_binary_distribution(X_trues, X_falses, title='', groups=('Trues', 'Falses'), labels=('1', '0')):
    """
    Plot the distribution of values a single feature across two classes for binary (0, 1) features.
        Example:
            - Feature: did call Hana last week
            - Class: is_psagotnik
    """
    assert len([x for x in X_trues if x == 1 or x==0]) == len(X_trues), "X_trues should have only 1, 0"
    assert len([x for x in X_falses if x == 1 or x==0]) == len(X_falses), "X_falses should have only 1, 0"

    true_1_density = len([x for x in X_trues if x == 1]) / len(X_trues)
    true_0_density = len([x for x in X_trues if x == 0]) / len(X_trues)

    false_1_density = len([x for x in X_falses if x == 1]) / len(X_falses)
    false_0_density = len([x for x in X_falses if x == 0]) / len(X_falses)

    keys = itertools.product(groups, labels)
    values = [true_1_density, true_0_density, false_1_density, false_0_density]
    bars = hv.Bars([k+(v,) for k, v in zip(keys, values)], ['Group', 'Label'], '%').opts(tools=['hover'])
    print(values)
    stacked = bars.opts(stacked=True, clone=True, tools=['hover'])
    display((bars.relabel(group='Grouped') + stacked.relabel(group='Stacked')).options(title=title))


def t_test(arr1, arr2):
    """
    Calculate the T-test for the means of *two independent* samples of scores.
    Test if arr1 and arr2 come from the same distribution.
    This can help us test if for a single feature f1 behaves differently for two different populations (i.e. boys, girls)
    Returns
    -------
    pvalue : float or array
        The two-tailed p-value (smaller means that we have a higher chance that the two populations behave differently).
            lower than 0.05 usually means that they ARE DIFFERENT
    Note:
        You can refer to the documentation of stats.ttest_ind() for more details
    """
    return stats.ttest_ind(arr1, arr2)[1]


# holoviews confusion
def plot_confusion_matrix(y_pred, 
                        y_true):
    """
        Plots an interactive confusion matrix using 
    """
    pdf = pd.DataFrame(list(zip(y_pred, y_true)), columns=['Prediction', 'Actual'])

    graph = pdf.groupby(['Prediction', 'Actual']).size().to_frame().reset_index()
    confusion = graph.rename(columns={0: 'Count'})
    # in a format for holoviews
    conf_values = map(lambda l: [str(l[0]), str(l[1]), l[2]], [a.tolist() for a in confusion.values])  
    p = hv.HeatMap(conf_values, label='Confusion Matrix', kdims=['Predicted', 'Actual'], vdims=['Count']).sort().options(
            xrotation=45, width=400, height=400, cmap='blues', tools=['hover'], invert_yaxis=True, zlim=(0,1))
    display(p)
DATA_DIR = '/kaggle/input/psagot2020mldata/'
family_data_df = pd.read_csv(os.path.join(DATA_DIR, 'family_data.csv'))
person_data_df = pd.read_csv(os.path.join(DATA_DIR, 'person_data.csv'))
with open(os.path.join(DATA_DIR, "train_ids_labels.json"), 'rb') as f:
    train_ids_labels = {int(k): bool(v) for k, v in json.load(f).items()}
    
with open(os.path.join(DATA_DIR, "dev_ids.json"), 'rb') as f:
    dev_ids = json.load(f)

with open(os.path.join(DATA_DIR, "test_ids.json"), 'rb') as f:
    test_ids = json.load(f)


def get_terror_ids():
    return [k for k, v in train_ids_labels.items() if v]

def get_not_terror_ids():
    return [k for k, v in train_ids_labels.items() if not v]
def merge_call_dfs(paths):
    df_from_each_file = (pd.read_csv(f, index_col=False) for f in all_files)
    calls_df = pd.concat(df_from_each_file, ignore_index=True)
    calls_df = calls_df.drop('Unnamed: 0', axis=1)
    return calls_df

all_files = glob.glob(os.path.join(DATA_DIR, 'phonecalls', "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
calls_df = merge_call_dfs(all_files)
def display_dataframe(df, title):
    display(hv.Table(df).opts(height=200, width=1000, title=title))
display_dataframe(person_data_df, title='person_data')
display_dataframe(family_data_df, title='family_data')
# Insert more visualizations here (for call data, etc.)
c = 0
def fill_parent_id_by_siblings(parent1_id, parent2_id, parent1_col, parent2_col, f_df=family_data_df):
    # try to fill parent1_id, if its already full or parent2_id doesn't exist, pass
    if parent1_id != 0 or parent2_id == 0:
        return parent1_id
    
    # look for rows where both parents exists, and parent2_id matches ours
    matching_rows_df = f_df[(f_df[parent2_col] == parent2_id) & (f_df[parent1_col] != 0)]
    if len(matching_rows_df) != 0:
        global c
        c+=1
        return matching_rows_df[parent1_col].values[0]  # return first match

#TODO - add more filling functions        

family_data_df["father_id"] = family_data_df.apply(lambda x: fill_parent_id_by_siblings(x.father_id, x.mother_id, "father_id", "mother_id"), axis=1)
family_data_df["mother_id"] = family_data_df.apply(lambda x: fill_parent_id_by_siblings(x.mother_id, x.father_id, "mother_id", "father_id"), axis=1)

print(c)
data_df = person_data_df.merge(family_data_df, on='id')
display_dataframe(data_df, title='data_df')
train_id_label_list = list(train_ids_labels.items())

id_label_list_train, id_label_list_val =  train_test_split(train_id_label_list, test_size=0.20, random_state=32)
train_ids, val_ids = [x[0] for x in id_label_list_train], [x[0] for x in id_label_list_val]
y_train, y_val = [x[1] for x in id_label_list_train], [x[1] for x in id_label_list_val]

train_data_df = data_df[data_df['id'].isin(train_ids)].set_index('id')
train_data_df = train_data_df.loc[train_ids].reset_index()

val_data_df = data_df[data_df['id'].isin(val_ids)].set_index('id')
val_data_df = val_data_df.loc[val_ids].reset_index()
num_training = len(train_data_df)
num_validation = len(val_data_df)

print(f"Number of training examples: {num_training}")
print(f"Number of validation examples: {num_validation}")
plot_binary_distribution(y_train, y_val, title='Distribution of labels in data splits', groups=('Train', 'Validation'), labels=('True', 'False'))
def extract_gender(samples_df):
    return samples_df["gender"].apply(lambda g: 0 if g == "m" else 1)

def extract_age(samples_df):
    return samples_df["age"]

# phonecall features
def how_many_calls_i_made(samples_df, all_calls_df=calls_df):
    called_to = all_calls_df[['id_to', 'id_from']].groupby('id_from').aggregate('count')
    def num_calls(row):
        return called_to.loc[row.id][0]
    
    return samples_df.apply(num_calls, axis=1)
    
# TODO - add more feature functions

features = [
    extract_gender, 
    extract_age, 
    how_many_calls_i_made
] # add more functions here
terror_ids = get_terror_ids()
not_terror_ids = get_not_terror_ids()

def get_feat_val_terrors_not_terrors(feat_func):
    feat_data = feat_func(data_df)
    feat_val_terror = feat_data[data_df['id'].isin(terror_ids)].values.astype(float)
    feat_val_not_terror = feat_data[data_df['id'].isin(not_terror_ids)].values.astype(float)
    return feat_val_terror, feat_val_not_terror

def plot_feature(feat_func):
    feat_val_terror, feat_val_not_terror = get_feat_val_terrors_not_terrors(feat_func)
    plot_distribution(feat_val_terror, feat_val_not_terror, feat_func.__name__)
    
def plot_binary_feature(feat_func):
    feat_val_terror, feat_val_not_terror = get_feat_val_terrors_not_terrors(feat_func)
    plot_binary_distribution(feat_val_terror, feat_val_not_terror, feat_func.__name__)

for f in features:
    plot_feature(f)
def create_features_matrix(samples, features_functions=features):
    return pd.concat([pd.DataFrame(feat(samples).values, columns=[feat.__name__]) for feat in features_functions], axis=1)
# Extract features using create_features_matrix
features_matrix_train = create_features_matrix(train_data_df)
features_matrix_val = create_features_matrix(val_data_df)
# SVM
svm = SVC()
svm.fit(features_matrix_train, y_train)
# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(features_matrix_train, y_train)
# try several models
def train_val_score(trained_model, evaluation_metric):
    print(f"Model: {trained_model.__class__.__name__}\n"
          f"Metric: {evaluation_metric.__name__}")
    y_pred_train = trained_model.predict(features_matrix_train)
    y_pred_val = trained_model.predict(features_matrix_val)
    print(f"Train score: {evaluation_metric(y_train, y_pred_train):>15.3f}")
    print(f"Validation score: {evaluation_metric(y_val, y_pred_val):>10.3f}")
train_val_score(svm, accuracy_score)
train_val_score(tree, accuracy_score)
# Measure your performance in more ways here
# Perform some hyper-parameter optimization here
best_model = svm
def prepare_features_for_ids(ids):
    xs = data_df[data_df['id'].isin(ids)].reset_index(drop=True).set_index('id')
    xs = xs.loc[ids].reset_index()
    return create_features_matrix(xs)

features_matrix_dev = prepare_features_for_ids(dev_ids)
features_matrix_test = prepare_features_for_ids(test_ids)
def get_submission_dict(feature_matrix, ids, trained_model):
    y_preds = trained_model.predict(feature_matrix)
    return {str(k): int(v) for k, v in zip(ids, y_preds)}

submission_dict = {'dev': get_submission_dict(features_matrix_dev, dev_ids, best_model),
                   'test': get_submission_dict(features_matrix_test, test_ids, best_model)}

submission_path = 'my_first_submission.json'

with open(submission_path, 'w') as f:
    json.dump(submission_dict, f)