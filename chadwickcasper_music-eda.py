# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas.plotting import scatter_matrix
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
music_df = pd.read_csv('../input/data.csv')
music_df.head(3)
genres = music_df.groupby('label')

features = list(music_df.columns)
features.remove('filename')
features.remove('label')
print(features)

for feat in features:
    fig, ax = plt.subplots(figsize=(20,10))
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    sns.violinplot(data=music_df, x='label', y=feat, figsize=(20,10))
music_features_df = music_df[features]
music_features_norm_df = scale(music_features_df)

def lower_diag_matrix_plot(matrix, title=None):
    """ Args:
        matrix - the full size symmetric matrix of any type that is lower diagonalized
        title - title of the plot
    """
    plt.style.use('default')
    
    # Create lower triangular matrix to mask the input matrix
    triu = np.tri(len(matrix), k=0, dtype=bool) == False
    matrix = matrix.mask(triu)
    fig, ax = plt.subplots(figsize=(20,20))
    if title:
        fig.suptitle(title, fontsize=32, verticalalignment='bottom')
        fig.tight_layout()
    plot = ax.matshow(matrix)
    
    # Add grid lines to separate the points
    # Adjust the ticks to create visually appealing grid/labels
    # Puts minor ticks every half step and bases the grid off this
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    # Puts major ticks every full step and bases the labels off this
    ax.set_xticks(np.arange(0, len(matrix.columns), 1))
    ax.set_yticks(np.arange(0, len(matrix.columns), 1))
    plt.yticks(range(len(matrix.columns)), matrix.columns)
    # Must put this here for x axis grid to show
    plt.xticks(range(len(matrix.columns)))
    ax.tick_params(axis='both', which='major', labelsize=24)
    # Whitens (transparent) x labels
    ax.tick_params(axis='x', colors=(0,0,0,0))
    
    # Add a colorbar for reference
    cax = make_axes_locatable(ax)
    cax = cax.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(axis='both', which='major', labelsize=24)
    fig.colorbar(plot, cax=cax, cmap='hot')
    
    # Get rid of borders of plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

cov_matrix = music_features_df.cov()
lower_diag_matrix_plot(cov_matrix, 'Covariance Matrix')

corr_matrix = music_features_df.corr()
lower_diag_matrix_plot(corr_matrix, 'Correlation Matrix')
music_df_no_categories = music_features_df.copy()
music_df_no_categories['label'] = music_df['label']
# sns.pairplot(music_df, hue='label')

enc = OneHotEncoder()
solvers = ['svd', 'eigen']
for solver in solvers:
    clf = LinearDiscriminantAnalysis(solver=solver, n_components=2)
    le = LabelEncoder()

    new_labels = pd.DataFrame(le.fit_transform(music_df['label']))
    music_df['label'] = new_labels

    params = clf.fit_transform(music_features_norm_df, new_labels,)
    fig, ax = plt.subplots()
    labels_list = list(set(list(new_labels)))
    ax.scatter(params[:,0], params[:,1], c=new_labels.as_matrix().reshape(params[:,0].shape))
    ax.legend()

def param_search(param_names, params_list, model, data, plot=True, seed=None, verbose=False):
    params_accuracies = []
    train_data, train_labels, test_data, test_labels = data
    for param_name, param_list in zip(param_names, params_list):
        accuracies = []
        kwargs = {}
        if verbose:
            print("--------------------------------")
            print("Parameter Under Test: {}\n".format(param_name))
        if type(param_list) != list:
            kwargs[param_name] = param_list
            continue
        for param_val in param_list:
            if verbose: print("Parameter search ({0} -> {1})".format(param_name, param_val))
            kwargs[param_name] = param_val
            classifier = model(**kwargs)
            classifier.fit(train_data, train_labels)
            accuracy = classifier.score(test_data, test_labels)
            accuracies.append(accuracy)
        params_accuracies.append(accuracies)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(param_list, accuracies)
        if verbose: print("--------------------------------")
    if type(param_list) != list:
        kwargs[param_name] = param_list
        classifier = model(**kwargs)
        classifier.fit(train_data, train_labels)
        accuracy = classifier.score(test_data, test_labels)
        params_accuracies.append([accuracy])
    return params_accuracies
param_names = ['n_neighbors']
params_list = [[i+3 for i in range(20)]]
dummy_param_names = ['strategy']
dummy_params_list = [['stratified', 'most_frequent', 'uniform', 'prior']]
tree_param_names = ['min_samples_split', 'max_depth']
tree_params_list = [10, 8]
tree_params_lists_only = [x for x in tree_params_list if type(x) == list]
dummy_params_lists_only = [x for x in dummy_params_list if type(x) == list]
params_lists_only = [x for x in params_list if type(x) == list]
folds = 10
random_state = random.randint(1, 65536)
print("Random State: {}".format(random_state))

cv = StratifiedKFold(n_splits=folds,
                     shuffle=True,
                     random_state=random_state,
                     )

data = list(cv.split(music_features_df, music_df['label']))
fig, ax = plt.subplots()
dummy_fig, dummy_ax = plt.subplots()
tree_fig, tree_ax = plt.subplots()

for i, indices in enumerate(data):
    train_index, test_index = indices
    title = "Training on fold {}/{}...\n".format(i+1, len(data))
    print(title)
    train_data = music_features_df.iloc[train_index]
    train_labels = music_df['label'].iloc[train_index]
    test_data = music_features_df.iloc[test_index]
    test_labels = music_df['label'].iloc[test_index]
    full_data = (train_data, train_labels, test_data, test_labels)
    
    dummy_accuracies = param_search(dummy_param_names, 
                                    dummy_params_list, 
                                    DummyClassifier, 
                                    data=full_data,
                                    plot=False)
    for X, y in zip(dummy_params_list, dummy_accuracies):
        dummy_ax.scatter(X, y, label='K Fold {}'.format(i+1))
        dummy_ax.legend()
    
    accuracies = param_search(param_names, 
                              params_list, 
                              KNeighborsClassifier, 
                              data=full_data,
                              plot=False)
    for X, y in zip(params_list, accuracies):
        ax.plot(X, y, label='K Fold {}'.format(i+1))
        ax.legend()
    
    tree_accuracies = param_search(tree_param_names, 
                                   tree_params_list, 
                                   DecisionTreeClassifier,
                                   data=full_data,
                                   plot=False)
    print(tree_params_lists_only, tree_accuracies)
    if not tree_params_lists_only:
        tree_params_lists_only = [['K Fold Results']]
    for X, y in zip(tree_params_lists_only, tree_accuracies):
        tree_ax.scatter(X, y, label='K Fold {}'.format(i+1))
        tree_ax.legend()

    

