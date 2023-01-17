import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from collections import defaultdict

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



target_name = 'target'

NULL_VALUE = -1



def load_data():

    print('Loading data...')

    train_df = pd.read_csv('../input/train.csv')

    test_df = pd.read_csv('../input/test.csv')

    train_df = train_df.fillna(NULL_VALUE)

    test_df = test_df.fillna(NULL_VALUE)

    return train_df, test_df





def get_distributions_for_categorical_columns(df):

    cols = [c for c in df.columns if '_cat' in c]

    vline_value = df[target_name].mean()

    for col in cols:

        col_values = df[col].values

        group_ixs = get_group_ixs(col_values)

        values = np.nan * np.zeros(len(group_ixs))

        counts = np.nan * np.zeros(len(group_ixs))

        labels = np.repeat('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', len(group_ixs))

        for i, ixs in enumerate(group_ixs.values()):

            values[i] = np.mean(df[target_name].values[ixs])

            labels[i] = df[col].values[ixs][0]

            counts[i] = len(ixs)

        values, labels, counts = _select_relevant_values(values, labels, counts)

        labels = zip(labels, counts.astype('str'))

        _plot_categorical_from_lists(values, labels,

                                     'target ratio for elements in ' + col,

                                     vline=vline_value)





def _select_relevant_values(values, labels, counts, n_top=20, min_count=30):

    sufficient_elements = counts > min_count

    values, labels, counts = values[sufficient_elements], labels[sufficient_elements], counts[sufficient_elements]

    if len(values) < n_top:

        n_top = len(values)

    top_values_ixs = np.argpartition(values, -n_top) < n_top

    botton_values_ixs = np.argpartition(-values, -n_top) < n_top

    relevant_ixs = top_values_ixs | botton_values_ixs

    return values[relevant_ixs], labels[relevant_ixs], counts[relevant_ixs]





def get_group_ixs(ids):

    id_hash = defaultdict(list)

    for j, key in enumerate(ids):

        id_hash[key].append(j)

    id_hash = {k: np.array(v) for k, v in id_hash.items()}

    return id_hash





def get_correlations_to_target(df):

    df = _filter_non_numeric_data(df)

    x_cols = [col for col in df.columns if col not in [target_name]]

    labels = []

    values = []

    for col in x_cols:

        labels.append(col)

        values.append(np.corrcoef(df[col], df[target_name])[0, 1])

    _plot_categorical_from_lists(values, labels, 'correlations for numerical columns (all of them)', x=12, y=12)

    output_df = pd.DataFrame({'correlation':values, 'column':labels})

    print(output_df.sort_values('correlation').to_string())





def _filter_non_numeric_data(df):

    return df._get_numeric_data()





def _plot_categorical_from_lists(values, labels, message, x=9, y=10, vline=None):

    ind = np.arange(len(values))

    width = 0.7

    fig, ax = plt.subplots(figsize=(x, y))

    ax.barh(ind, np.array(values), color='y')

    ax.set_yticks(ind+((width)/2.))

    ax.set_yticklabels(labels, rotation='horizontal')

    ax.set_xlabel(message)

    ax.set_title(message)

    plt.tight_layout()

    if vline is not None:

        plt.axvline(x=vline)

    plt.show()



    

def check_all_categorical_values_are_in_test_too(train, test):

    ''' This checks that all the categorical values found in the train set

        are at least 1 time in the test set '''

    cols = [c for c in train.columns if '_cat' in c]

    values_not_in_test = 0

    for col in cols:

        values = train[col].value_counts().index

        for val in values:

            ixs_test = test[col].values == val

            if sum(ixs_test) == 0:

                values_not_in_test += 1

                print ('column {} value {} not found in test!'.format(col, val))

    return values_not_in_test == 0



train_df, test_df = load_data()

targets = train_df[target_name].values



get_distributions_for_categorical_columns(train_df)
get_correlations_to_target(train_df)


assert check_all_categorical_values_are_in_test_too(train_df, test_df)