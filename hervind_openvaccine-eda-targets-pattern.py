import pandas as pd, numpy as np

from collections import Counter

import math, json, gc, random, os, sys

from matplotlib import pyplot as plt

import seaborn as sns

from tqdm import tqdm
#get comp data

train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

#target columns

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

train_data = []

for mol_id in train['id'].unique():

    sample_data = train.loc[train['id'] == mol_id]

    seq_scored = sample_data['seq_scored'].values[0]

    signal_to_noise = sample_data['signal_to_noise'].values[0]

    SN_filter = sample_data['SN_filter'].values[0]

    

    for seq_order in range(seq_scored):

        i = seq_order

        sample_tuple = (mol_id, seq_order, signal_to_noise, SN_filter, 

                        sample_data['sequence'].values[0][i],

                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i],

                        sample_data['reactivity'].values[0][i], sample_data['reactivity_error'].values[0][i],

                        sample_data['deg_Mg_pH10'].values[0][i], sample_data['deg_error_Mg_pH10'].values[0][i],

                        sample_data['deg_pH10'].values[0][i], sample_data['deg_error_pH10'].values[0][i],

                        sample_data['deg_Mg_50C'].values[0][i], sample_data['deg_error_Mg_50C'].values[0][i],

                        sample_data['deg_50C'].values[0][i], sample_data['deg_error_50C'].values[0][i])

        train_data.append(sample_tuple)

        

train_data = pd.DataFrame(train_data, columns=['mol_id', 'seq_order', 'signal_to_noise', 'SN_filter', 

                                               'sequence', 'structure', 'predicted_loop_type', 

                                               'reactivity', 'reactivity_error', 'deg_Mg_pH10', 'deg_error_Mg_pH10',

                                               'deg_pH10', 'deg_error_pH10', 'deg_Mg_50C', 'deg_error_Mg_50C', 

                                               'deg_50C', 'deg_error_50C'])

train_data.head()
feature = 'SN_filter'

feature_values = [[0,1], [0], [1]]

fig, ax = plt.subplots(figsize = (15, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        sns.distplot(train_data.loc[train_data[feature].isin(feature_value) , target_], color = color);

        plt.title(f'{target_} distribution when {feature} in {feature_value}');
feature = 'SN_filter'

feature_values = [[0,1], [0], [1]]

fig, ax = plt.subplots(figsize = (15, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        

        plt.title(f'{target_} distribution when {feature} in {feature_value}');

feature = 'SN_filter'

feature_values = [[0,1], [0], [1]]

fig, ax = plt.subplots(figsize = (15, 27))

for target_idx, target_ in enumerate(target_cols): 

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), 3, plt_idx);

        

        df_show = train_data.loc[train_data[feature].isin(feature_value) , target_].value_counts(normalize = True).iloc[:5] * 100

        df_show = pd.DataFrame(df_show).reset_index()

        df_show.columns = [f'{target_} value', 'value_pct']

        df_show = df_show.round(3)

        df_show[f'{target_} value'] = df_show[f'{target_} value'].astype('category')

        sns.barplot(data = df_show, y = f'{target_} value', x = 'value_pct');

        

        plt.title(f'{target_} value pct when {feature} in {feature_value}');
feature = 'SN_filter'

feature_values = [[0,1], [0], [1]]

fig, ax = plt.subplots(figsize = (15, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), 3, plt_idx);

        

        without_zero_df = train_data.loc[train_data[target_] != 0]

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        # remove 0 value

        show_data = show_data.loc[show_data[target_] != 0]

        sns.distplot(show_data[target_], color = color);

        

        plt.title(f'{target_} distribution when {feature} in {feature_value}');

        
print("Number of row that have value == 0")

(train_data[target_cols] == 0).sum(axis = 0)
at_least_one_target_zero = (train_data[target_cols] == 0).sum(axis = 1) >= 1 

print("total row of at least one target value == 0 is" , 

      at_least_one_target_zero.sum())



all_scored_target_zero = (train_data[['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] == 0).sum(axis = 1) == 3

print("total row of scored target value == 0 is" , 

      all_scored_target_zero.sum())



all_target_zero = (train_data[target_cols] == 0).sum(axis = 1) == 5

print("total row of all target value == 0 is" , 

      all_target_zero.sum())

train_data.loc[all_scored_target_zero].sample(10, random_state = 1)
train_data.loc[:, 'SN_filter'].value_counts(normalize = True)
train_data.loc[all_scored_target_zero, 'SN_filter'].value_counts(normalize = True)
feature = 'sequence'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (22, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} distribution when {feature} in {feature_value}');
feature = 'sequence'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (22, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} distribution when {feature} in {feature_value}');
feature = 'sequence'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (22, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

#         # add log(x+1)

#         sns.distplot(np.log1p(show_data[target_]) , color = color);



        ## I remove non positive value because log(x+1) is too much to add 1 hence i use np.log

        show_data = show_data.loc[show_data[target_] > 0 ]

        sns.distplot(np.log(show_data[target_]) , color = color);

        plt.title(f'{target_} distribution when {feature} in {feature_value}');
pd.DataFrame((train_data.loc[train_data['SN_filter'] == 1, target_cols] < 0 ).value_counts(normalize = True)).head() * 100
feature = 'structure'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (22, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} distribution when {feature} in {feature_value}');
feature = 'predicted_loop_type'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (37, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} | {feature} in {feature_value}');
total_lag = 3

for lag in range(1, total_lag + 1) : 

    train_data[f'sequence_lag_{lag}'] = train_data.groupby('mol_id')[['sequence']].shift(lag)

    train_data[f'sequence_lag_{lag}'] = train_data[f'sequence_lag_{lag}'].fillna('Z') # Z for null
feature = 'sequence_lag_1'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (25, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} | {feature} in {feature_value}');
feature = 'sequence_lag_2'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (25, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} | {feature} in {feature_value}');
feature = 'sequence_lag_3'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (25, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} | {feature} in {feature_value}');
total_lead = 3

for lead in range(1, total_lead + 1) : 

    train_data[f'sequence_lead_{lead}'] = train_data.groupby('mol_id')[['sequence']].shift(-1 * lead)

    train_data[f'sequence_lead_{lead}'] = train_data[f'sequence_lead_{lead}'].fillna('Z') # Z for null
feature = 'sequence_lead_1'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (25, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} | {feature} in {feature_value}');
feature = 'sequence_lead_2'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (25, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} | {feature} in {feature_value}');
feature = 'sequence_lead_3'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (25, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} | {feature} in {feature_value}');
pd.DataFrame(train_data[['sequence_lag_3', 'sequence_lag_2', 'sequence_lag_1' , 'sequence', 

                        'sequence_lead_1', 'sequence_lead_2', 'sequence_lead_3',]].value_counts().head(10))
pd.DataFrame(train_data[['sequence', 'structure']].value_counts())
sequence_ = 'A'

feature = 'structure'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (22, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        show_data = show_data.loc[show_data['sequence'] == sequence_]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} distribution when {feature} {sequence_} in {feature_value}');
sequence_ = 'C'

feature = 'structure'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (22, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        show_data = show_data.loc[show_data['sequence'] == sequence_]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} distribution when {feature} {sequence_} in {feature_value}');
sequence_ = 'G'

feature = 'structure'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (22, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        show_data = show_data.loc[show_data['sequence'] == sequence_]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} distribution when {feature} {sequence_} in {feature_value}');
sequence_ = 'U'

feature = 'structure'

feature_values = [[val] for val in sorted(train_data[feature].unique())]

fig, ax = plt.subplots(figsize = (22, 27))

colors = ['b', 'g', 'r', 'c' , 'm'] 

for target_idx, target_ in enumerate(target_cols): 

    color = colors[target_idx]

    for feature_idx, feature_value in enumerate(feature_values): 

        plt_idx = target_idx * len(feature_values) + feature_idx + 1

        plt.subplot(len(target_cols), len(feature_values), plt_idx);

        

        show_data = train_data.loc[train_data[feature].isin(feature_value)]

        show_data = show_data.loc[show_data['sequence'] == sequence_]

        # filter SN_filter == 1

        show_data = show_data.loc[show_data['SN_filter'] == 1]

        # remove outlier (1%) to more visible plot

        show_data = show_data.loc[(show_data[target_] > show_data[target_].quantile(0.01)) & 

                                  (show_data[target_] < show_data[target_].quantile(0.99))]

        

        sns.distplot(show_data[target_], color = color);

        plt.title(f'{target_} distribution when {feature} {sequence_} in {feature_value}');