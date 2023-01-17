import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import os

PATH = "../input/stanford-covid-vaccine/"



test = pd.read_json(os.path.join(PATH, 'test.json'), lines=True)

sample_submission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

train = pd.read_json(os.path.join(PATH, 'train.json'), lines=True)

print(f"Train set length before filtering: {len(train)}")



# https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183992

train = train[train.SN_filter == 1]

print(f"Train set length after filtering: {len(train)}")
submission = pd.DataFrame(columns=[ 'id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C'])



for i in range(len(test)):

    df = test.loc[i]



    dummy_values = np.zeros(df.seq_length)

    dummy_values[:df.seq_scored]  = np.full(df.seq_scored, 0.5) # Dummy value



    new_df = pd.DataFrame(data={'id': df.id, 'pos': list(range(df.seq_length)), 

                                'reactivity': dummy_values, # Dummy value

                                'deg_Mg_pH10': dummy_values, # Dummy value

                                'deg_pH10': np.zeros(df.seq_length),# not scored

                                'deg_Mg_50C': dummy_values, # Dummy value

                                'deg_50C': np.zeros(df.seq_length), # not scored

                               })



    new_df['id_seqpos'] = new_df.apply(lambda x: f"{x['id']}_{x['pos']}", axis=1)

    new_df = new_df.drop(['id', 'pos'], axis=1)

    new_df = new_df[[ 'id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']] 

    submission = submission.append(new_df)



submission = submission.reset_index(drop=True)

print(f"Length of sample submission ({len(sample_submission)}) == Length of submission file ({len(submission)})")

submission.to_csv('submission.csv', index = False)

submission.head()
submission[65:70]
train = train[['id', 'sequence', 'structure', 'predicted_loop_type',

       'signal_to_noise', 'seq_length', 'seq_scored',

       'reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',

       'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 'deg_Mg_pH10',

       'deg_pH10', 'deg_Mg_50C', 'deg_50C']].reset_index(drop=True)



train['train_data'] = 1

test = test[['id', 'sequence', 'structure', 'predicted_loop_type', 'seq_length', 'seq_scored']]

test['train_data'] = 0

all_data = train.append(test).reset_index(drop=True)



for c in ['.','(',')']:

    all_data[f"cnt_structure_{c}"] = all_data.apply(lambda x: ((list(x['structure']).count(str(c)))/(x['seq_length'])), axis = 1)



for c in ['S','M','I','B','H','E','X']:

    all_data[f"cnt_loop_{c}"] = all_data.apply(lambda x: ((list(x['predicted_loop_type']).count(str(c)))/(x['seq_length'])), axis = 1)



    

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    

correlation_matrix = all_data[['cnt_structure_.', 'cnt_structure_(', 'cnt_structure_)', 'cnt_loop_S', 'cnt_loop_M', 'cnt_loop_I', 'cnt_loop_B', 'cnt_loop_H', 'cnt_loop_E', 'cnt_loop_X']].corr()

matrix = np.triu(correlation_matrix)

sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot = True, cmap='coolwarm', mask=matrix, ax=ax[0])



correlation_matrix = all_data[['cnt_structure_.', 'cnt_structure_(', 'cnt_structure_)', 'cnt_loop_S']].corr()

matrix = np.triu(correlation_matrix)

sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot = True, cmap='coolwarm', mask=matrix, ax=ax[1])



plt.show()
temp = all_data.groupby(['train_data', 'seq_length' ]).id.count().to_frame().reset_index()

temp.columns = ['train_data', 'seq_length', 'counts']

temp.train_data = temp.train_data.replace({0:'Test', 1:'Train'})

temp = temp.pivot(index='train_data',columns='seq_length')

temp.columns = ['seq_length_107', 'seq_length_130']



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

temp.plot(kind='bar', stacked=True, ax= ax,  fontsize=14)

plt.xlabel('Dataset', fontsize=14)

plt.ylabel('Counts', fontsize=14)

plt.show()