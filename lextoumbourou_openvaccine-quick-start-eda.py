import os

from collections import Counter



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import mode



SEED = 420
train_df = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

test_df = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

sample_submission_df = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
one_example = train_df.iloc[0]

one_example.sequence
rotated_df = pd.DataFrame(list(one_example.sequence), columns=['sequence'])

rotated_df
one_example.structure
rotated_df_structure = pd.concat([

    rotated_df,

    pd.DataFrame(list(one_example.structure), columns=['structure'])

], axis=1)

rotated_df_structure.iloc[:10]
rotated_df_pred_loop = pd.concat([

    rotated_df_structure,

    pd.DataFrame(list(one_example.predicted_loop_type), columns=['predicted_loop_type'])

], axis=1)

rotated_df_pred_loop.iloc[:10]
pd.DataFrame(

    {

        'reactivity': list(one_example.reactivity),

        'deg_Mg_pH10': list(one_example.deg_Mg_pH10),

        'deg_pH10': list(one_example.deg_pH10),

        'deg_Mg_50C': list(one_example.deg_Mg_50C),

        'deg_50C': list(one_example.deg_50C)

    }

)
train_sequence_breakdown = Counter(''.join(list(train_df.sequence)))

test_sequence_breakdown = Counter(''.join(list(test_df.sequence)))



x_labels = ['G', 'A', 'C', 'U']



plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.title('Sequence character counts (train)')

plt.bar(x_labels, [train_sequence_breakdown[l] for l in x_labels])



plt.subplot(1, 2, 2)

plt.title('Sequence character counts (test)')

plt.bar(x_labels, [test_sequence_breakdown[l] for l in x_labels])



plt.show()
# This function takes a list of sequences and converts into a python `Counter` of ngram tokens



def get_ngrams_counters(sequences, n=2):

    output = Counter()

    for sequence in sequences:

        output += Counter([sequence[i:i+n] for i in range(len(sequence)-1)])

        

    return output
train_ngram_sequence = get_ngrams_counters(train_df.sequence)

test_ngram_sequence = get_ngrams_counters(test_df.sequence)



# Used to sort by frequency.

train_ngram_sequence = dict(train_ngram_sequence.most_common(10000))

test_ngram_sequence = dict(test_ngram_sequence.most_common(10000))



plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.title(f'Sequence character bigram counts (train) ({len(train_ngram_sequence)} unique bigrams)')

plt.bar(dict(train_ngram_sequence).keys(), dict(train_ngram_sequence).values())



plt.subplot(1, 2, 2)

plt.title(f'Sequence character bigram counts (test) ({len(test_ngram_sequence)} unique bigrams)')

plt.bar(dict(test_ngram_sequence).keys(), dict(test_ngram_sequence).values())



plt.show()
train_ngram_sequence = get_ngrams_counters(train_df.sequence, 3)

test_ngram_sequence = get_ngrams_counters(test_df.sequence, 3)



# Used to sort by frequency.

train_ngram_sequence = dict(train_ngram_sequence.most_common(10000))

test_ngram_sequence = dict(test_ngram_sequence.most_common(10000))



plt.figure(figsize=(25, 10))

plt.title(f'Sequence character trigram (train) ({len(train_ngram_sequence)} unique trigrams)')

plt.bar(dict(train_ngram_sequence).keys(), dict(train_ngram_sequence).values())

plt.xticks(rotation=45)



plt.figure(figsize=(25, 10))

plt.title(f'Sequence character trigram (test) ({len(test_ngram_sequence)} unique trigrams)')

plt.bar(dict(test_ngram_sequence).keys(), dict(test_ngram_sequence).values())

plt.xticks(rotation=45)



plt.show()
train_structure_breakdown = Counter(''.join(list(train_df.structure)))

test_structure_breakdown = Counter(''.join(list(test_df.structure)))



x_labels = ['(', ')', '.']



plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.title('Structure character counts (train)')

plt.bar(x_labels, [train_structure_breakdown[l] for l in x_labels])



plt.subplot(1, 2, 2)

plt.title('Structure character counts (test)')

plt.bar(x_labels, [test_structure_breakdown[l] for l in x_labels])



plt.show()
def get_paired_tokens(*sequences):

    output = Counter()

    for seq_chars in zip(*sequences):

        for i in range(len(seq_chars[0])-1):

            new_token = ''

            for seq_char in seq_chars:

                new_token += ' '+seq_char[i]

            output += Counter([new_token])

        

    return output
train_sequence_structure_pairs = get_paired_tokens(train_df.sequence, train_df.structure)

test_sequence_structure_pairs = get_paired_tokens(train_df.sequence, train_df.structure)



train_sequence_structure_pairs = dict(train_sequence_structure_pairs.most_common(1000))

test_sequence_structure_pairs = dict(test_sequence_structure_pairs.most_common(1000))



plt.figure(figsize=(25, 10))

plt.title(f'Train sequence and structure character pairs dist')

plt.bar(train_sequence_structure_pairs.keys(), train_sequence_structure_pairs.values())

plt.xticks(rotation=45)



plt.figure(figsize=(25, 10))

plt.title(f'Test sequence and structure character pairs dist')

plt.bar(test_sequence_structure_pairs.keys(), test_sequence_structure_pairs.values())

plt.xticks(rotation=45)



plt.show()
train_predicted_loop_type_breakdown = Counter(''.join(list(train_df.predicted_loop_type)))

test_predicted_loop_type_breakdown = Counter(''.join(list(test_df.predicted_loop_type)))



train_predicted_loop_type_breakdown = dict(train_predicted_loop_type_breakdown.most_common(1000))

test_predicted_loop_type_breakdown = dict(test_predicted_loop_type_breakdown.most_common(1000))



x_labels = train_predicted_loop_type_breakdown.keys()



plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)

plt.title('Structure character counts (train)')

plt.bar(x_labels, [train_predicted_loop_type_breakdown[l] for l in x_labels])



plt.subplot(1, 2, 2)

plt.title('Structure character counts (test)')

plt.bar(x_labels, [test_predicted_loop_type_breakdown[l] for l in x_labels])



plt.show()
train_sequence_predicted_loop_pairs = get_paired_tokens(train_df.sequence, train_df.predicted_loop_type)

test_sequence_predicted_loop_pairs = get_paired_tokens(train_df.sequence, train_df.predicted_loop_type)



train_sequence_predicted_loop_pairs = dict(train_sequence_predicted_loop_pairs.most_common(1000))

test_sequence_predicted_loop_pairs = dict(test_sequence_predicted_loop_pairs.most_common(1000))



plt.figure(figsize=(25, 10))

plt.title(f'Train sequence and predicted loop character pairs dist')

plt.bar(train_sequence_structure_pairs.keys(), train_sequence_structure_pairs.values())

plt.xticks(rotation=45)



plt.figure(figsize=(25, 10))

plt.title(f'Test sequence and predicted loop character pairs dist')

plt.bar(test_sequence_structure_pairs.keys(), test_sequence_structure_pairs.values())

plt.xticks(rotation=45)



plt.show()
# This function performs some basic statististical analysis on the various labels.



def do_analysis(df, column_name):

    all_vals = [y for x in df[column_name] for y in x]

    print(f"Analysis across all samples for {column_name}")

    print(f'Mean: {np.mean(all_vals)}')

    print(f'Max: {np.max(all_vals)}')

    print(f'Min: {np.min(all_vals)}')

    print(f'Mode: {mode(all_vals).mode[0]}')

    print(f'STD: {np.std(all_vals)}')

    print()

    

    plt.hist(all_vals)

    plt.title(f'Histogram for {column_name} across all samples')

    plt.show()

    

    print("Statistics aggregated per sample")

    fig, axes = plt.subplots(1, 4, figsize=(15, 5), squeeze=False)



    df[column_name].apply(

        lambda x: np.mean(x)).plot(

            kind='hist',

            bins=50, ax=axes[0,0],

            title=f'Mean dist {column_name}')



    df[column_name].apply(

        lambda x: np.max(x)).plot(

            kind='hist',

            bins=50, ax=axes[0,1],

            title=f'Max dist {column_name}')



    df[column_name].apply(

        lambda x: np.min(x)).plot(

            kind='hist',

            bins=50, ax=axes[0,2],

            title=f'Min dist {column_name}')

    df[column_name].apply(

        lambda x: np.std(x)).plot(

            kind='hist',

            bins=50, ax=axes[0,3],

            title=f'Std {column_name}')

    plt.show()
do_analysis(train_df, 'reactivity')
do_analysis(train_df, 'deg_Mg_pH10')
do_analysis(train_df, 'deg_50C')
do_analysis(train_df, 'deg_Mg_50C')
bpps_files = os.listdir('../input/stanford-covid-vaccine/bpps/')

example_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_files[0]}')

print('bpps file shape:', example_bpps.shape)
plt.style.use('default')

fig, axs = plt.subplots(5, 5, figsize=(15, 15))

axs = axs.flatten()

for i, f in enumerate(bpps_files):

    if i == 25:

        break

    example_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{f}')

    axs[i].imshow(example_bpps)

    axs[i].set_title(f)

plt.tight_layout()

plt.show()