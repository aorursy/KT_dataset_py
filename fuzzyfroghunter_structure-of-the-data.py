!ls ../input
TRAIN_DATA = '../input/en_train.csv'

TEST_DATA = '../input/en_test.csv'
import pandas as pd
train_data = pd.read_csv(TRAIN_DATA)

test_data = pd.read_csv(TEST_DATA)
train_data.iloc[:5]
train_data.sample(5)
test_data.iloc[:5]
test_data.sample(5)
nontrivial_train_data = train_data[train_data.before != train_data.after]
proportion_nontrivial = nontrivial_train_data.shape[0]/train_data.shape[0]

print(proportion_nontrivial)
1 - proportion_nontrivial
train_data[train_data['class'] == 'PLAIN'].sample(1)
train_data[(train_data['class'] == 'LETTERS')].sample(1)
CLASSES = sorted(list(train_data['class'].unique()))

print(CLASSES)
len(CLASSES)
grouped_by_class = train_data.groupby('class')
def proportion_nontrivial(df):

    """

    Args:

    1. df - Dataframe with 'before' and 'after' columns

    

    Returns:

    Proportion of rows in dataframe for which 'before' is not equal to 'after'

    """

    nontrivial = df[df.before != df.after]

    return nontrivial.shape[0]/df.shape[0]
class_nontriviality = [(key, proportion_nontrivial(group)) for key, group in grouped_by_class]
print('Proportion of rows for which nontrivial normalization is required (by class):\n')

for key, s in class_nontriviality:

    print('{} - {}'.format(key, s))
class_weights = [(key, group.shape[0]/train_data.shape[0]) for key, group in grouped_by_class]
sorted_class_weights = sorted(class_weights, key=lambda p: -p[1])
print('Proportion of the training data made up by each class (sorted in descending order of weight):\n')

for key, weight in sorted_class_weights:

    print('{} - {}'.format(key, weight))
nontriviality_dict = dict(class_nontriviality)

weight_dict = dict(class_weights)
total_nontriviality = sum([nontriviality_dict[k]*weight_dict[k] for k in nontriviality_dict])
total_nontriviality
total_nontriviality == proportion_nontrivial(train_data)
from collections import Counter
def generate_recorder_fn(counter):

    def recorder_fn(iterable):

        for c in iterable:

            counter[c] += 1

    return recorder_fn
char_counter = Counter()



_ = train_data['before'].astype(str).apply(generate_recorder_fn(char_counter))
len(char_counter)
char_counter.most_common(20)
import matplotlib.pyplot as plt

import numpy as np
labels, values = zip(*char_counter.most_common(50))
indexes = np.arange(len(labels))

width = 0.2



plt.bar(indexes, values, width)

plt.xticks(indexes + width * 0.5, labels)

plt.xlabel('characters')

plt.ylabel('counts')

plt.show()
def n_grams(string, n):

    return zip(*(string[k:] for k in range(n)))
def n_gram_frequency(strings, n=1):

    counter = Counter()

    record = generate_recorder_fn(counter)

    for string in strings:

        record(n_grams(string, n))

    return counter
class_char_counters = {k:n_gram_frequency(df['before'].astype(str), 1) for k,df in grouped_by_class}
sum(class_char_counters[k][('e',)] for k in class_char_counters)
sum(class_char_counters[k][('e',)] for k in class_char_counters) == char_counter['e']
bigram_counter = n_gram_frequency(train_data['before'].astype(str), 2)
len(bigram_counter)
bigram_counter.most_common(20)
def plot_frequencies(counter, n, width=0.5, font_size=5):

    labels, values = zip(*counter.most_common(n))

    indexes = np.arange(len(labels))

    width = 0.5

    

    plt.bar(indexes, values, width)

    plt.xticks(indexes + width * 0.5, labels, rotation='vertical', fontsize=font_size)

    plt.xlabel('grams')

    plt.ylabel('counts')

    plt.show()
plot_frequencies(bigram_counter, 80)
class_bigram_counters = {k:n_gram_frequency(df['before'].astype(str), 2) for k,df in grouped_by_class}
sum(class_bigram_counters[k][('a','l')] for k in class_bigram_counters)
sum(class_bigram_counters[k][('a','l')] for k in class_bigram_counters) == bigram_counter[('a','l')]
def freq_to_dist(counter):

    total = sum(counter[k] for k in counter)

    return {k:counter[k]/total for k in counter}
class_char_dists = {c:freq_to_dist(class_char_counters[c]) for c in CLASSES}
class_char_dists_sums = {c:sum(class_char_dists[c][k] for k in class_char_dists[c]) for c in CLASSES}
sum(abs(class_char_dists_sums[c] - 1) < 0.000001 for c in CLASSES) == len(CLASSES)
class_bigram_dists = {c:freq_to_dist(class_bigram_counters[c]) for c in CLASSES}
def l1_distance(dist1, dist2):

    """

    Args:

    1. dist1 - a probability distribution represented as a Python dictionary

    2. dist2 - a probability distribution represented as a Python dictionary

    (Note: Dictionary representation of probability distributions is as {value:probability for value in universe})

    

    Returns:

    L^1 distance between dist1 and dist2

    """

    keys = set(dist1) | set(dist2)

    return sum(abs(dist1.get(k,0) - dist2.get(k, 0)) for k in keys)
distribution = class_char_dists['PLAIN']
l1_distance(distribution, distribution) == 0
dist1 = class_char_dists['PLAIN']
dist2 = class_char_dists['VERBATIM']
abs(l1_distance(dist1, dist2) - l1_distance(dist2, dist1)) < 0.000001
char_dist_distances_dict = {c1:{c2:l1_distance(class_char_dists[c1], class_char_dists[c2]) for

                           c2 in CLASSES} for

                       c1 in CLASSES}
char_dist_distances = pd.DataFrame.from_dict(char_dist_distances_dict)
bigram_dist_distances_dict = {c1:{c2:l1_distance(class_bigram_dists[c1], class_bigram_dists[c2]) for

                             c2 in CLASSES} for

                         c1 in CLASSES}
bigram_dist_distances = pd.DataFrame.from_dict(bigram_dist_distances_dict)
import seaborn as sns
sns.heatmap(char_dist_distances)

plt.show()
sns.heatmap(bigram_dist_distances)

plt.show()
electronic = [df for k,df in grouped_by_class if k == 'ELECTRONIC'][0]
electronic.sample(5)
SAMPLE_SIZE = 1000
sample_df = train_data.sample(SAMPLE_SIZE)[['before','class']]
sample = list(sample_df['before'])
labels = list(sample_df['class'])
def char_inferred_class(token):

    token_dist = freq_to_dist(n_gram_frequency([token], 1))

    distances = [(c, l1_distance(token_dist, class_char_dists[c])) for c in CLASSES]

    return min(distances, key=lambda p: p[1])[0]
inferences = [char_inferred_class(token) for token in sample]
accuracy = sum(inferred == actual for inferred, actual in zip(inferences, labels))/len(labels)
accuracy
class_samples = {c:list(train_data[train_data['class'] == c].sample(SAMPLE_SIZE, replace=True)['before'].astype(str)) for

                 c in CLASSES}
class_inferences = {c:[char_inferred_class(token) for token in class_samples[c]] for c in CLASSES}
class_accuracies = {c:sum(i==c for i in class_inferences[c])/SAMPLE_SIZE for c in CLASSES}
class_accuracies
inferred_class_confusion = {c:[] for c in CLASSES}



for c in CLASSES:

    for i in class_inferences[c]:

        inferred_class_confusion[i].append(c)
inferred_class_accuracies = {c:(sum(l==c for l in inferred_class_confusion[c])/len(inferred_class_confusion[c]),

                                len(inferred_class_confusion[c])) for

                             c in inferred_class_confusion if len(inferred_class_confusion[c]) > 0}
inferred_class_accuracies
confusion = {c:Counter(class_inferences[c]) for c in CLASSES}
for c in confusion:

    for k in CLASSES:

        if not k in confusion[c]:

            confusion[c][k] = 0
sns.heatmap(pd.DataFrame.from_dict(confusion))

plt.xlabel('Inferred labels')

plt.ylabel('True labels')

plt.show()