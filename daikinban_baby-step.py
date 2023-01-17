import numpy as np
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv(
    '../input/tuning_labels.csv', header=None, names=['img_id', 'labels'])
description = pd.read_csv('../input/class-descriptions.csv')

label_lookup = {label: descr for label, descr in zip(
    description.label_code.values, description.description.values)}
import csv

label_count = {}
with open('../input/tuning_labels.csv') as f:
    csv_rdr = csv.reader(f)
    for row in csv_rdr:
        inst_labels = row[1].split()
        label_count = {**label_count, **{l: label_count.get(l, 0) + 1 for l in inst_labels}}
top_label_name, top_label_count = zip(*[
    l for l in sorted(label_count.items(), reverse=True, key=lambda l: l[1])[:100]])
label_count_df = pd.DataFrame(data={'label': list(map(label_lookup.get, top_label_name)), 'count': top_label_count})
sns.set(rc={'figure.figsize':(12, 18)})
ax = sns.barplot(data=label_count_df, y='label', x='count')
label_idx = {l: i for i, l in enumerate(label_count)}
cooccurrence = np.zeros((len(label_idx), len(label_idx)))
with open('../input/tuning_labels.csv') as f:
    csv_rdr = csv.reader(f)
    for row in csv_rdr:
        inst_labels = row[1].split()
        cooccurrence[list(zip(*[(label_idx[l1], label_idx[l2]) for l1 in inst_labels for l2 in inst_labels]))] += 1
labels, _ = zip(*[
    l for l in sorted(label_count.items(), reverse=True, key=lambda l: l[1])])

most_freq_labels = labels[:30]
most_freq_label_name = [label_lookup[l] for l in most_freq_labels]
most_freq_label_idx = [label_idx[l] for l in most_freq_labels]
most_freq_label_cooccurrence = cooccurrence[:, most_freq_label_idx]

most_freq_label_corr = pd.DataFrame(data=most_freq_label_cooccurrence, columns=most_freq_label_name).corr()

less_freq_labels = labels[-30:]
less_freq_label_name = [label_lookup[l] for l in less_freq_labels]
less_freq_label_idx = [label_idx[l] for l in less_freq_labels]
less_freq_label_cooccurrence = cooccurrence[:, less_freq_label_idx]

less_freq_label_corr = pd.DataFrame(data=less_freq_label_cooccurrence, columns=less_freq_label_name).corr()
def plot_corr(corr):
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize = (12, 12))
    return sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=.3, center=0, 
        square=True, linewidths=.5, cbar_kws={"shrink": .5})
plot_corr(most_freq_label_corr)
plot_corr(less_freq_label_corr)




