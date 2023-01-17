import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

import plotly.graph_objs as go



%matplotlib inline
submission = pd.read_csv('../input/sample_submission.csv')
label_names = ['yes', 'no', 'up', 'down', 'left', 'right', 

               'on', 'off', 'stop', 'go', 'silence', 'unknown']

len(label_names)
new_submission_path_list = []

for label in label_names:

    submission.label = label

    new_submission_path = 'all_%s.csv.gz' % label

    new_submission_path_list.append(new_submission_path)

    submission.to_csv(new_submission_path, index=False, compression='gzip')
for label, new_submission_path in zip(label_names, new_submission_path_list):

    print(new_submission_path)

    submission = pd.read_csv(new_submission_path)

    unique_labels = submission.label.unique()

    assert len(unique_labels) == 1

    assert unique_labels[0] == label

    print(submission.label.unique())

    print()
score_dict = {label:None for label in label_names}

score_dict
score_dict = {

    'down': 0.06,

    'go': 0.08,

    'left': 0.08,

    'no': 0.07,

    'off': 0.07,

    'on': 0.08,

    'right': 0.07,

    'silence': 0.09,

    'stop': 0.08,

    'unknown': 0.09,

    'up': 0.08,

    'yes': 0.08

}
label_scores = [score_dict[label] for label in label_names]

np.sum(label_scores)
plt.figure(figsize=(15, 6))

plt.title('Frequency of the labels on the public test set')

plt.bar(np.arange(len(label_scores)), label_scores, tick_label=label_names);
plt.figure(figsize=(15, 6))

sns.distplot(label_scores)

plt.title('Frequency histogram')

plt.xlabel('Frequency');