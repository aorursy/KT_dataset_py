import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import seaborn as sns

from IPython.display import YouTubeVideo

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# print(check_output(["ls", "../input/video_level"]).decode("utf8"))
labels_df = pd.read_csv('../input/labldata1/label_names.csv')

filenames = ["../input/youtube8m-2019/frame-sample/frame/train01.tfrecord".format(i) for i in range(10)]

print("we have {} unique labels in the dataset".format(len(labels_df['label_name'].unique())))
labels_df = pd.read_csv('../input/labldata1/label_names.csv')

labels = []

textual_labels = []

textual_labels_nested = []

filenames = ["../input/youtube8m-2019/frame-sample/frame/train01.tfrecord".format(i) for i in range(10)]

total_sample_counter = 0



label_counts = []



for filename in filenames:

    for example in tf.python_io.tf_record_iterator(filename):

        total_sample_counter += 1

        tf_example = tf.train.Example.FromString(example)



        label_example = list(tf_example.features.feature['labels'].int64_list.value)

        label_counts.append(len(label_example))

        labels = labels + label_example

        label_example_textual = list(labels_df[labels_df['label_id'].isin(label_example)]['label_name'])

        textual_labels_nested.append(set(label_example_textual))

        textual_labels = textual_labels + label_example_textual

        if len(label_example_textual) != len(label_example):

            print('label names lookup failed: {} vs {}'.format(label_example, label_example_textual))



print('label ids missing from label_names.csv: {}'.format(sorted(set(labels) - set(labels_df['label_id']))))

print('Found {} samples in all of the 10 available tfrecords'.format(total_sample_counter))
sns.distplot(label_counts, kde=False)

plt.title('distribution of number of labels')


def grouped_data_for(l):

    # wrap the grouped data into dataframe, since the inner is pd.Series, not what we need

    l_with_c = pd.DataFrame(

        pd.DataFrame({'label': l}).groupby('label').size().rename('n')

    ).sort_values('n', ascending=False).reset_index()

    return l_with_c
N = 20



textual_labels_with_counts_all = grouped_data_for(textual_labels)



sns.barplot(y='label', x='n', data=textual_labels_with_counts_all.iloc[0:N, :])

plt.title('Top {} labels with sample count'.format(N))
two_element_labels = ['|'.join(sorted(x)) for x in textual_labels_nested if len(x) == 2]



N = 20



textual_labels_with_counts = grouped_data_for(two_element_labels)



sns.barplot(y='label', x='n', data=textual_labels_with_counts.iloc[0:N, :])

plt.title('Top {} labels with sample count from only samples with two labels'.format(N))
two_element_labels = ['|'.join(sorted(x)) for x in textual_labels_nested if len(x) == 3]



N = 20



textual_labels_with_counts = grouped_data_for(two_element_labels)



sns.barplot(y='label', x='n', data=textual_labels_with_counts.iloc[0:N, :])

plt.title('Top {} labels with sample count from only samples with three labels'.format(N))
top_50_labels = list(textual_labels_with_counts_all['label'][0:50].values)

top_50_labels
label_group_counts = []

labels_for_group_counts = []

for label in top_50_labels:

    # this is a list of lists and

    # for each of the inner lists we want to know how many elements there are

    nested_labels_with_label = [x for x in textual_labels_nested if label in x]

    group_counts = [len(x) for x in nested_labels_with_label]

    label_group_counts = label_group_counts + group_counts

    labels_for_group_counts = labels_for_group_counts + [label]*len(group_counts)



count_df = pd.DataFrame({'label': labels_for_group_counts, 'group_size': label_group_counts})

count_df.head()
plt.figure(figsize=(12,8))

sns.barplot(y='label', x='group_size', data=count_df)

plt.title('avg number of labels per top 50 categories')
bins = [0, 1, 3, 14]

colors = ['r', 'g', 'b', 'y']



plt.figure(figsize=(12,8))

for bin, color in zip(bins, colors):

    sns.distplot(count_df[count_df['label'] == top_50_labels[bin]]['group_size'], kde=True, color=color)



plt.legend([top_50_labels[bin] for bin in bins])
K_labels = []



for i in top_50_labels:

    row = []

    for j in top_50_labels:

        # find all records that have label `i` in them

        i_occurs = [x for x in textual_labels_nested if i in x]

        # how often does j occur in total in them?

        j_and_i_occurs = [x for x in i_occurs if j in x]

        k = 1.0*len(j_and_i_occurs)/len(i_occurs)

        row.append(k)

    K_labels.append(row)



K_labels = np.array(K_labels)

K_labels = pd.DataFrame(K_labels)

K_labels.columns = top_50_labels

K_labels.index = top_50_labels
K_labels.head()
plt.figure(figsize=(12,8))

sns.heatmap(K_labels)

# probability of observing column label given row label

plt.title('P(column|row)')
filenames = ["../input/youtube8m-2019/frame-sample/frame/train01.tfrecord".format(i) for i in range(10)]



cosmetics = []

games = []

car = []

vehicle = []

animal = []



for filename in filenames:

    for example in tf.python_io.tf_record_iterator(filename):

        tf_example = tf.train.Example.FromString(example)

        label_example = list(tf_example.features.feature['labels'].int64_list.value)

        label_example_textual = list(labels_df[labels_df['label_id'].isin(label_example)]['label_name'])

        mean_mean_rgb = np.mean(tf_example.features.feature['mean_rgb'].float_list.value)

        for label in label_example_textual:

            if label == 'Cosmetics':

                cosmetics.append(mean_mean_rgb)

            elif label == 'Games':

                games.append(mean_mean_rgb)

            elif label == 'Car':

                car.append(mean_mean_rgb)

            elif label == 'Vehicle':

                vehicle.append(mean_mean_rgb)

            elif label == 'Animal':

                animal.append(mean_mean_rgb)
len(cosmetics), len(games), len(car), len(vehicle), len(animal)
df_submission = pd.read_csv('../input/youtube8m-2019/sample_submission.csv')

df_submission.head()

df_submission.to_csv('submission.csv', index=False)