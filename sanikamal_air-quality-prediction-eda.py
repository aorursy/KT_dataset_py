import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))

# load dataset

dataset = pd.read_csv('../input/TrainingData.csv', header=0)
dataset.head()
dataset.tail()
dataset.shape
# trim and transform to floats

values = dataset.values

data = values[:, 6:].astype('float32')
# summarize amount of missing data

total_missing = np.count_nonzero(np.isnan(data))

percent_missing = total_missing / data.size * 100

print('Total Missing: %d/%d (%.1f%%)' % (total_missing, data.size, percent_missing))
# split data into chunks

# split the dataset by 'chunkID', return a dict of id to rows

def to_chunks(values, chunk_ix=1):

    chunks = dict()

    # get the unique chunk ids

    chunk_ids = np.unique(values[:, chunk_ix])

    # group rows by chunk id

    for chunk_id in chunk_ids:

        selection = values[:, chunk_ix] == chunk_id

        chunks[chunk_id] = values[selection, :]

    return chunks
# plot distribution of chunk durations

def plot_chunk_durations(chunks):

    # chunk durations in hours

    chunk_durations = [len(v) for k,v in chunks.items()]

    # boxplot

    plt.subplot(2, 1, 1)

    plt.boxplot(chunk_durations)

    # histogram

    plt.subplot(2, 1, 2)

    plt.hist(chunk_durations)

    # histogram

    plt.show()
# group data by chunks

values = dataset.values

chunks = to_chunks(values)

print('Total Chunks: %d' % len(chunks))
plt.figure(figsize=(15,8))

# plot chunk durations

plot_chunk_durations(chunks)
# plot chunks that do not have all data

def plot_discontiguous_chunks(chunks, row_in_chunk_ix=2):

    n_steps = 8 * 24

    for c_id,rows in chunks.items():

        # skip chunks with all data

        if rows.shape[0] == n_steps:

            continue

        # create empty series

        series = [np.nan for _ in range(n_steps)]

        # mark all rows with data

        for row in rows:

            # convert to zero offset

            r_id = row[row_in_chunk_ix] - 1

            # mark value

            series[r_id] = c_id

            # plot

        plt.plot(series)

    plt.show()
# group data by chunks

values = dataset.values

chunks = to_chunks(values)

# plot discontiguous chunks

plt.figure(figsize=(15,8))

plot_discontiguous_chunks(chunks)
# plot distribution of chunk start hour

def plot_chunk_start_hour(chunks, hour_in_chunk_ix=5):

    # chunk start hour

    chunk_start_hours = [v[0, hour_in_chunk_ix] for k,v in chunks.items() if len(v)==192]

    # boxplot

    plt.subplot(2, 1, 1)

    plt.boxplot(chunk_start_hours)

    # histogram

    plt.subplot(2, 1, 2)

    plt.hist(chunk_start_hours, bins=24)

    # histogram

    plt.show()
# group data by chunks

values = dataset.values

chunks = to_chunks(values)

# plot distribution of chunk start hour

plt.figure(figsize=(15,8))

plot_chunk_start_hour(chunks)
# plot all inputs for one or more chunk ids

def plot_chunk_inputs(chunks, c_ids):

    plt.figure(figsize=(15,8))

    inputs = range(6, 56)

    for i in range(len(inputs)):

        ax = plt.subplot(len(inputs), 1, i+1)

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        column = inputs[i]

        for chunk_id in c_ids:

            rows = chunks[chunk_id]

            plt.plot(rows[:,column])

    plt.show()
# group data by chunks

values = dataset.values

chunks = to_chunks(values)

# plot inputs for some chunks

plot_chunk_inputs(chunks, [1])
plot_chunk_inputs(chunks, [1, 3 ,5])
# boxplot for input variables for a chuck

def plot_chunk_input_boxplots(chunks, c_id):

    rows = chunks[c_id]

    plt.boxplot(rows[:,6:56])

    plt.show()
# group data by chunks

values = dataset.values

chunks = to_chunks(values)

# boxplot for input variables

plt.figure(figsize=(15,8))

plot_chunk_input_boxplots(chunks, 1)
# plot all targets for one or more chunk ids

def plot_chunk_targets(chunks, c_ids):

    plt.figure(figsize=(15,8))

    targets = range(56, 95)

    for i in range(len(targets)):

        ax = plt.subplot(len(targets), 1, i+1)

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        column = targets[i]

        for chunk_id in c_ids:

            rows = chunks[chunk_id]

            plt.plot(rows[:,column])

    plt.show()

    
# group data by chunks

values = dataset.values

chunks = to_chunks(values)

# plot targets for some chunks

plot_chunk_targets(chunks, [1])
# plot targets for some chunks

plot_chunk_targets(chunks, [1, 3 ,5])
# boxplot for target variables for a chuck

def plot_chunk_targets_boxplots(chunks, c_id):

    rows = chunks[c_id]

    plt.boxplot(rows[:,56:])

    plt.show()
# group data by chunks

values = dataset.values

chunks = to_chunks(values)

# boxplot for target variables

plt.figure(figsize=(15,8))

plot_chunk_targets_boxplots(chunks, 1)
# boxplot for all target variables

def plot_target_boxplots(values):

    plt.boxplot(values[:,56:])

    plt.show()

# boxplot for target variables

values = dataset.values

plt.figure(figsize=(15,8))

plot_target_boxplots(values)
# bar chart of the ratio of missing data per column

def plot_col_percentage_missing(values, ix_start=5):

    ratios = list()

    # skip early columns, with meta data or strings

    for col in range(ix_start, values.shape[1]):

        col_data = values[:, col].astype('float32')

        ratio = np.count_nonzero(np.isnan(col_data)) / len(col_data) * 100

        ratios.append(ratio)

        if ratio > 90.0:

            print(ratio)

    col_id = [x for x in range(ix_start, values.shape[1])]

    plt.bar(col_id, ratios)

    plt.show()
# plot ratio of missing data per column

values = dataset.values

plt.figure(figsize=(15,8))

plot_col_percentage_missing(values)
# plot distribution of targets for one or more chunk ids

def plot_chunk_targets_hist(chunks, c_ids):

    plt.figure(figsize=(15,8))

    targets = range(56, 95)

    for i in range(len(targets)):

        ax = plt.subplot(len(targets), 1, i+1)

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        column = targets[i]

        for chunk_id in c_ids:

            rows = chunks[chunk_id]

            # extract column of interest

            col = rows[:,column].astype('float32')

            # check for some data to plot

            if np.count_nonzero(np.isnan(col)) < len(rows):

                # only plot non-nan values

                plt.hist(col[~np.isnan(col)], bins=100)

    plt.show()
# group data by chunks

values = dataset.values

chunks = to_chunks(values)

# plot targets for some chunks

plot_chunk_targets_hist(chunks, [1])