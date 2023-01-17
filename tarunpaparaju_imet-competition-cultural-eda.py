import os

import gc

import numpy as np

import pandas as pd

from tqdm import tqdm

tqdm.pandas()

from collections import Counter

from operator import itemgetter

import scipy

import cv2

from cv2 import imread

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
DATA_LEN = 109237

SAMPLE_SIZE= 5000
rows = np.arange(0, DATA_LEN)

select_rows = np.random.choice(rows, SAMPLE_SIZE, replace=False)
train_images = []

image_dirs = np.take(os.listdir('../input/train'), select_rows)



for image_dir in tqdm(sorted(image_dirs)):

    image = imread('../input/train/'+image_dir)

    train_images.append(image)

    del image

    gc.collect()

    

train_images = np.array(train_images)
train_df = pd.read_csv('../input/train.csv')

targets_df = train_df.loc[(train_df.id.apply(lambda x: x + '.png')).isin(image_dirs)]
labels_df = pd.read_csv('../input/labels.csv')

label_dict = dict(zip(labels_df.attribute_id, labels_df.attribute_name))

for key in label_dict:

    if 'culture' in label_dict[key]:

        label_dict[key] = label_dict[key][9:]

    if 'tag' in label_dict[key]:

        label_dict[key] = label_dict[key][5:]
train_targets = []



for targets in targets_df.attribute_ids:

    target = targets.split()

    target = list(map(lambda x: label_dict[int(x)], target))

    train_targets.append(target)

    

train_targets = np.array(train_targets)
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(50, 50))

count = 0



for i in range(4):

    for j in range(4):

        ax[i, j].imshow(cv2.cvtColor(train_images[count], cv2.COLOR_BGR2RGB))

        ax[i, j].set_title(str(train_targets[count]), fontsize=24)

        count = count + 1
def display_culture_examples(culture, fraction=1, max_count=5):

    count = 0

    for i in range(int(len(train_images)*fraction)):

        if count == max_count:

            break

        if culture in train_targets[i]:

            fig, ax = plt.subplots(figsize=(3, 3))

            ax.imshow(cv2.cvtColor(train_images[i], cv2.COLOR_BGR2RGB))

            plt.title(train_targets[i], fontsize=8)

            plt.show()

            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

            count = count + 1

            

def show_common_tags(culture, palette):

    tags = []

    for i in range(len(train_images)):

        if culture in train_targets[i]:

            tags.extend(train_targets[i])

    tag_counts = Counter(tags)

    size = len(tag_counts.keys())

    tags_df = pd.DataFrame(np.zeros((size, 2)))

    tags_df.columns = ['tag', 'count']

    tags_df['tag'] = list(tag_counts.keys())

    tags_df['count'] = list(tag_counts.values())

    tags_df = tags_df.sort_values(by=['count'], ascending=False)

    tags_df = tags_df.reset_index()

    del tags_df['index']

    tag_counts = sorted(tag_counts.items(), key = lambda kv:(kv[1], kv[0]))

    tags = reversed(list(tag_counts)[-6:-1])

    

    print("MOST COMMON " + culture.upper() + " TAGS")

    print("")

    for i, tag in enumerate(tags_df[1:6].tag):

        print(str(i+1) + '). ' + tag)

    

    sns.set_context(rc={'xtick.labelsize': 8})

    fig, ax = plt.subplots(figsize=(12, 5))

    plot = sns.barplot(x='tag', y='count', data=tags_df.loc[1:6], palette=palette)

    plt.title('Common ' + culture[0].upper() + culture[1:] + ' tags')

    plt.show()

    sns.reset_defaults()
display_culture_examples('french')
show_common_tags('french', palette=['blue', 'blue', 'navajowhite', 'navajowhite', 'red', 'red'])
display_culture_examples('spanish')
show_common_tags('spanish', palette=['red', 'red', 'yellow', 'yellow', 'red', 'red'])
display_culture_examples('italian')
show_common_tags('italian', palette=['forestgreen', 'forestgreen', 'navajowhite', 'navajowhite', 'crimson', 'crimson'])
display_culture_examples('german')
show_common_tags('german', palette=['black', 'black', 'red', 'red', 'yellow' ,'yellow'])
display_culture_examples('russian')
show_common_tags('russian', palette=['navajowhite', 'navajowhite', 'blue', 'blue', 'red', 'red'])
display_culture_examples('danish')
show_common_tags('danish', palette=['red', 'navajowhite', 'red', 'navajowhite', 'red', 'navajowhite'])
display_culture_examples('swedish')
show_common_tags('swedish', palette=['blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow'])
display_culture_examples('iran')
show_common_tags('iran', palette=['limegreen', 'limegreen', 'navajowhite', 'navajowhite', 'red', 'red'])
display_culture_examples('syrian')
show_common_tags('syrian', palette=['red', 'red', 'green', 'black', 'black'])
display_culture_examples('assyrian')
show_common_tags('assyrian', palette=['red', 'red', 'navajowhite', 'navajowhite', 'red', 'red'])
display_culture_examples('india')
show_common_tags('india', palette=['darkorange', 'darkorange', 'navajowhite', 'navajowhite', 'green', 'green'])
display_culture_examples('china')
show_common_tags('china', palette=['gold', 'red', 'red', 'red', 'red', 'gold'])
display_culture_examples('japan')
show_common_tags('japan', palette=['navajowhite', 'navajowhite', 'red', 'red', 'navajowhite', 'navajowhite'])
display_culture_examples('vietnam')
show_common_tags('vietnam', palette=['red', 'red', 'yellow', 'yellow', 'red', 'red'])
display_culture_examples('thailand')
show_common_tags('thailand', palette=['crimson', 'navajowhite', 'darkblue', 'darkblue', 'navajowhite', 'crimson'])
display_culture_examples('egyptian')
show_common_tags('egyptian', palette=['red', 'red', 'navajowhite', 'navajowhite', 'black', 'black'])
display_culture_examples('american')
show_common_tags('american', palette=['darkblue', 'navajowhite', 'red', 'red', 'navajowhite', 'darkblue'])
display_culture_examples('mexican')
show_common_tags('mexican', palette=['green', 'green', 'navajowhite', 'navajowhite', 'red', 'red'])