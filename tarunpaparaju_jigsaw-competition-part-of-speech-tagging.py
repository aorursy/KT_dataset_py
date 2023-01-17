import os

import numpy as np



import pandas as pd

from tqdm import tqdm

tqdm.pandas()



from nltk import word_tokenize, pos_tag

from collections import Counter



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
DATA_LEN = 1804874

SAMPLE_SIZE = 100000
rows = np.arange(0, DATA_LEN)

skip_rows = list(np.random.choice(rows[1:], DATA_LEN - SAMPLE_SIZE, replace=False))
data = pd.read_csv('../input/train.csv', skiprows=skip_rows)
pos_tags = data['comment_text'].progress_apply(lambda x: pos_tag(word_tokenize(x)))

targets = data['target']
print(pos_tags[0])
tags = []

for i, tag in enumerate(pos_tags):

    pos_tags[i] = list(map(list, tag))

    tags.append(np.array(pos_tags[i])[:, 1])

flat_tags = np.concatenate([tag for tag in tags])
counts = [dict(Counter(tag)) for tag in tags]
def count_pos(tag_dict, tag_name):

    if tag_name in tag_dict:

        return tag_dict[tag_name]

    else:

        return 0
all_tags = set(flat_tags)

df = pd.DataFrame(np.zeros((SAMPLE_SIZE, 3)))

df.columns = ['count_dict', 'pos_feature', 'target']

df.count_dict = counts

df.target = targets
all_tags
def visualize_count_feature(tag):

    df.pos_feature = [count_pos(counts[i], tag) for i in range(SAMPLE_SIZE)]

    df.pos_feature = df.pos_feature.mask(df.pos_feature == 0, np.nan) # Ignore sample when tag not present in sentence



    fig, ax = plt.subplots(figsize=(10, 10))

    sns.distplot(ax=ax, a=[count for count in df.pos_feature.loc[df.target<0.5] if count==count], color='darkorange', label='non-toxic', hist=False)

    sns.distplot(ax=ax, a=[count for count in df.pos_feature.loc[df.target>0.5] if count==count], color='navy', label='toxic', hist=False)

    plt.title('" ' + tag + ' " ' + 'PoS tag count', fontsize=16, color='maroon')



    plt.show()
visualize_count_feature('CC')
visualize_count_feature('CD')
visualize_count_feature('DT')
visualize_count_feature('IN')
visualize_count_feature('POS')
visualize_count_feature('VBD')

visualize_count_feature('VBG')

visualize_count_feature('VBZ')

visualize_count_feature('VBP')
visualize_count_feature('UH')