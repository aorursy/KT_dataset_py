import warnings, re



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

pd.options.mode.chained_assignment = None

# dir(pd.options.display)

warnings.simplefilter(action='ignore', category=FutureWarning)



plt.style.use('ggplot')
train = pd.read_csv('../input/birdsong-recognition/train.csv')

print(train.shape)

train.head(3)
train['background'].value_counts(dropna=True, sort=True).to_frame().head(20).plot.bar(

    color='deeppink', figsize=(15, 5)

);
def get_multi_label(s, bird_l):

    if type(s) != str: s = str(s)

    return [b in re.sub(r' \([^()]*\)', '', s).split('; ') for b in bird_l]
bird_l = train.species.unique().tolist()
# default label

label_df = pd.get_dummies(train.species).set_index(train.xc_id)

label_df.head(3)
%%time

background_arr = np.array([get_multi_label(r, bird_l) for r in train.background])

background_arr.shape
label_df.iloc[:, :] = label_df.values + background_arr

label_df['label_n'] = label_df.sum(axis=1)
label_df.label_n.value_counts().plot.bar(figsize=(10, 5), color='deeppink');
label_df.drop('label_n', axis=1).to_csv('multi-label.csv')