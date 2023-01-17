import tensorflow as tf

import pandas as pd

import numpy as np



train_df = pd.read_csv("../input/train.csv")

train_df.head()

unique_contents = list(enumerate(np.unique(train_df['species'])))

species_dict = {name: i for i,name in unique_contents}

train_df['species'] = train_df['species'].map(lambda x: species_dict[x])

train_df.head()