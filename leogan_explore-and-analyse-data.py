import numpy as np 

import pandas as pd 

from tqdm import tqdm
data_path = 'data/'

subm_path = data_path + 'subm/'



train_file = data_path + 'ru_train.csv'

test_file = data_path +'ru_test.csv'
train_df = pd.read_csv(train_file)

print(train_df.shape)

train_df.head()
train_df.info()
diff = data[data.before != data.after]

print(diff.shape)

diff.head(10)
classes = train_df['class'].unique(); classes
train_df[train_df['class'] == 'PUNCT']['before'].unique()
def class_sizes(df):

    class_sizes = {}

    df_size = df.shape[0]

    for cl in tqdm(classes):

        #write_file(INPUT_PATH, train_file[:-4] + '.' + cl + '.csv', train_df[train_df['class'] == cl])

        cl_size = df[df['class'] == cl].shape[0]

        cl_unique_size =  df[df['class'] == cl]['before'].unique().size

        class_sizes[cl] = (int((cl_size  / df_size) * 100), cl_size,cl_unique_size)

    return class_sizes
class_sizes_of_train = class_sizes(train_df); class_sizes_of_train
class_sizes_of_diff = class_sizes(diff); class_sizes_of_diff
[train_df[train_df['class'] == c][['before','after']].to_csv(train_file[:-3] + c + '.csv', encoding='UTF-8', index=False) for c in classes]