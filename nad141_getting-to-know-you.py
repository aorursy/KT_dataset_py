# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pyarrow.parquet as pq
train_df = pd.read_csv('../input/bengaliai-cv19/train.csv')
train_df.head()


print(train_df.shape)
grapheme_counts = train_df.grapheme_root.value_counts().to_dict()

vowel_counts = train_df.vowel_diacritic.value_counts().to_dict()

consonant_counts = train_df.consonant_diacritic.value_counts().to_dict()
class_map = pd.read_csv('../input/bengaliai-cv19/class_map.csv')
import matplotlib.pyplot as plt

%matplotlib inline

def get_bar_plot(dict_, name):

    plt.bar(range(len(dict_)), list(dict_.values()), align='center')

#     plt.rcParams["figure.figsize"] = (10,10)

    plt.title(name)

    plt.xticks(range(len(dict_)), list(dict_.keys()))

    plt.show()

get_bar_plot(vowel_counts,'vowel counts')

get_bar_plot(consonant_counts, 'consonant counts')

# get_bar_plot(grapheme_counts, 'grapheme counts')
df_train_0 = pq.read_table(source='../input/bengaliai-cv19/train_image_data_0.parquet').to_pandas()

# df_train_1 = pd.read_table(source='../input/bengaliai-cv19/train_image_data_1.parquet')

# df_train_2 = pd.concat(pd.read_table(source='../input/bengaliai-cv19/train_image_data_2.parquet').to_pandas(),df_train_0)

# del df_train_0

# df_train_3 = pd.concat(pd.read_table(source='../input/bengaliai-cv19/train_image_data_3.parquet').to_parquetndas(), df_train_2)

# del df_train_2

# df_train = pd.concat(df_train_0, df_train_1,df_train_2,df_train_3)
def display_img_label(img_num):

    img = np.array(df_train_0.iloc[img_num].values[1:].reshape(137,-1)).astype('float')

    label = train_df.iloc[img_num].values[1:-1]

    plt.imshow(img,cmap = 'gray')

    print(label)

    plt.title(''.join(str(label)))
display_img_label(23)
from mpl_toolkits import mplot3d





fig = plt.figure()

ax = plt.axes(projection='3d')

ax.scatter3D(train_df.grapheme_root.values,train_df.vowel_diacritic.values,train_df.consonant_diacritic)

frequency_combinations = {}

all_combinations = train_df[['grapheme_root','vowel_diacritic','consonant_diacritic']].values
for i in all_combinations:

    if not tuple(i) in frequency_combinations:

        frequency_combinations[tuple(i)] = 1

    else:

        frequency_combinations[tuple(i)] +=1
len(sorted(frequency_combinations))
frequency_combinations = sorted(frequency_combinations.items(), key=lambda x: x[1], reverse=True)
def get_img(gr,vo,c):

    df_ = train_df[train_df['grapheme_root'] == gr][train_df['vowel_diacritic'] == vo][train_df['consonant_diacritic'] == c]

    display_img_label(int(df_.iloc[0]['image_id'].split('_')[-1]))
fig = plt.figure()

fig.subplots_adjust(hspace=1, wspace=1)

for i,f in enumerate(frequency_combinations[:10]):

    ax = fig.add_subplot(4, 3, i+1)

    

    get_img(*f[0])
fig = plt.figure()

fig.subplots_adjust(hspace=1, wspace=1)

for i,f in enumerate(frequency_combinations[-10:]):

    ax = fig.add_subplot(4, 3, i+1)

    

    get_img(*f[0])
print(frequency_combinations[-10:])