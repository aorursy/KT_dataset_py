



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import PIL.Image

import time

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
for dirname,_, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
DATA_FOLDER = '/kaggle/input/bengaliai-cv19/'

train_df = pd.read_csv(os.path.join(DATA_FOLDER,'train.csv'))

train_df.head()
train_df.shape
test_df = pd.read_csv(os.path.join(DATA_FOLDER,'test.csv'))

test_df.head()
test_df.shape
class_map_df = pd.read_csv(os.path.join(DATA_FOLDER,'class_map.csv'))

class_map_df.head()
class_map_df.shape
sample_submission_df = pd.read_csv(os.path.join(DATA_FOLDER,'sample_submission.csv'))

sample_submission_df.head()
sample_submission_df.shape
start_time = time.time()

train_0_df = pd.read_parquet(os.path.join(DATA_FOLDER,'train_image_data_0.parquet'))

print(f"'train_image_data_0' read in {round(time.time()-start_time,2)} sec.")
train_0_df.shape
train_0_df.head()
start_time = time.time()

train_1_df = pd.read_parquet(os.path.join(DATA_FOLDER,'train_image_data_1.parquet'))

print(f"'train_image_data_1' read in {round(time.time()-start_time,2)} sec.")
train_1_df.shape
train_1_df.head()
start_time = time.time()

test_0_df = pd.read_parquet(os.path.join(DATA_FOLDER,'test_image_data_0.parquet'))

print(f"'test_image_data_0' read in{round(time.time()-start_time,2)} sec.")
test_0_df.shape
test_0_df.head()

print(f"Train: unique graphene roots: {train_df.grapheme_root.nunique()}")

print(f"Train: unique vowel diacritics: {train_df.vowel_diacritic.nunique()}")

print(f"Train: unique consonant diacritics: {train_df.consonant_diacritic.nunique()}")

print(f"Train: total unique elements: {train_df.grapheme_root.nunique() + train_df.vowel_diacritic.nunique() + train_df.consonant_diacritic.nunique()}")

print(f"Class map: unique elements: \n{class_map_df.component_type.value_counts()}")

print(f"Total combinations: {pd.DataFrame(train_df.groupby(['grapheme_root','vowel_diacritic','consonant_diacritic'])).shape[0]}")

cm_gr = class_map_df.loc[(class_map_df.component_type=='grapheme_root'),'component'].values

cm_vd = class_map_df.loc[(class_map_df.component_type=='vowel_diacritic'),'component'].values

cm_cd = class_map_df.loc[(class_map_df.component_type=='consonant_diacritic'),'component'].values

print(f"grapheme root:\n{15*'-'}\n{cm_gr}\n\n vowel diacritic:\n{18*'-'}\n{cm_vd}\n\n consonant diacritic:\n{20*'-'}\n{cm_cd}")
def most_frequent_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columsn = ['Total']

    items=[]

    vals = []

    for col in data.columns:

        itm = data[col].value_counts().index[0]

        val = data[col].value_counts().values[0]

        items.append(itm)

        vals.append(val)

    tt['Most frequent item']=items

    tt['Frequency']=vals

    tt['Percent from total'] = np.round(vals/total*100,3)

    return(np.transpose(tt))
most_frequent_values(train_df)
most_frequent_values(test_df)
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1,figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature],order=df[feature].value_counts().index[:20],palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size>2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

               height + 3,

               '{:1.2f}%'.format(100*height/total),

                   ha="center")

    plt.show()
plot_count('grapheme_root','grapheme_root(first most frequent 20 values - train)',train_df, size=4)
plot_count('vowel_diacritic', 'vowel_diacritic (train)',train_df, size=3)
plot_count('consonant_diacritic','consonant_diacritic (train)',train_df, size=3)
def plot_count_heatmap(feature1, feature2, df, size=1):

    tmp = train_df.groupby([feature1, feature2])['grapheme'].count()

    df = tmp.reset_index()

    df

    df_m = df.pivot(feature1, feature2, "grapheme")

    f,ax = plt.subplots(figsize=(9, size*4))

    sns.heatmap(df_m, annot=True, fmt='3.0f',linewidths=.5, ax=ax)
plot_count_heatmap('vowel_diacritic','consonant_diacritic',train_df)
plot_count_heatmap('grapheme_root','consonant_diacritic',train_df,size=8)
def display_image_from_data(data_df, size=5):

    plt.figure()

    fig, ax = plt.subplots(size, size, figsize=(12,12))

    for i,index in enumerate(data_df.index):

        image_id = data_df.iloc[i]['image_id']

        flattened_image = data_df.iloc[i].drop('image_id').values.astype(np.uint8)

        unpacked_image = PIL.Image.fromarray(flattened_image.reshape(137, 236))

        ax[i//size, i%size].imshow(unpacked_image)

        ax[i//size, i%size].set_title(image_id)

        ax[i//size, i%size].axis('on')
display_image_from_data(train_0_df.sample(25))
display_image_from_data(train_1_df.sample(16), size=4)
def display_writting_variety(data_df=train_0_df, grapheme_root=72, vowel_diacritic=0,\

                             consonant_diacritic=0, size=5):

    

    sample_train_df = train_df.loc[(train_df.grapheme_root == grapheme_root) & \

                                  (train_df.vowel_diacritic == vowel_diacritic) & \

                                  (train_df.consonant_diacritic == consonant_diacritic)]

    print(f"total: {sample_train_df.shape}")

    sample_df = data_df.merge(sample_train_df.image_id, how='inner')

    print(f"total: {sample_df.shape}")

    gr = sample_train_df.iloc[0]['grapheme']

    cm_gr = class_map_df.loc[(class_map_df.component_type=='grapheme_root')& \

                             (class_map_df.label==grapheme_root), 'component'].values[0]

    cm_vd = class_map_df.loc[(class_map_df.component_type=='vowel_diacritic')& \

                             (class_map_df.label==vowel_diacritic), 'component'].values[0]    

    cm_cd = class_map_df.loc[(class_map_df.component_type=='consonant_diacritic')& \

                             (class_map_df.label==consonant_diacritic), 'component'].values[0]    

    

    print(f"grapheme: {gr}, grapheme root: {cm_gr}, vowel discritic: {cm_vd}, consonant diacritic: {cm_cd}")

    sample_df = sample_df.sample(size * size)

    display_image_from_data(sample_df, size=size)
display_writting_variety(train_0_df,72,1,1,4)
display_writting_variety(train_0_df,64,1,2,4)
display_writting_variety(train_1_df,13,0,0,4)
display_writting_variety(train_1_df,23,3,2,4)