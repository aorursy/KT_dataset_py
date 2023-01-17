# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

sample_submission.head()
train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

train_df.head()
train_df.shape
print(f"""there are {train_df['grapheme_root'].nunique()} grapheme_root, {train_df['vowel_diacritic'].nunique()} vowel_diacritics and {train_df['consonant_diacritic'].nunique()} consonant_diacritic.""")
print(f"""Most frequent graphene_root is {train_df['grapheme_root'].value_counts().index[0]}

Most frequent vowel_diacritic is {train_df['vowel_diacritic'].value_counts().index[0]}

Most frequent consonant_diacritic is {train_df['consonant_diacritic'].value_counts().index[0]}""")
plots=train_df["grapheme_root"].value_counts().reset_index()

plots.columns = ['grapheme_root', 'counts']

plt.scatter( x=plots.grapheme_root, y=plots.counts, c='g', s= (plots.counts/20))
class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

class_map_df.tail()
class_map_df.head()
class_map_df.shape
img = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

img.shape
img.head()
img_id = img.iloc[:,0]

image = img.iloc[:,1:].values.reshape(-1, 137,236)

image
plt.figure(figsize=(12,12))

for i in range(10):

 plt.subplot(5,5,i+1)

 plt.imshow(image[i])