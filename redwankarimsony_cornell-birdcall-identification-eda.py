import numpy as np

import pandas as pd

from os import listdir

import matplotlib.pyplot as plt

import IPython.display as ipydisplay 
PATH_AUDIO = '../input/birdsong-recognition/train_audio/'

train_df = pd.read_csv('../input/birdsong-recognition/train.csv')

test_df = pd.read_csv('../input/birdsong-recognition/test.csv')

train_df.head()
plt.figure(figsize=(10, 5))

train_df.country.value_counts()[:10].plot.bar()

plt.title('Top 10 Countries where data is recorded')

plt.grid('True')

plt.xlabel('Courntries')

plt.ylabel('No of audio samples taken')

plt.show()
ebird_name_map = train_df[['ebird_code','species', 'primary_label']].drop_duplicates()

no_of_categories = ebird_name_map.shape[0]

print('No of bird categories: ', no_of_categories)
train_df.columns
# for ebird_code in listdir(PATH_AUDIO)[:10]:

#     sample_audio_path = f'{PATH_AUDIO}{ebird_code}/{listdir(PATH_AUDIO + ebird_code)[0]}'

#     species = ebird_name_map[ebird_name_map.ebird_code == ebird_code].species.values[0]

#     ipydisplay.display(ipydisplay.HTML(f"<h3>{ebird_code} ({species})</h3>"))    

#     ipydisplay.display(ipydisplay.Audio(sample_audio_path))
# train_df.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings"}).sort_values("recordings", ascending=False).plot.barh(figsize = (5,100))

output = train_df.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "recordings"}).sort_values("recordings", ascending=False).reset_index(drop=True, inplace=False).copy()

plt.figure(figsize = (7, 60))

plt.xticks(rotation = 90)

plt.barh(output.species, output.recordings)

plt.grid(True)

plt.show()
print('Categories with 100 recordings: ', output[output.recordings ==100].species.count())

print('Categories with less than 100 recordings:', output[output.recordings <100].species.count())