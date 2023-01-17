import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import soundfile as sf

import librosa

import IPython

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots

import plotly.graph_objects as go



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

df['path'] = '/kaggle/input/birdsong-recognition/train_audio/'+df['ebird_code']+'/'+df['filename']
df['secondary'] = df['secondary_labels'].str.replace(r"\[|\]|\'|\"","").str.split('\,').apply(lambda x: [i.strip() for i in x])

secondary_labels = [item.strip() for sublist in df['secondary'].tolist() for item in sublist if item != '']

secondary_labels = pd.Series(secondary_labels, name='secondary_labels')
print("Describe")

print(secondary_labels.describe())



print("\n10 most commonly heard secondary birds")

print(secondary_labels.groupby(by=secondary_labels).count().nlargest(10))



print("\n10 least commonly heard secondary birds")

print(secondary_labels.groupby(by=secondary_labels).count().nsmallest(10))



secondary_labels.groupby(by=secondary_labels).count().hist(bins=10);
species = df['species'].unique()

secondary_species = [arr[1] for arr in secondary_labels.str.split('_').tolist()]

intersection = [s for s in species if s in secondary_species] 
len(intersection)
def count_intersection(species, arr):

    count = 0

    for s in species:

        if s != '' and s.split("_")[1] in arr:

            count +=1

    return count
additional_examples = df[df['secondary'].apply(lambda x: count_intersection(x, intersection) > 0)]['secondary'].tolist()

additional_examples = [s for sublist in additional_examples for s in sublist if s.split("_")[1] in intersection]

additional_examples = pd.Series(additional_examples)
print("Count of secondary labels within examples that are also primary targets")

print(additional_examples.describe())



print("\n10 most common additional examples")

print(additional_examples.groupby(by=additional_examples).count().nlargest(10))



print("\n10 least common additional examples")

print(additional_examples.groupby(by=additional_examples).count().nsmallest(10))



additional_examples.groupby(by=additional_examples).count().hist(bins=10);
def filter_secondary(arr, species):

    for bird in arr:

        if bird != '' and bird.split("_")[1] == species:

            return True

    return False
rewbla = df[df['secondary'].apply(lambda x: filter_secondary(x, 'Red-winged Blackbird'))]

rewbla['ebird_code'].describe()

rewbla['ebird_code'].groupby(by=rewbla['ebird_code']).count().nlargest(10)
rewbla_audio, sr = librosa.load('../input/birdsong-recognition/train_audio/rewbla/XC135672.mp3', sr=32000);

IPython.display.Audio(rewbla_audio, rate=sr)
wilsni1_audio, sr = librosa.load('../input/birdsong-recognition/train_audio/wilsni1/XC185863.mp3', sr=32000);

wilsni1_audio = wilsni1_audio[17*sr:22*sr]

IPython.display.Audio(wilsni1_audio, rate=sr)
melspectrogram_parameters = {

    "n_mels": 128,

    "fmin": 20,

    "fmax": 16000

}

rewbla_melspec = librosa.feature.melspectrogram(rewbla_audio, sr=32000, **melspectrogram_parameters)

rewbla_melspec = librosa.power_to_db(rewbla_melspec).astype(np.float32)



wilsni1_melspec = librosa.feature.melspectrogram(wilsni1_audio, sr=32000, **melspectrogram_parameters)

wilsni1_melspec = librosa.power_to_db(wilsni1_melspec).astype(np.float32)



fig=plt.figure(figsize=(16, 8))

fig.add_subplot(1, 2, 1).set_title("Red-winged Blackbird")

plt.imshow(rewbla_melspec)

fig.add_subplot(1, 2, 2).set_title("Wilson's Snipe")

plt.imshow(wilsni1_melspec)

plt.show()
rewbla=rewbla[rewbla.latitude!='Not specified']

fig = go.Figure()

fig.add_trace(go.Scattergeo(

        lon = rewbla['longitude'],

        lat = rewbla['latitude'],

        text = rewbla['ebird_code'],

        name='Co-occuring Locations',

        marker = dict(

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area',

        )))





fig.add_trace(go.Scattergeo(

        lon = df[df['ebird_code'] == 'rewbla']['longitude'],

        lat = df[df['ebird_code'] == 'rewbla']['latitude'],

        text = df[df['ebird_code'] == 'rewbla']['ebird_code'],

        name='Red-winged Blackbird Locations',

        marker = dict(

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area',

        )))



fig.update_layout(

        title_text = 'Geographical Correlation between Red-winged Blackbird and co-occuring species',

        showlegend = True,

        geo = dict(

            landcolor = 'rgb(217, 217, 217)',

            scope="north america",

        )

    )



fig.show()
ebird_dict = {}

for bird in intersection:

    ebird_dict[bird] = df[df['species'] == bird]['ebird_code'].iloc[0]
secondary_dict = {}

for index, row in df.iterrows():

    s = [ebird_dict[bird.split("_")[1]] for bird in row['secondary'] if bird != "" and bird.split("_")[1] in ebird_dict]

    secondary_dict[row['filename']] = s
import pickle

with open('secondary_ebird.pkl', 'wb') as f:

    pickle.dump(secondary_dict, f)
# to load it, use the following snippet

# secondary_dict = {}

# with open(PATH, 'rb') as f:

#     secondary_dict = pickle.load(f)