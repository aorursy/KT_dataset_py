import gc

import numpy as np

import librosa

import librosa.display

import matplotlib.pyplot as plt

import pandas as pd

import os

from pathlib import Path

import IPython.display as ipd

%matplotlib inline
# Here is my set up cell. Getting access to the sound files is a little intimidating 

# at first, but it is pretty close to what you would do with images which has already 

# been 'solved' just watch fastai classes with the man Jeremy Howard



# Set up paths to sound containing folders

path_crtd = Path('../input/train_curated')

path_nsy = Path('../input/train_noisy')

path_test = Path('../input/test')

FART_TAG = ['Electric_guitar'] # ['Fart']



# list of files to be read in

cur_wv = [path_crtd/fname for fname in os.listdir(path_crtd)]

nsy_wv = [path_nsy/fname for fname in os.listdir(path_nsy)]

tst_wv = [path_test/fname for fname in os.listdir(path_test)]



# label and sample submission dfs

cur_df = pd.read_csv("../input/train_curated.csv")

nsy_df = pd.read_csv("../input/train_noisy.csv")

sam_sub = pd.read_csv('../input/sample_submission.csv')
## Here are a few handy helper functions to seperate and look at the data:

## Remember, this cell can be hidden to keep a close eye on your variables.



def seperate_sound_tag(file_path, label_df, sound_tag):

    """This function expects a list of file paths, a df with labels and file names, and a label.

    Returns a list of file paths that contain the sound tag and no other tags.

    

    parameters

    ----------

    file_path(Path): path to .wav files

    label_df(df): df containing filenames and labels

    sound_tag(string): one of the labels included in df.lables

    

    returns

    -------

    list of files containing the sound_tag string

    """    

   

    df_st = label_df[ label_df.labels == sound_tag ]

    ## dictionary with filenames as keys and file paths as values

    file_dict = {fname:file_path/fname for fname in os.listdir(file_path)}

    tag_file_names = list(df_st['fname'])

    tag_files = {key:value for key,value in file_dict.items() if key in tag_file_names}    

    return list(tag_files.values())



def random_file(file_list, label_df=None, sound_tag=None):

    """Returns random file from list."""    

    if type(label_df) == pd.DataFrame:

        file_list = seperate_sound_tag(file_path=file_list, label_df=label_df, 

                    sound_tag=sound_tag)

    

    file_path = file_list[np.random.randint(len(file_list))] 

    return file_path



def make_spectrogram(file_path=None, clip=None, sample_rate=None):

    """Expects a numpy array and returns a melspectogram"""

    if file_path:

        clip, sample_rate = librosa.load(file_path, sr=None)

    spectrogram = librosa.feature.melspectrogram(clip, sr=44100)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)

    return spectrogram



def line_graph_with_mel(file_path, df=cur_df):

    """"""

    clip, sample_rate = librosa.load(str(file_path), sr=None)

    spect = make_spectrogram(file_path=file_path)

    ## grab the tags from the label df for the title

    tags = df[df['fname'] == file_path.parts[-1]].labels.iloc[0]

    

    f, axs = plt.subplots(nrows=2, figsize=(12,6))

    axs[0].plot(np.array( range(len(clip) ) )/sample_rate, clip, )

    axs[0].set_xlabel('Seconds', size=15)

    axs[0].set_title(f'{tags} lineplot', size=22)

    axs[0].grid()

    librosa.display.specshow(spect, x_axis='time', y_axis='mel', hop_length=100, ax=axs[1])

    # plt.colorbar()

    axs[1].set_title(f'{tags} Mel Spectogram', size=22)

    plt.tight_layout()

    

def grid_melspectrogram(data_files, tags, label_df, num_graphs=10):

    """Makes a grid of melspectrograms with tags.

    

    Args:

        data_files(Path): path to a group of .wav files

        tags(list or str): list of tags or string with single tag

        label_df(pd.DataFrame): containing filename and coorisponding tag(s)

        num_graphs(int): number of graphs in figure

    Returns:

        matplotlib object ?

        """

    

    ## Get a list of .wav files and coorisponding tags

    file_list = []

    new_tags = []

    if type(tags) == list:

        for graph_number in range(num_graphs):

            tag_ix = graph_number % len(tags)

            tag = tags[tag_ix]

            sound_files = seperate_sound_tag(file_path=data_files, label_df=label_df, sound_tag=tag)

            sound_file = random_file(sound_files)

            file_list.append(sound_file)

            new_tags.append(tag)

    

    else:

        sound_files = seperate_sound_tag(tags)

        for graph_number in range(num_graphs):

            file_list.append(random_file(file_list = sound_files))

            new_tags.append(tags)

    

    ## Cycle through tag list and file paths to make graphs

    ## Cycle through tag list and file paths to make graphs

    

    row_number = int(np.ceil(num_graphs/2))

    f, axs = plt.subplots(nrows=row_number, ncols=2, figsize=(12,row_number*3))

    axs = axs.ravel()

    for i in range(num_graphs):

        spec = make_spectrogram(file_path=file_list[i])

        librosa.display.specshow(spec, x_axis='time', y_axis='mel', hop_length=100, ax=axs[i])

        axs[i].set_title(new_tags[i])



    plt.tight_layout()

## Run this cell to get get a list of (top 10) individual tags and how many files there are in the curated dataset

# cur_df['len'] = cur_df.iloc[:,1].apply(lambda x: len(x.split(',')))

# training_set = cur_df[cur_df['len'] == 1]

# training_set['labels'].value_counts().iloc[:10]
SOUND_TO_VIEW = 'Fart'



file_path = random_file(path_crtd, cur_df, SOUND_TO_VIEW)



line_graph_with_mel(file_path=file_path, df=cur_df)

ipd.Audio( os.fspath(file_path) ) # load a local WAV file

## Question how to add a colorbar to the mel spectogram visulization



SOUNDS_TO_COMPARE=['Fart', 'Electric_guitar',]



grid_melspectrogram(data_files=path_crtd, tags=SOUNDS_TO_COMPARE, label_df=cur_df)