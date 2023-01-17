import os

import re

import json

import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
single_song_path = '../input/singlesong/single_song.json'

with open(single_song_path, 'rb') as f:

    song_json = json.load(f)

# Uncomment this     

# print(song_json)
# print(json.dumps(None, indent=4, sort_keys=True))
dictionary = {'dog': 'woof',

              'cat': 'meow',

              'lazer': 'zapppp',

              'list_of_things': ['a', 3, dict()],

              'numbers': 10012}

dictionary
dictionary.keys()
dictionary['dog']
dictionary['list_of_things']
list_of_things = dictionary['list_of_things']

list_of_things[0]
dictionary['list_of_things'][0]
song_json.keys()
song_json['meta']
song_json['segments']
# Access the first element (dictionary) in the list

first_segment = song_json['segments'][0]

first_segment
# Then access the timbre values

first_segment['timbre']
# song_timbres = []

# for segment in song_json[None]:

#     song_timbres.append(segment[None])

# song_timbres
training_data_path = '../input/musicdata/'

os.listdir(training_data_path)
with open(os.path.join(training_data_path,'hiphop.json'), 'rb') as f:

    hiphop = json.load(f)
# Each key is a unique identifer for a song known as a URI

hiphop.keys()
hiphop['spotify:track:3MnwLa9KRUiv2gNFtWPvib'].keys()
hiphop['spotify:track:3MnwLa9KRUiv2gNFtWPvib']['artist']
hiphop['spotify:track:3MnwLa9KRUiv2gNFtWPvib']['song']
hiphop['spotify:track:3MnwLa9KRUiv2gNFtWPvib']['meta'].keys()
hiphop['spotify:track:3MnwLa9KRUiv2gNFtWPvib']['meta']['segments'][0]
hiphop['spotify:track:3MnwLa9KRUiv2gNFtWPvib']['meta']['segments'][0]
def get_song_name(json_data, song_uri):

    '''Returns song name from song URI key

     Args:

     * json_data- (JSON) 

     * song_uri- (str) URI

     

     Return

     * (str)- Song name

     '''

    return



def get_artist_name(json_data, song_uri):

    '''Returns Artist name from song URI key

     Args:

     * json_data- (JSON) 

     * song_uri- (str) URI

     

     Return:

     * (str)- Artist name

     '''



    return



def get_timbre_values(json_data, song_uri):

    '''Returns timbre values from a song

    Args:

    * json_data- (JSON) 

    * song_uri- (str) URI 

    

    Return:

    * (list) Each element is a list of timbre values

    '''

    timbre_data = []

    for segment in json_data[None][None][None]:

        timbre_data.append(segment[None])

    return timbre_data



def get_segment_start_time(json_data, song_uri):

    '''Returns start times of segments from a song

    Args:

    * json_data- (JSON) 

    * song_uri- (str) URI 

    

    Return:

    * (list) Each element is float representing time in milliseconds

    '''

    start_times = []

    for segment in json_data[None][None][None]:

        start_times.append(segment[None])

    return start_times



def get_segment_duration(json_data, song_uri):

    '''Returns duration of segments from a song

    Args:

    * json_data- (JSON) 

    * song_uri- (str) URI 

    

    Return:

    * (list) Each element is float representing the duration of a segment

    '''

    durations = [] 

    for segment in json_data[None][None][None]:

        durations.append(segment[None])

    return durations
def get_genre_data(genre_data, genre_type):

    '''

    Processes a JSON object of a single genre

    Args:

    * genre data (JSON)

    * single genre (str) Name of genre

    

    Returns:

    * pandas DataFrame containing training data and label for ML

    '''

    genre_dataframes = []

    for song_uri in genre_data.keys():

        # Extract the relevant data with our functions

        timbres = None

        start_times = None

        durations = None

        artist_name = None

        song_name = None

        # Create a dataframe per song

        # We'll build the timbre parts first then add columns

        song_df = pd.DataFrame(timbres)

        song_df['start'] = None

        song_df['durations'] = None

        song_df['song_name'] = None

        song_df['artist'] = None

        # Remember to add the genre so we can use it for supervised learning later!

        song_df['genre'] = None

        # Now we need to store/append all the songs in a genre dataframe

        genre_dataframes.append(None)

    # Now concatenate the song dataframes into a single genre specific dataframe

    genre_df = pd.concat(None)

    return genre_df
# get_genre_data(hiphop, 'hiphop').head()
genre_data_path = '../input/musicdata/'

genre_list = os.listdir(genre_data_path)
# all_genre_list = []

# for genre in genre_list:

#     # Get rid of the pesky .DS_Store files with this clause

#     if not genre.endswith('.DS_Store'):

#         path = os.path.join(genre_data_path, genre)

#         with open(path, 'rb') as f:

#             genre_json = json.load(f)

#         # Extract the genre from the file name    

#         genre_label = genre.replace('.json', '')   

#         # Apply our function

#         genre_data = get_genre_data(genre_json, genre_label)

#         all_genre_list.append(genre_data)



# df = pd.concat(all_genre_list)

# df.head()
# df.rename(columns={i: 'timbre_value_'+str(i) for i in range(0,12)}, inplace=True)
# df.rename(columns={'timbre_value_0':'loudness',

#                    'timbre_value_1': 'brightness',

#                    'timbre_value_2': 'flatness'}, inplace=True)
def get_genre_df(genre_data_path):

    genre_list = os.listdir(genre_data_path)

    all_genre_list = []

    for genre in genre_list:

    # Get rid of the pesky .DS_Store files with this clause

        if not genre.endswith(None):

            path = os.path.join(genre_data_path, genre)

            with open(path, 'rb') as f:

                genre_json = json.load(f)

            # Extract the genre from the file name    

            genre_label = genre.replace(None, '')   

            # Apply our function get_genre_data

            genre_data = None

            all_genre_list.append(None)



    df = pd.concat(all_genre_list)

    df.rename(columns={i: 'timbre_value_'+str(i) for i in range(0,12)}, inplace=True)

    df.rename(columns={'timbre_value_0':'loudness',

                   'timbre_value_1': 'brightness',

                   'timbre_value_2': 'flatness'}, inplace=True)

    return df



    
# df = get_genre_df(genre_data_path)

# df.head()
# df_pop = df[df['genre']==None]

# df = df[df['genre']!=None]
# training_colummns = ['loudness', 'brightness', 'flatness', 'timbre_value_3',

#                      'timbre_value_4', 'timbre_value_5', 'timbre_value_6', 'timbre_value_7',

#                      'timbre_value_8', 'timbre_value_9', 'timbre_value_10',

#                      'timbre_value_11']

# X = df[training_colummns]

# y = df['genre']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# clf = LogisticRegression(random_state=0, solver='lbfgs',

#                          multi_class='multinomial')

# clf.fit(X_train, y_train)

# y_pred_log_reg = clf.predict(X_test)
# print('f1 score {}'.format(f1_score(y_test, y_pred_log_reg, average='weighted')))

# print('recall score {}'.format(recall_score(y_test, y_pred_log_reg, average='weighted')))

# print('precision score {}'.format(precision_score(y_test, y_pred_log_reg, average='weighted')))
# {key:value for key, value in zip(sorted(df['genre'].unique()), f1_score(y_test, y_pred_log_reg, average=None))}

# log_reg_results = pd.DataFrame({'y_Actual':y_test,

#                         'y_Predicted':y_pred_log_reg})

# confusion_matrix_log_reg = pd.crosstab(log_reg_results['y_Actual'], log_reg_results['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)

# confusion_matrix_log_reg
# rf = RandomForestClassifier()

# rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
# print('f1 score {}'.format(f1_score(y_test, y_pred_rf, average='weighted')))

# print('recall score {}'.format(recall_score(y_test, y_pred_rf, average='weighted')))

# print('precision score {}'.format(precision_score(y_test, y_pred_rf, average='weighted')))
# {key:value for key, value in zip(sorted(df['genre'].unique()), f1_score(y_test, y_pred_log_reg, average=None))}

# y_pred_rf = rf.predict(X_test)

# results_rf = pd.DataFrame({'y_Actual':y_test,

#                            'y_Predicted':y_pred_rf})

# confusion_matrix_rf = pd.crosstab(results_rf['y_Actual'], results_rf['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)

# confusion_matrix_rf
# pop_timbre = df_pop[training_colummns]

# df_pop['predicted_genre'] = rf.predict(None)
# df_pop['song_name'].unique()
# pop_song = df_pop[df_pop['song_name']=='CHopstix (with Travis Scott)']
# pop_song['predicted_genre'].value_counts().plot(kind='bar')



# plt.title('Genre Composition for CHopstix by ScHoolboy Q with Travis Scott')
# plt.rcParams["figure.figsize"] = (10,10)



# colors = {'hiphop':'m',

#            'funk': 'g',

#            'metal': 'k',

#            'jazz':'y',

#            'blues':'b',

#            'classical':'r',

#            'electronic': 'C1'}



# for segment in range(len(pop_song)):

#     prediction = pop_song.iloc[segment]['predicted_genre']

#     start = pop_song.iloc[segment]['start']

#     duration = pop_song.iloc[segment]['durations']

#     plt.hlines(xmin=start, xmax= start+duration, y=1,

#                colors=colors[prediction], linewidth= 200)

# plt.yticks([])

# plt.xlabel='Seconds'

# patches = [mpatches.Patch(color=color, label=genre) for genre, color in colors.items()]

# plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.05),

#            fancybox=True, shadow=True, ncol=7,

#            loc='upper center')

# plt.title('{} by {}'.format(pop_song.iloc[segment]['song_name'],

#                             pop_song.iloc[segment]['artist']))



# plt.show()
# test_songs = get_genre_df('../input/testsongdata/')
# test_songs.head()
# Predict the genre

# test_songs_timbre = test_songs[training_colummns]

# test_songs['predicted_genre'] = rf.predict(test_songs_timbre)
def get_song_data(song_dataframe, genre=None, song_name=None):

    if song_name:

        song_df = song_dataframe[song_dataframe['song_name']==song_name]

        return song_df

    else:

        genre_df = song_dataframe[song_dataframe['genre']==genre]

        random_song = np.random.choice(genre_df['song_name'].unique())

        random_song_data = genre_df[genre_df['song_name']==random_song]

        return random_song_data

        
def get_song_composition_bar(song_data):

    song_data['predicted_genre'].value_counts().plot(kind='bar')

    plt.title('{} by {}'.format(song_data.iloc[0]['song_name'],

                                song_data.iloc[0]['artist']))

    plt.show()
def get_song_composition_timeline(song_data):

    plt.rcParams["figure.figsize"] = (10,10)

    colors = {'hiphop':'m',

               'funk': 'g',

               'metal': 'k',

               'jazz':'y',

               'blues':'b',

               'classical':'r',

               'electronic': 'C1'}



    for segment in range(len(song_data)):

        prediction = song_data.iloc[segment]['predicted_genre']

        start = song_data.iloc[segment]['start']

        duration = song_data.iloc[segment]['durations']

        plt.hlines(xmin=start, xmax= start+duration, y=1,

                   colors=colors[prediction], linewidth= 200)

    plt.yticks([])

    plt.xlabel='Seconds'

    patches = [mpatches.Patch(color=color, label=genre) for genre, color in colors.items()]

    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.05),

               fancybox=True, shadow=True, ncol=7,

               loc='upper center')

    plt.title('{} by {}'.format(song_data.iloc[segment]['song_name'],

                                song_data.iloc[segment]['artist']))



    plt.show()
def song_composition(song_dataframe, genre=None, song_name=None):

    song_data = get_song_data(song_dataframe, genre, song_name=song_name)

    get_song_composition_bar(None)

    get_song_composition_timeline(None)

    
# song_composition(test_songs, 'classical')
# test_songs[test_songs['genre']=='metal']['song_name'].unique()
# song_composition(test_songs, genre=None, song_name='My Own Summer (Shove It)')
# genre_types = ['metal','hiphop', 'funk', 'jazz', 'blues', 'classical', 'electronic']

# for g in genre_types:

#     print(g)

#     song_composition(test_songs, g)