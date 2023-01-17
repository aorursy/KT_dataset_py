
import pandas as pd
import os
import warnings

%matplotlib notebook
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")

import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
%matplotlib notebook

plt.style.use('ggplot')
### Read the datasets from the given locations and do following
path = "../input/spotify-playlist/"
filename = "spotify_dataset.csv"
### While Loading datasets we say error_bad_lines = false which drops rows with errors 
### As it is experimental project and we have huge datasets, dropping 100-200 Bad rows will not impact any results
print ("Reading the data")
spotify_data = pd.read_csv(os.path.join(path,filename),escapechar= '.',error_bad_lines = False,warn_bad_lines=False)
print ("Read Succesful with shape {}".format(spotify_data.shape))
### Columns names were not very clean give them manual names
spotify_data.columns = ['user_id','artistname','trackname','playlistname']
print ("Some General statistics about data are as follows:",spotify_data.info())
print ("Lets look at the summary stats about the data :",spotify_data.describe(include ='object'))
print ("The number of rows in the datasets are as follows :",spotify_data.shape[0])
print (" The columns in the data are as follows :",spotify_data.columns)
### Now lets define function which creates a dictionary and convert songs names to dictionary

tracklist = spotify_data['trackname'].unique()
spotify_data.dropna(inplace = True)
### Create a function which takes a dataset name and column name 

def create_dict(dataset, column):
    ''' Takes two input from user column name and dataset name and return dictionary with hash map '''
    unique_list = dataset[column].unique()
    out_dict = {}
    out_dict1 = {}
    
    for j,i in enumerate(unique_list):
        out_dict[i.lower()] = str(j)
        out_dict1[str(j)] = i.lower()
        
    print ("Number of distinct in vocab is :",j)
    return (out_dict,out_dict1)
### call the dict functions on track names and artistname
track_map, track_map_comp= create_dict(spotify_data,'trackname')
artist_map,artist_map_comp = create_dict(spotify_data,'artistname')
with open('track_map_dict.pickle','wb') as track_file:
    pickle.dump(track_map,track_file)
with open('track_map_comp_dict.pickle','wb') as track_file_comp:
    pickle.dump(track_map_comp,track_file_comp)
with open('artist_map_dict.pickle','wb') as artist_file:
    pickle.dump(artist_map,artist_file)
with open('artist_map_comp_dict.pickle','wb') as artist_file_comp:
    pickle.dump(artist_map_comp,artist_file_comp)
### Lets shuffle the data first
print ("Shape of data before sampling is:", spotify_data.shape)
spotify_data.sample(frac = 1,  random_state = 10000).reset_index(drop=True)
print ("Shape of data after sampling is :", spotify_data.shape)
### Load the pickle files stored for song to numeric 
with open('track_map_dict.pickle','rb') as dict1:
    track_dict= pickle.load( dict1)
print ("Track dict has {} observations".format(len(track_dict)))
#### Load the prcikle file for artist to numeric
with open('artist_map_dict.pickle','rb') as dict2:
    artist_dict = pickle.load(dict2)
print ("Track dict has {} observations".format(len(artist_dict)))
### Now we will use this mapping to convert names to numeric
print ("Data before mapping dict :", spotify_data.head(5))
spotify_data['trackname'] = spotify_data['trackname'].str.lower().map(track_dict)
spotify_data['artistname'] = spotify_data['artistname'].str.lower().map(artist_dict)
print ("Data after mapping dict :")
print (spotify_data.head(5))
### We want to create a list of songs in zip file 
def zip_list(x):
    return ([str(z) for z in x])
spotify_summary = spotify_data.groupby(['user_id','playlistname'])['trackname'].apply(zip_list).reset_index()
print (" Distinct playlist after summarizing the data is :",spotify_summary.shape[0])
print (" The data looks like this :")
print (spotify_summary.head(5))
### We will Dump this data in the pickle file and work in it later
with open("spotify_summary.pickle",'wb') as pick_data:
    pickle.dump(spotify_summary,pick_data)
    print ("The dataset is pickled at ",os.getcwd())
### Load the pickled datasets 
with open('spotify_summary.pickle','rb') as dataset:
    spotify_summary = pickle.load(dataset)
    print (" The dataset is loaded succesfully")
    print (" The shape of the dataset is as follows",spotify_summary.shape)
    print (spotify_summary.head(5))
### Gensim takes input as a list of list. Our tracknames are already a list convert them to list of list
spotify_wrd2vec_input = [ x for x in spotify_summary['trackname']]
print ("Input data is ready for gensim models")
print ("The number of input playlists we have are as follows :",len(spotify_wrd2vec_input))
### Define traing the word 2 vec model we will use Skip Gram using negative sampling as oftmax can be slow
# seed = 1000, hs = 0,negative = 10,workers=10,iter = 100)
### Skip Gram : Predict Context given the middle word works well with infrequent datasets.
### Good idea for songs as some songs may ne liked by a few users oly
print ("Model Training has started")
model = gensim.models.Word2Vec(spotify_wrd2vec_input, size = 200 , window = 4 , min_count = 15,
                               seed = 1000, hs = 0,negative = 10,workers=16,iter = 100)
print ("Model Trainin Finished")
### Pickle the model datasets and save it to a pickle file 

with open('model_spotify_word2vec.pickle','wb') as model_file:
    pickle.dump(model,model_file)
    print (" Dumping the model succesful ")
    print (" The model is dumped at this location :",os.getcwd())
### From the dump load the model dictionary and model pickle files
with open('model_spotify_word2vec.pickle','rb') as model_file:
    model_spotify = pickle.load(model_file)

### Load the pickle files stored for song to numeric 
with open('track_map_dict.pickle','rb') as dict1:
    track_dict= pickle.load( dict1)
print ("Track dict has {} observations".format(len(track_dict)))
#### Load the prcikle file for artist to numeric
with open('track_map_comp_dict.pickle','rb') as dict2:
    track_map_comp_dict = pickle.load(dict2)
print ("Track dict has {} observations".format(len(track_map_comp_dict)))
#### Define a function which takes as input songs from list and returns similar songs
def similar_songs(songname,n):
    ''' Gets the songname from user and return the n songs similar'''
    song_id = track_dict[songname]
    print ("Searching for songs similar to :",songname)
    
    similar = model_spotify.most_similar(song_id,topn = n)
    print ("Similar songs are as follow")
    for i in similar[:]:
        print (track_map_comp_dict[i[0]])
#### Define a function which takes as input songs from list and returns similar songs
def create_play_list(list_songs,n):
    ''' Gets the songname from user and return the 5 songs similar'''  
    list1 = []
    for i in list_songs:
        list1.append(track_dict[i])      
        
    print ("Searching for songs similar to :",list_songs)
    
    similar = model_spotify.most_similar(positive = list1,topn = n)
    print ("Playlist based on your list is as follows")
    for i in similar[:]:
        print (track_map_comp_dict[i[0]])
create_play_list(['wonderwall','paradise','yellow','let her go','fireflies'],10)
### Lets check the results for a different music taste - Classic Metal | Rock
create_play_list(['enter sandman','fade to black','kashmir'],15)
create_play_list(['hey you','time','hypnotised','fix you'],10)
similar_songs('kashmir',5)