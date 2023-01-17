### Load the required packages in the required format
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
### Lets first look at the distribution of number of playlist by user_id
spotify_user_summary = spotify_data.groupby(['user_id'])["playlistname"].nunique().reset_index()
### Just for better visualisation remove very high playlist
fig, ax = plt.subplots(2, figsize=(6,8))
sns.distplot(spotify_user_summary['playlistname'], hist=True,ax = ax[0])
ax[0].set_title("Number of playlist per user")
spotify_user_summary = spotify_user_summary[spotify_user_summary['playlistname'] <= 50]
sns.distplot(spotify_user_summary['playlistname'], hist=True,ax=ax[1])
ax[1].set_title("Number of playlist per user (where <=50)")
plt.show()
### Lets first look at the distribution of number of playlist by user_id
spotify_user_summary = spotify_data.groupby(['user_id'])["artistname"].nunique().reset_index()

### Just for better visualisation remove very high playlist
fig, ax = plt.subplots(2, figsize=(6,8))
sns.distplot(spotify_user_summary["artistname"], hist=True,ax = ax[0])
ax[0].set_title("Number of artist per user")
spotify_user_summary = spotify_user_summary[spotify_user_summary["artistname"] <= 500]
sns.distplot(spotify_user_summary["artistname"], hist=True,ax=ax[1])
ax[1].set_title("Number of artist per user (where <=500)")
plt.show()
### Insight 2
###### Most of the users listen to less number of artist as compared higher number of artist
###### Also We see a few users which may have very high number of playlists 
### Lets first look at the distribution of number of playlist by user_id
spotify_user_summary = spotify_data.groupby(['user_id'])["trackname"].nunique().reset_index()

### Just for better visualisation remove very high playlist
fig, ax = plt.subplots(2, figsize=(6,8))
sns.distplot(spotify_user_summary["trackname"], hist=True,ax = ax[0])
ax[0].set_title("Number of tracks per user")
spotify_user_summary = spotify_user_summary[spotify_user_summary["trackname"] <= 500]
sns.distplot(spotify_user_summary["trackname"], hist=True,ax=ax[1])
ax[1].set_title("Number of tracks per user (where <=500)")
plt.show()
### Lets first look at the distribution of number of playlist by user_id
spotify_user_summary = spotify_data.groupby(['user_id'])["trackname","artistname"].nunique().reset_index()
spotify_user_summary = spotify_user_summary[spotify_user_summary['artistname']>0]

spotify_user_summary['Track_to_artist_ratio'] = spotify_user_summary['trackname'] / spotify_user_summary['artistname']
spotify_user_summary.dropna(inplace = True)
spotify_user_summary['Track_to_artist_ratio'] = spotify_user_summary['Track_to_artist_ratio'].astype(int)

### Just for better visualisation remove very high playlist
fig, ax = plt.subplots(2, figsize=(6,8))
sns.distplot(spotify_user_summary["Track_to_artist_ratio"], hist=True,ax = ax[0])
ax[0].set_title("AVerage tracks per artist")
spotify_user_summary = spotify_user_summary[spotify_user_summary["Track_to_artist_ratio"] <= 40]
sns.distplot(spotify_user_summary["Track_to_artist_ratio"], hist=True,ax=ax[1])
ax[1].set_title("Number of tracks per artist (where <=40)")
plt.show()
%matplotlib inline
### Lets have a look at most common playlist
## We could have aslo created a word cloud
spotify_data['playlistname'].value_counts()[3:53].plot(kind= 'bar',title="Most Common Playlist names",figsize = (16,6))
%matplotlib inline
### Lets have a look at most common playlist
## We could have aslo created a word cloud
spotify_data['artistname'].value_counts()[0:51].plot(kind= 'bar',title="Top 50 Most Popular Artist ",figsize = (15,6))
%matplotlib inline
### Lets have a look at most common playlist
## We could have aslo created a word cloud
spotify_data['trackname'].value_counts()[0:51].plot(kind= 'bar',title="Top 50 Most Popular Songs  ",figsize = (15,6))
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
