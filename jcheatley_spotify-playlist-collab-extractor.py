import pandas as pd

import sys

import requests

import base64

import json

import time

from itertools import permutations



CLIENT_ID = 'df5d6fb6ac844a48a111d2d1df333fae'

CLIENT_SECRET = 'd550fd131774431597f57383d23af017'

CREDENTIALS = base64.urlsafe_b64encode((CLIENT_ID + ':' + CLIENT_SECRET).encode()).decode()

PLAYLIST_ID = '0m4StDrOqwFjNiYygelXy5'
# Auth token related code

EXPIRATION = 0



def get_new_token():

    global EXPIRATION

    url = 'https://accounts.spotify.com/api/token'

    headers = {'Authorization': f'Basic {CREDENTIALS}'}

    payload = {'grant_type': 'client_credentials'}



    response = requests.post(url, headers=headers, data=payload, verify=True)

    json_response = json.loads(response.text)

    EXPIRATION = int(time.time()) + json_response['expires_in']

    return json_response



def token_expired():

    return time.time() + 60 >= EXPIRATION
# Extracting a whole playlist

next_url = None



artist_info = set()  # (artist_id, artist_name)

song_info = []  # (song_id, song_name)

songs = []  # (song_id, artist_id, collab?)



while True:

    token = get_new_token() if token_expired() else token

    

    headers = {'Authorization': f'Bearer {token["access_token"]}'}

    

    if next_url is None:

        url = f'https://api.spotify.com/v1/playlists/{PLAYLIST_ID}/tracks'

        payload = {'fields': 'items(track(artists,name,id)),next'}



        response = requests.get(url, headers=headers, params=payload)

    else:

        url = next_url

        response = requests.get(url, headers=headers)

    json_response = json.loads(response.text)

    next_url = json_response['next']

    if next_url is None:

        break

        

    # Read all the songs in the response

    for track in json_response['items']:

        track_data = track['track']

        song_name = track_data['name']

        song_id = track_data['id']

        song_artists = [(artist['id'], artist['name']) for artist in track_data['artists']]

        

        # Add to my dataset

        song_info.append((song_id, song_name))

        for artist in song_artists:

            artist_info.add(artist)

            songs.append((song_id, artist[0], len(song_artists) > 1))
print(list(artist_info)[:10])

print(song_info[:10])

print(songs[:10])
# Saving to CSV

artist_info_df = pd.DataFrame.from_records(list(artist_info), columns=['artist_id', 'artist_name'])

song_info_df = pd.DataFrame.from_records(song_info, columns=['song_id', 'song_name'])

songs_df = pd.DataFrame.from_records(songs, columns=['song_id', 'artist_id', 'collab'])



artist_info_df.to_csv('spotify_artist_info.csv', index=False)

song_info_df.to_csv('spotify_song_info.csv', index=False)

songs_df.to_csv('spotify_songs.csv', index=False)