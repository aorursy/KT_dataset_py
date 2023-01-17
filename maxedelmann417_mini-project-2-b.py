# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from pandas.io.json import json_normalize

import json
import json 

import csv 



with open('/kaggle/input/bob-dylan-lyrics/Lyrics_BobDylan.json') as json_file: 

    data = json.load(json_file)



data.keys()



print(data['songs'][0]['lyrics'])

    
for song in data['songs']:

    print(song['full_title'])
song_lyrics = []

song_title = []



for song in data['songs']:

    if song['full_title'][0:5] not in song_title:

        song_title.append(song['full_title'][0:5])

        song_lyrics.append(song['lyrics'].replace('\n','. '))



print(len(song_lyrics))

#song_lyrics[0]

df=pd.DataFrame({'lyrics':song_lyrics})

df.head(5)



df.head(200).to_csv('Lyrics6.csv')
for n in range(1):

        print(data['songs'][n]['lyrics'].strip(','))
