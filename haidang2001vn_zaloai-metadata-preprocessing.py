import pandas as pd

import numpy as np



train_info = pd.read_csv("../input/zaloai-2019-rawmetadata/train_info.tsv", sep='\t')

train_rank = pd.read_csv("../input/zaloai-2019-rawmetadata/train_rank.csv")

test_info = pd.read_csv("../input/zaloai-2019-rawmetadata/test_info.tsv", sep='\t')
train = train_info.merge(train_rank, on='ID')
test = test_info
train['composer_track'] = 0

train['artist_track'] = 0

train['composer_total'] = 0

train['artist_total'] = 0



for i in range(10):

    train['composer_rank%d'%(i + 1)] = 0

    train['artist_rank%d'%(i + 1)] = 0
test['composer_track'] = 0

test['artist_track'] = 0

test['composer_total'] = 0

test['artist_total'] = 0



for i in range(10):

    test['composer_rank%d'%(i + 1)] = 0

    test['artist_rank%d'%(i + 1)] = 0
train.head()
test.head()
composers_columns = ['composer_rank%d'%(rank + 1) for rank in range(10)]

composers_columns.append('composer_track')

composers = pd.DataFrame(columns = composers_columns, dtype = np.int64)

composers.head()
composers_dict = {}

for i in range(train.shape[0]):

    composers_id = train.loc[i, 'composers_id'].replace(',', '.')

    composers_id = composers_id.split('.')

    for idx in composers_id:

        id = int(idx)

        if id not in composers_dict:

            composers_dict[id] = len(composers_dict);

            composers.loc[composers_dict[id]] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        composers.loc[composers_dict[id], 'composer_rank%d'%train.loc[i, 'label']] += 1;

        composers.loc[composers_dict[id], 'composer_track'] += 1;



for i in range(composers.shape[0]):

    for j in range(10):

        composers.loc[i, 'composer_rank%d'%(j + 1)] /= composers.loc[i, 'composer_track']



composers.head()
for i in range(train.shape[0]):

    composers_id = train.loc[i, 'composers_id'].replace(',', '.')

    composers_id = composers_id.split('.')

    num_composers = len(composers_id)

    for idx in composers_id:

        id = int(idx)

        train.loc[i, 'composer_total'] += composers.loc[composers_dict[id], 'composer_track']

        for j in range(10):

            num_track = composers.loc[composers_dict[id], 'composer_rank%d'%(j + 1)]

            train.loc[i, 'composer_rank%d'%(j + 1)] +=  num_track / num_composers

    train.loc[i, 'composer_track'] = train.loc[i, 'composer_total'] / num_composers
blanks = [composers[label].mean() for label in list(composers)]

print(blanks)
for i in range(test.shape[0]):

    composers_id = test.loc[i, 'composers_id'].replace(',', '.')

    composers_id = composers_id.split('.')

    num_composers = len(composers_id)

    for idx in composers_id:

        id = int(idx)

        if id not in composers_dict:

            composers_dict[id] = len(composers_dict);

            composers.loc[composers_dict[id]] = blanks

        test.loc[i, 'composer_total'] += composers.loc[composers_dict[id], 'composer_track']

        for j in range(10):

            num_track = composers.loc[composers_dict[id], 'composer_rank%d'%(j + 1)]

            test.loc[i, 'composer_rank%d'%(j + 1)] +=  num_track / num_composers

    test.loc[i, 'composer_track'] = test.loc[i, 'composer_total'] / num_composers
artists_columns = ['artist_rank%d'%(rank + 1) for rank in range(10)]

artists_columns.append('artist_track')

artists = pd.DataFrame(columns = artists_columns, dtype = np.int64)

artists.head()
artists_dict = {}

for i in range(train.shape[0]):

    artists_id = train.loc[i, 'artist_id'].replace(',', '.')

    artists_id = artists_id.split('.')

    for idx in artists_id:

        id = int(idx)

        if id not in artists_dict:

            artists_dict[id] = len(artists_dict);

            artists.loc[artists_dict[id]] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        artists.loc[artists_dict[id], 'artist_rank%d'%train.loc[i, 'label']] += 1;

        artists.loc[artists_dict[id], 'artist_track'] += 1;



for i in range(artists.shape[0]):

    for j in range(10):

        artists.loc[i, 'artist_rank%d'%(j + 1)] /= artists.loc[i, 'artist_track']



artists.head()
for i in range(train.shape[0]):

    artists_id = train.loc[i, 'artist_id'].replace(',', '.')

    artists_id = artists_id.split('.')

    num_artists = len(artists_id)

    for idx in artists_id:

        id = int(idx)

        train.loc[i, 'artist_total'] += artists.loc[artists_dict[id], 'artist_track']

        for j in range(10):

            num_track = artists.loc[artists_dict[id], 'artist_rank%d'%(j + 1)]

            train.loc[i, 'artist_rank%d'%(j + 1)] +=  num_track / num_artists

    train.loc[i, 'artist_track'] += train.loc[i, 'artist_total'] / num_artists
blanks = [artists[label].mean() for label in list(artists)]

print(blanks)
for i in range(test.shape[0]):

    artists_id = test.loc[i, 'artist_id'].replace(',', '.')

    artists_id = artists_id.split('.')

    num_artists = len(artists_id)

    for idx in artists_id:

        id = int(idx)

        if id not in artists_dict:

            artists_dict[id] = len(artists_dict);

            artists.loc[artists_dict[id]] = blanks

        test.loc[i, 'artist_total'] += artists.loc[artists_dict[id], 'artist_track']

        for j in range(10):

            num_track = artists.loc[artists_dict[id], 'artist_rank%d'%(j + 1)]

            test.loc[i, 'artist_rank%d'%(j + 1)] +=  num_track / num_artists

    test.loc[i, 'artist_track'] += test.loc[i, 'artist_total'] / num_artists
composers.describe()
artists.describe()
train.describe()
test.describe()
from matplotlib import pyplot as plt

corr = train.corr()

plt.matshow(corr)

plt.show()
train['year'] = 0

train['month'] = 0

train['day'] = 0

train['week_day'] = 0

train['time'] = 0

train['hour'] = 0

train['minute'] = 0
from datetime import datetime as dt



begin_year = 2017



for i in range(train.shape[0]):

    time = train.loc[i, 'release_time'].split()

    daytime = list(map(int, time[0].split('-')))

    hourtime = list(map(int, time[1].split(':')))

    

    train.loc[i, 'year'] = daytime[0] - begin_year

    train.loc[i, 'month'] = daytime[1]

    train.loc[i, 'day'] = daytime[2]

    train.loc[i, 'week_day'] = dt(daytime[0], daytime[1], daytime[2]).weekday()

    

    train.loc[i, 'hour'] = hourtime[0]

    train.loc[i, 'minute'] = hourtime[1]

    train.loc[i, 'time'] = hourtime[0] * 3600 + hourtime[1] * 60 + hourtime[2]
test['year'] = 0

test['month'] = 0

test['day'] = 0

test['week_day'] = 0

test['time'] = 0

test['hour'] = 0

test['minute'] = 0
from datetime import datetime as dt



begin_year = 2017



for i in range(test.shape[0]):

    time = test.loc[i, 'release_time'].split()

    daytime = list(map(int, time[0].split('-')))

    hourtime = list(map(int, time[1].split(':')))

    

    test.loc[i, 'year'] = daytime[0] - begin_year

    test.loc[i, 'month'] = daytime[1]

    test.loc[i, 'day'] = daytime[2]

    test.loc[i, 'week_day'] = dt(daytime[0], daytime[1], daytime[2]).weekday()

    

    test.loc[i, 'hour'] = hourtime[0]

    test.loc[i, 'minute'] = hourtime[1]

    test.loc[i, 'time'] = hourtime[0] * 3600 + hourtime[1] * 60 + hourtime[2]
plt.matshow(train.corr().to_numpy()[:-1, :-1])

plt.show()
train.corr()
trains = train

tests = test
train = trains

test = tests

train = train.drop(['composers_name', 'composers_id', 'artist_name', 'artist_id', 'title', 'release_time'], axis = 1)

test = test.drop(['composers_name', 'composers_id', 'artist_name', 'artist_id', 'title', 'release_time'], axis = 1)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64
# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(train, filename='train.csv')
# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(test, filename='test.csv')