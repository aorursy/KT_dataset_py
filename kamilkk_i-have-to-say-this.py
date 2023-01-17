import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
songs_extra = pd.read_csv('../input/song_extra_info.csv')



def isrc_to_year(isrc):

    if type(isrc) == str:

        if int(isrc[5:7]) > 17:

            return 1900 + int(isrc[5:7])

        else:

            return 2000 + int(isrc[5:7])

    else:

        return np.nan

        

songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)

songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)



train = train.merge(songs_extra, on = 'song_id', how = 'left')

test = test.merge(songs_extra, on = 'song_id', how = 'left')
train['2017_songs_frac'] = (train['song_year'] == 2017).rolling(window = 50000, center = True).mean()

test['2017_songs_frac'] = (test['song_year'] == 2017).rolling(window = 50000, center = True).mean()
plt.figure()

plt.plot(train.index.values, train['2017_songs_frac'], '-',

        train.shape[0] + test.index.values, test['2017_songs_frac'], '-');