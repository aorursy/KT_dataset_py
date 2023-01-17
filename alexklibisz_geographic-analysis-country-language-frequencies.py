import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



M = pd.read_csv('../input/members.csv')

S = pd.read_csv('../input/songs.csv')

E = pd.read_csv('../input/song_extra_info.csv')

TRN = pd.read_csv('../input/train.csv')

print('Done reading')
# Join S and E to include name and isrc in the same dataframe.

# There's probably a cleaner/faster way to join, but this is good enough for now.

S['language'] = S['language'].fillna(-2)

E['isrc'] = E['isrc'].fillna('na')

S = S.set_index('song_id', drop=False)

E = E.set_index('song_id', drop=False)

SE = S.join(E[['name', 'isrc']])

SE.head()
# Extract the country codes and languages for all songs.

# Plot them as a co-occurrence frequency heat-map. This ends up being very sparse.

# Most of the country codes can be looked up here:

# https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#Officially_assigned_code_elements

codes = sorted(set([str(x)[:2] for x in E['isrc']]))

langs = sorted(set(S['language']))

code2idx = {x:i for i,x in enumerate(codes)}

lang2idx = {x:i for i,x in enumerate(langs)}

freq = np.zeros((len(codes), len(langs)))

for i,r in SE.iterrows():

    cci = code2idx[str(r['isrc'])[:2]]

    lgi = lang2idx[r['language']]

    freq[cci,lgi] += 1



plt.figure(figsize=(len(langs),len(codes)//3))

plt.imshow(freq, interpolation='none', aspect='auto')

plt.title('Song count for each (country code, language) pair')

plt.ylabel('Country Code')

plt.xlabel('Language')

plt.yticks(range(len(codes)), codes)

plt.xticks(range(len(langs)), langs)

plt.colorbar()

plt.show()        