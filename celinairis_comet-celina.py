import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



data_set = pd.read_csv("../input/anime.csv")



data_set
top20 = data_set.head(20)

top20
r = top20['rating']

np.mean(r)
top20['type'].value_counts()
e = top20['episodes']

r = top20['rating']

e = pd.to_numeric(e, errors='coerce')

r = pd.to_numeric(r, errors='coerce')



print(np.mean(e))

print(np.max(e))

print(np.min(e))



df = pd.DataFrame({'rating': r, 'episodes': e})

df.corr('pearson')
m = top20['members']

r = top20['rating']

m = pd.to_numeric(m, errors='coerce')

r = pd.to_numeric(r, errors='coerce')



print(np.mean(m))

print(np.max(m))

print(np.min(m))



df = pd.DataFrame({'rating': r, 'members': m})

df.corr('pearson')
genre = top20['genre']



print(genre.str.contains('Action').sum())

print(genre.str.contains('Adventure').sum())

print(genre.str.contains('Cars').sum())

print(genre.str.contains('Comedy').sum())

print(genre.str.contains('Dementia').sum())

print(genre.str.contains('Demons').sum())

print(genre.str.contains('Drama').sum())

print(genre.str.contains('Ecchi').sum())

print(genre.str.contains('Fantasy').sum())

print(genre.str.contains('Game').sum())

print(genre.str.contains('Harem').sum())

print(genre.str.contains('Hentai').sum())

print(genre.str.contains('Historical').sum())

print(genre.str.contains('Horror').sum())

print(genre.str.contains('Josei').sum())

print(genre.str.contains('Kids').sum())

print(genre.str.contains('Magic').sum())

print(genre.str.contains('Martial Arts').sum())

print(genre.str.contains('Mecha').sum())

print(genre.str.contains('Military').sum())

print(genre.str.contains('Music').sum())

print(genre.str.contains('Mystery').sum())

print(genre.str.contains('Parody').sum())

print(genre.str.contains('Police').sum())

print(genre.str.contains('Psychological').sum())

print(genre.str.contains('Romance').sum())

print(genre.str.contains('Samurai').sum())

print(genre.str.contains('School').sum())

print(genre.str.contains('Sci-Fi').sum())

print(genre.str.contains('Seinen').sum())

print(genre.str.contains('Shoujo').sum())

print(genre.str.contains('Shoujo Ai').sum())

print(genre.str.contains('Shounen').sum())

print(genre.str.contains('Shounen Ai').sum())

print(genre.str.contains('Slice of Life').sum())

print(genre.str.contains('Space').sum())

print(genre.str.contains('Sports').sum())

print(genre.str.contains('Super Power').sum())

print(genre.str.contains('Supernatural').sum())

print(genre.str.contains('Thriller').sum())

print(genre.str.contains('Vampire').sum())

print(genre.str.contains('Yaoi').sum())

print(genre.str.contains('Yuri').sum())
top20Members = data_set.sort_values('members', ascending=False).head(20)

top20Members
r = top20Members['rating']

np.mean(r)
genre = top20Members['genre']



print(genre.str.contains('Action').sum())

print(genre.str.contains('Adventure').sum())

print(genre.str.contains('Cars').sum())

print(genre.str.contains('Comedy').sum())

print(genre.str.contains('Dementia').sum())

print(genre.str.contains('Demons').sum())

print(genre.str.contains('Drama').sum())

print(genre.str.contains('Ecchi').sum())

print(genre.str.contains('Fantasy').sum())

print(genre.str.contains('Game').sum())

print(genre.str.contains('Harem').sum())

print(genre.str.contains('Hentai').sum())

print(genre.str.contains('Historical').sum())

print(genre.str.contains('Horror').sum())

print(genre.str.contains('Josei').sum())

print(genre.str.contains('Kids').sum())

print(genre.str.contains('Magic').sum())

print(genre.str.contains('Martial Arts').sum())

print(genre.str.contains('Mecha').sum())

print(genre.str.contains('Military').sum())

print(genre.str.contains('Music').sum())

print(genre.str.contains('Mystery').sum())

print(genre.str.contains('Parody').sum())

print(genre.str.contains('Police').sum())

print(genre.str.contains('Psychological').sum())

print(genre.str.contains('Romance').sum())

print(genre.str.contains('Samurai').sum())

print(genre.str.contains('School').sum())

print(genre.str.contains('Sci-Fi').sum())

print(genre.str.contains('Seinen').sum())

print(genre.str.contains('Shoujo').sum())

print(genre.str.contains('Shoujo Ai').sum())

print(genre.str.contains('Shounen').sum())

print(genre.str.contains('Shounen Ai').sum())

print(genre.str.contains('Slice of Life').sum())

print(genre.str.contains('Space').sum())

print(genre.str.contains('Sports').sum())

print(genre.str.contains('Super Power').sum())

print(genre.str.contains('Supernatural').sum())

print(genre.str.contains('Thriller').sum())

print(genre.str.contains('Vampire').sum())

print(genre.str.contains('Yaoi').sum())

print(genre.str.contains('Yuri').sum())
e = top20Members['episodes']

r = top20Members['rating']

e = pd.to_numeric(e, errors='coerce')

r = pd.to_numeric(r, errors='coerce')



print(np.mean(e))

print(np.max(e))

print(np.min(e))



df = pd.DataFrame({'rating': r, 'episodes': e})

df.corr('pearson')
m = top20Members['members']

r = top20Members['rating']

m = pd.to_numeric(m, errors='coerce')

r = pd.to_numeric(r, errors='coerce')



print(np.mean(m))

print(np.max(m))

print(np.min(m))



df = pd.DataFrame({'rating': r, 'members': m})

df.corr('pearson')
genre = data_set['genre']



print(genre.str.contains('Action').sum())

print(genre.str.contains('Adventure').sum())

print(genre.str.contains('Cars').sum())

print(genre.str.contains('Comedy').sum())

print(genre.str.contains('Dementia').sum())

print(genre.str.contains('Demons').sum())

print(genre.str.contains('Drama').sum())

print(genre.str.contains('Ecchi').sum())

print(genre.str.contains('Fantasy').sum())

print(genre.str.contains('Game').sum())

print(genre.str.contains('Harem').sum())

print(genre.str.contains('Hentai').sum())

print(genre.str.contains('Historical').sum())

print(genre.str.contains('Horror').sum())

print(genre.str.contains('Josei').sum())

print(genre.str.contains('Kids').sum())

print(genre.str.contains('Magic').sum())

print(genre.str.contains('Martial Arts').sum())

print(genre.str.contains('Mecha').sum())

print(genre.str.contains('Military').sum())

print(genre.str.contains('Music').sum())

print(genre.str.contains('Mystery').sum())

print(genre.str.contains('Parody').sum())

print(genre.str.contains('Police').sum())

print(genre.str.contains('Psychological').sum())

print(genre.str.contains('Romance').sum())

print(genre.str.contains('Samurai').sum())

print(genre.str.contains('School').sum())

print(genre.str.contains('Sci-Fi').sum())

print(genre.str.contains('Seinen').sum())

print(genre.str.contains('Shoujo').sum())

print(genre.str.contains('Shoujo Ai').sum())

print(genre.str.contains('Shounen').sum())

print(genre.str.contains('Shounen Ai').sum())

print(genre.str.contains('Slice of Life').sum())

print(genre.str.contains('Space').sum())

print(genre.str.contains('Sports').sum())

print(genre.str.contains('Super Power').sum())

print(genre.str.contains('Supernatural').sum())

print(genre.str.contains('Thriller').sum())

print(genre.str.contains('Vampire').sum())

print(genre.str.contains('Yaoi').sum())

print(genre.str.contains('Yuri').sum())
rating = data_set['rating']

members = data_set['members']

rating = pd.to_numeric(rating, errors='coerce')

members = pd.to_numeric(members, errors='coerce')



df = pd.DataFrame({'rating': rating, 'members': members})

df.corr('pearson')