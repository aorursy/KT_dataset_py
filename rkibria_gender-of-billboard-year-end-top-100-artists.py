import collections

import string



table = collections.defaultdict(lambda: None)

table.update({

    ord('é'):'e',

    ord('ô'):'o',

    ord(' '):' ',

    ord('\N{NO-BREAK SPACE}'): ' ',

    ord('\N{EN SPACE}'): ' ',

    ord('\N{EM SPACE}'): ' ',

    ord('\N{THREE-PER-EM SPACE}'): ' ',

    ord('\N{FOUR-PER-EM SPACE}'): ' ',

    ord('\N{SIX-PER-EM SPACE}'): ' ',

    ord('\N{FIGURE SPACE}'): ' ',

    ord('\N{PUNCTUATION SPACE}'): ' ',

    ord('\N{THIN SPACE}'): ' ',

    ord('\N{HAIR SPACE}'): ' ',

    ord('\N{ZERO WIDTH SPACE}'): ' ',

    ord('\N{NARROW NO-BREAK SPACE}'): ' ',

    ord('\N{MEDIUM MATHEMATICAL SPACE}'): ' ',

    ord('\N{IDEOGRAPHIC SPACE}'): ' ',

    ord('\N{IDEOGRAPHIC HALF FILL SPACE}'): ' ',

    ord('\N{ZERO WIDTH NO-BREAK SPACE}'): ' ',

    ord('\N{TAG SPACE}'): ' ',

    })

table.update(dict(zip(map(ord,string.ascii_uppercase), string.ascii_lowercase)))

table.update(dict(zip(map(ord,string.ascii_lowercase), string.ascii_lowercase)))

table.update(dict(zip(map(ord,string.digits), string.digits)))



def cleanArtistName(s):

	result = s.lower().strip().translate(table,)

	result = result.replace('$', 's')

	result = result.replace('!', 'i')

	result = result.replace('-', ' ')

	result = result.replace('.', '')

	return result

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dfG = pd.read_csv("../input/singersgender/singers_gender.csv", encoding='latin1')

dfG.artist = dfG.artist.apply(cleanArtistName)

dfG.drop('category', axis=1, inplace=True)

dfG.rename(index=str, columns={"artist": "Artist"}, inplace=True)

print('-' * 10, '\ndfG head:\n', dfG.head())



dfB = pd.read_csv("../input/billboard-lyrics/billboard_lyrics_1964-2015.csv", encoding='latin1', parse_dates=['Year'])

dfB.Artist = dfB.Artist.apply(cleanArtistName)

dfB.drop('Lyrics', axis=1, inplace=True)

dfB.drop('Source', axis=1, inplace=True)

print('-' * 10, '\ndfB head:\n', dfB.head())

dfM = pd.merge(dfB, dfG, on='Artist', how='left')

dfM.set_index(['Year'], inplace=True)

dfM.sort_index(inplace=True)

print('-' * 10, '\ndfM head:\n', dfM.head())
srG = dfM['gender'].value_counts(dropna=False)

dfC = pd.DataFrame(srG)

dfC.rename(index=str, columns={"gender": "count"}, inplace=True)

dfC.rename(index={"nan": "UNKNOWN"}, inplace=True)



numCases = dfC['count'].sum()

dfC['percent'] = dfC['count'] / numCases * 100



print(dfC)

allYears = {'year': []}

for i in range(1965, 2016):

	srYearCounts = dfM.loc[str(i)]['gender'].value_counts(dropna=False)

	allYears['year'].append(i)

	yearDict = srYearCounts.to_dict()

	for key,value in yearDict.items():

		if not key in allYears.keys():

			allYears[key] = []

		allYears[key].append(value)



dfY = pd.DataFrame(allYears)

dfY = dfY.set_index(['year'])

print('-' * 10, '\ndfY head:\n', dfY.head())

import matplotlib.pyplot as plt

dfY.plot()

plt.show()

dfM.to_csv('OUTPUT_billboard_genders.csv')
