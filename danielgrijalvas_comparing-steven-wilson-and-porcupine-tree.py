import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 



sw = pd.read_csv('../input/Steven Wilson.csv') # Steven Wilson

pt = pd.read_csv('../input/Porcupine Tree.csv', nrows=len(sw)) # Porcupine Tree



# remove useless columns 

ignore = ['analysis_url', 'id', 'track_href', 'uri', 'type', 'album', 'name', 'artist', 'lyrics']

sw.drop(ignore, axis=1, inplace=True)

pt.drop(ignore, axis=1, inplace=True)



sw.describe()
# custom color palette 

red_blue = ['#19B5FE', '#EF4836']

palette = sns.color_palette(red_blue)

sns.set_palette(palette)
# let's compare the songs of SW and PT using histograms

fig = plt.figure(figsize=(15,15))



for i, feature in enumerate(sw):

    ax = plt.subplot(4,4,i+1)

    ax.set_title(feature)

    sw[feature].hist(alpha=0.7, label='Steven Wilson')

    pt[feature].hist(alpha=0.7, label='Porcupine Tree')

    

plt.legend(loc='upper right')