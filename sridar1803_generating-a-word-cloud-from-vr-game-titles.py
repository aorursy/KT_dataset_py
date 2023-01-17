# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt # plotting library

from wordcloud import WordCloud # library to generate word cloud

import csv
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read game titles into a list.

with open('../input/vr_titles.csv') as f:

    game_titles = f.readlines()

#clean up

game_titles = [x.strip() for x in game_titles] 

words = ' '.join(game_titles)

words = words.replace('VR', '')

words = words.title().replace('Game','').replace('Virtual', '').replace('Reality', '')

words = words.replace('Trial', '').replace('Edition','').replace('Htc', '').replace('Vive', '')
#plot

cloud = WordCloud(width=1440, height=1080).generate(words)

fig = plt.figure(figsize=(12, 9))

plt.imshow(cloud)

plt.axis('off')

fig.savefig('./vr_wordcloud.png')