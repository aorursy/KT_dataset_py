import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from wordcloud.wordcloud import WordCloud

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
overall_over_85 = data.loc[data.overall > 84]
wonderkids = data.loc[(data.potential > 84) & (data.age < 23)]
overall_over_85.plot(kind="scatter", x='headingaccuracy', y='jumping', alpha=0.5, color="red")

plt.xlabel('Heading Accuracy')

plt.ylabel('Jumping')

plt.show()
overall_over_85.plot(kind="scatter", x='slidingtackle', y='interceptions', alpha=0.5, color="yellow")

plt.xlabel('Sliding Tackle')

plt.ylabel('Interceptions')

plt.show()

overall_over_85.plot(kind="scatter", x='shotpower', y='longshots', alpha=0.5, color="magenta")

plt.xlabel('Shot Power')

plt.ylabel('Long Shots')

plt.show()
overall_over_85.plot(kind="scatter", x='volleys', y='finishing', alpha=0.5, color="green")

plt.xlabel('Volleys')

plt.ylabel('Finishing')

plt.show()
overall_over_85.plot(kind="scatter", x='longpassing', y='crossing', alpha=0.5, color="orange")

plt.xlabel('Long Passing')

plt.ylabel('Crossing')

plt.show()
overall_over_85.plot(kind="scatter", x='curve', y='fkaccuracy', alpha=0.5, color="purple")

plt.xlabel('Curve')

plt.ylabel('FK Accuracy')

plt.show()
wonderkids.overall.plot(kind='hist', bins = 25, figsize=(20,20))

plt.show()
wonderkids.potential.plot(kind='hist', bins =10, figsize=(20,20))

plt.show()
plt.subplots(figsize=(24,15))

#separating words of country names

country_name = wonderkids.nationality



wordcloud = WordCloud(background_color='white', width = 1920, height = 1080).generate(" ".join(country_name))



plt.imshow(wordcloud)

plt.axis('off')

plt.show()