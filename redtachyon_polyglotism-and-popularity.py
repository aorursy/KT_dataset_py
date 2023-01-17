from collections import defaultdict



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.DataFrame.from_csv('../input/survey_results_public.csv')
lang_counts = defaultdict(int)

all_languages = set()

user_count = defaultdict(int)

popularity = defaultdict(int)

total_users = 0



# Getting some stats on languages' popularity

for i in range(1, 51393):

    try:

        languages = data["HaveWorkedLanguage"][i].split('; ')

        total_users += 1

    except AttributeError: 

        continue

    

    count = len(languages) - 1

    for lang in languages:

        all_languages.add(lang)

        lang_counts[lang] += count

        user_count[lang] += 1

    

    #print('%d: ' % i, languages)

    

means = defaultdict(float)

for i in all_languages:

    means[i] = lang_counts[i] / user_count[i]

    

ndata = [[i, means[i]] for i in means]

pdata = [[i, user_count[i]] for i in user_count]



ndata.sort(key=lambda x: x[1], reverse=True)

pdata.sort(key=lambda x: x[1], reverse=True)



ndata = np.array(ndata)

pdata = np.array(pdata)
df = pd.DataFrame(data=ndata[:, 1],

                 index=ndata[:, 0], 

                 columns=['Polyglotism'])



df['Popularity'] = pd.Series(data=pdata[:, 1],

                            index=pdata[:, 0])



df['Polyglotism'] = df['Polyglotism'].astype(np.float32)

df['Popularity'] = df['Popularity'].astype(np.float32) / total_users



ndf = df.sort_values(by=['Popularity'], ascending=False)
fig = plt.figure(figsize=(13, 8)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes

ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.



width = 0.35



ndf['Popularity'].plot(kind='bar', color='red', ax=ax, width=width, position=0)

ndf['Polyglotism'].plot(kind='bar', color='blue', ax=ax2, width=width, position=1)



ax.set_ylabel('Popularity')

ax2.set_ylabel('Polyglotism')



ax.yaxis.label.set_color('red')

ax.yaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontweight(1000)

ax2.yaxis.label.set_color('blue')

ax2.yaxis.label.set_fontsize(16)

ax2.yaxis.label.set_fontweight(1000)



plt.show()
fig = plt.figure(figsize=(13, 8)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes

ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.



width = 0.35



df['Popularity'].plot(kind='bar', color='red', ax=ax, width=width, position=0)

df['Polyglotism'].plot(kind='bar', color='blue', ax=ax2, width=width, position=1)



ax.set_ylabel('Popularity')

ax2.set_ylabel('Polyglotism')



ax.yaxis.label.set_color('red')

ax.yaxis.label.set_fontsize(16)

ax.yaxis.label.set_fontweight(1000)

ax2.yaxis.label.set_color('blue')

ax2.yaxis.label.set_fontsize(16)

ax2.yaxis.label.set_fontweight(1000)



plt.show()