import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import pandas as pd

import os

from matplotlib import pyplot as plt

from matplotlib import style



style.use('ggplot')







q = []

total_word_dict = {}

episode_word_dict = {}

s = {}

good = 0

bad = 0



x = []

y = []



xnum = 0



with open('../input/sum.json') as f:

    data_json = f.read()

    data_loads = json.loads(data_json)

    #a = d['s23e02']['results']['items']

    for episode in data_loads:

        episode_word_dict[episode] = {}

        #print(episode)

        #print(episode_word_dict)

        items = data_loads[episode]['results']['items']

        

        for entry in items:

            sub_entry = entry['alts']

            if sub_entry[0]['cfd'] != None:

                if float(sub_entry[0]['cfd']) > 0.9850: #Other values to try? 0.9900, 0.9800, 0.9850

                    good = good + 1

                    word = sub_entry[0]['ct']

                    lower_word = word.lower()

                    

                    if lower_word not in total_word_dict:

                        total_word_dict[lower_word] = 1

                    else:

                        total_word_dict[lower_word] = total_word_dict[lower_word] + 1



                    if lower_word not in episode_word_dict[episode]:

                        episode_word_dict[episode][lower_word] = 1

                    else:

                        episode_word_dict[episode][lower_word] = episode_word_dict[episode][lower_word] + 1

                        

                    

                    #s[z] = [c]

                    #if c not in s[z]:

                    #    s[z][c] = 1

                    

                else:

                    bad = bad + 1;

        

#for w in sorted(r, key=r.get, reverse=True):

    #print(w, r[w])    

    

#print(good)

#print(bad)

#print(total_word_dict)

#print(episode_word_dict)



for episode in episode_word_dict:

    xnum += 1

    print(episode_word_dict[episode]["love"])

    y.append(episode_word_dict[episode]["love"])

    x.append(xnum)



plt.bar(x, y, align='center')

plt.title('Mentions of "Love" per Episode')

plt.ylabel('Mentions')

plt.xlabel('Episode')

plt.show()

    



print("")



#for episode in episode_word_dict:

#    print(episode_word_dict[episode]["happy"])

    

    



#I want to see episode;word;count



#print(total_word_dict)

#print(episode_word_dict["s23e08"]["love"])

#print(episode_word_dict[1])
