import random

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
win_rate = 60

top_rate = 50
stars_needed = {10:2,9:2,8:2,7:3,6:3,5:3,4:4,3:4,2:4}

irrev_rank = {10:True, 9:True, 8:False,7:True,6:False,5:False, 4:True, 3:False, 2:False, 1:True}
title = "Win Rate: " + str(win_rate) + "%, Finish Top: " + str(top_rate) + "%"

results = []



#1,000 simulations

for x in range(1000):

    rank = 10

    stars = 0

    games = 0

    while rank > 1:

        games = games + 1

        n = random.randrange(1,100)

        #Loss or draw

        if n > win_rate:

            #second chance if top on xp

            n = random.randrange(1,100)

            if n > top_rate:

                #if not you lose a star

                stars = stars - 1

                if stars <0:

                    #irrevocable rank so stay on that rank

                    if irrev_rank[rank]:

                        stars = 0

                    #otherwise demoted

                    else:

                        rank = rank + 1

                        stars = stars_needed[rank]-1

        #Win

        else:

            stars = stars + 1

            if stars >= stars_needed[rank]:

                stars = 0

                rank = rank - 1



    results.append(games)



#calculate the average

mn = round(np.mean(results))

average = 'Average (' + str(int(mn)) + ')'



#Plotting

plt.figure(figsize=(12,6))

plt.grid()

plt.xlabel('Total Games Played')

plt.title(title)

sns.kdeplot(results,shade=True)

plt.annotate(average,xy = (mn,0),xytext=(mn+10,0.005),

    arrowprops=dict(facecolor='black',shrink=0.05))

plt.tight_layout()

plt.show()