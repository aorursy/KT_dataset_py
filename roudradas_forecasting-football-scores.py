import pandas as pd

import numpy as np

import seaborn as sns

import scipy

import matplotlib.pyplot as plt
def ScoreGrid(homeXg, awayXg):

    A = np.zeros(11)

    B = np.zeros(11)



    for i in range(10):

        A[i] = scipy.stats.poisson.pmf(i,homeXg)

        B[i] = scipy.stats.poisson.pmf(i,awayXg)

    

    A[10] = 1 - sum(A[:10])

    B[10] = 1 - sum(B[:10])



    #name = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]

    #zero = np.zeros(11)



    C = pd.DataFrame(np.zeros((11,11)))

    

    for j in range(11):

        for k in range(11):

            C[j][k] = A[j]*B[k]

    

    #C_new = C.rename(columns = name, index = name)



    return round(C*100,2)/100
ScoreGrid(1.7,1.1)
def ScoreHeatMap(home, away, homeXg, awayXg, datasource):

    adjustedHome = home.replace("_", " ")

    adjustedAway = away.replace("_", " ")



    df = ScoreGrid(homeXg, awayXg)



    ax = sns.heatmap(df, cbar = True, cmap = "OrRd")



    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)

    #ax.xaxis.tick_top()

    plt.xlabel(adjustedHome)

    plt.ylabel(adjustedAway)

    plt.title('Expected Scores')

    

    return ax
ScoreHeatMap("Barcelona", "Bayern_Munich", 1.2, 5.6, "FiveThirtyEight")
ScoreHeatMap("Paris_SG", "Bayern_Munich", 1.4,1.6,"FiveThirtyEight")
ScoreHeatMap("Leipzig", "Tottenham", 1.1, 0.3,"FiveThirtyEight")
ScoreHeatMap('Liverpool', "Atletico Madrid", 3.6, 1.1,"FiveThirtyEight")
ScoreHeatMap('Atalanta', "Valencia", 3.0, 2.1,"FiveThirtyEight")
ScoreHeatMap('Paris_SG', "Dortmund", 1.2, 0.4,"FiveThirtyEight")
ScoreHeatMap('Manchester City', "Real Madrid", 2.0, 0.7,"FiveThirtyEight")
ScoreHeatMap('Juventus', "Olympique Lyon", 2.7, 1.0,"FiveThirtyEight")
ScoreHeatMap('Barcelona', "Napoli", 1.2, 2.3,"FiveThirtyEight")
ScoreHeatMap('Bayern Munich', "Chelsea", 3.0, 1.0,"FiveThirtyEight")
ScoreHeatMap('Man. City', "Olympique Lyon", 3.1, 0.9, "FiveThirtyEight")
ScoreHeatMap('Leipzig', "Atletico Madrid", 0.9, 1.0,"FiveThirtyEight")
ScoreHeatMap('Barcelona', "Bayern Munich", 1.2, 5.6,"FiveThirtyEight")
ScoreHeatMap('Atalanta', "Paris SG", 0.5, 2.9,"FiveThirtyEight")
ScoreHeatMap('Olympique Lyon', "Bayern Munich", 1.6, 3.2,"FiveThirtyEight")
ScoreHeatMap('Leipzig', "Paris SG", 1.4, 3.7,"FiveThirtyEight")
ScoreHeatMap('Paris SG', "Bayern Munich", 1.4, 1.6,"FiveThirtyEight")