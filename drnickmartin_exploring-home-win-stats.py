import numpy as np

import pandas as pd

import sqlite3



conn = sqlite3.connect('../input/database v2.sqlite')

cur = conn.cursor()
q = 'SELECT home_team_goal,away_team_goal FROM Match'

res = cur.execute(q).fetchall()
games = len(res)

wins = sum([r[0]>r[1] for r in res])

draws = sum([r[0]==r[1] for r in res])

loss = games-(wins+draws)
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(10,5)) 

percs = [wins/games*100,draws/games*100,loss/games*100]

labels = ["Wins","Draws","Losses"]

y_pos = np.arange(len(percs))

plt.barh(y_pos, percs, align='center', alpha=0.4)

plt.yticks(y_pos, labels)

plt.xlabel(r'%')

plt.title('Total Home Win Ratios')

plt.show()