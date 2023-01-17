import numpy as np

import pandas as pd

import sqlite3



conn = sqlite3.connect('../input/database v2.sqlite')

cur = conn.cursor()
q = 'SELECT * FROM PLAYER'

res = cur.execute(q).fetchall()

weight = np.array([float(r[6]) for r in res])

height = np.array([float(r[5]) for r in res])

bmi = (weight*0.453592)/((height/100)**2)
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(10,5))

plt.hist(bmi,25,normed=True)

plt.plot([np.mean(bmi),np.mean(bmi)],[0,0.35],c='r',linewidth=2)

plt.plot([25.4,25.4],[0,0.35],c='g',linewidth=2)

plt.legend(["Mean BMI Footballers","Total Mean BMI","Histogram"])

plt.title("Football player BMI Distribution")

plt.grid()

plt.show()