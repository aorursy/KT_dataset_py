import os

os.listdir('../input')
import numpy as np

import matplotlib.pyplot as plt

df = np.loadtxt('../input/cardiovascular-disease-dataset/cardio_train.csv', delimiter=';', skiprows=1)

df.dtype
fig, ax = plt.subplots()

zdor=df[df[:,-1]==0, 4]

nezdor=df[df[:,-1]==1,4]

plt.bar([1, 2], height=[zdor.mean(), nezdor.mean()], color=['orange', 'green'])

ax.set_xticks([1, 2])

ax.set_xticklabels(['zdor', 'nezdor']);
fig, ax = plt.subplots()

zdor1=df[(df[:,-1]==0) & (df[:,7]==1)]

nezdor1=df[(df[:,-1]==1) & (df[:,7]==1)]

zdor2=df[(df[:,-1]==0) & (df[:,7]==2)]

nezdor2=df[(df[:,-1]==1) & (df[:,7]==2)]

zdor3=df[(df[:,-1]==0) & (df[:,7]==3)]

nezdor3=df[(df[:,-1]==1) & (df[:,7]==3)]

plt.bar([1, 2, 3, 4, 5, 6], height=[len(zdor1), len(zdor2), len(zdor3), len(nezdor1), len(nezdor2), len(nezdor3)], color=['orange', 'orange', 'orange', 'green', 'green', 'green'],label='zdor'),

ax.set_xticklabels([" " ,"1","2","3","1","2","3"])

ax.legend();
