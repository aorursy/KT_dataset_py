# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from concorde.tsp import TSPSolver
from matplotlib import collections as mc
import time
import pylab as pl

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cities = pd.read_csv('../input/cities.csv')
solver = TSPSolver.from_data(
cities.X,
cities.Y,
norm='EUC_2D')

t = time.time()
tour_data = solver.solve(time_bound=60.0, verbose=True, random_seed=42)
print(time.time() - t)
print(tour_data.found_tour)
pd.DataFrame({'Path':np.append(tour_data.tour,[0])}).to_csv('submission.csv', index=False)
print(tour_data.tour,[0])
a = pd.read_csv('submission.csv')
a.head()
# Plot tour
lines = [[(cities.X[tour_data.tour[i]],cities.Y[tour_data.tour[i]]),(cities.X[tour_data.tour[i+1]],cities.Y[tour_data.tour[i+1]])] for i in range(0,len(cities)-1)]
lc = mc.LineCollection(lines, linewidths=2)
fig, ax = pl.subplots(figsize=(20,20))
ax.set_aspect('equal')
ax.add_collection(lc)
ax.autoscale()
