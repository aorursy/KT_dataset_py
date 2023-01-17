!python -m pip install dtw

import numpy as np
from dtw import dtw

 
s1 = np.array([1,2,3,4,3,2,1,1,1,2]) 
s2 = np.array([0,1,1,2,3,4,3,2,1,1]) 



manhattan_distance = lambda s1,s2: np.abs(s1 - s2)

d, cost_matrix, acc_cost_matrix, path = dtw( s1,s2, dist=manhattan_distance)

print(d)
!pip install dtaidistance
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random
import numpy as np

x = np.arange(0, 20, .5)
s1 = np.sin(x)
s2 = np.sin(x - 1)
random.seed(1)
for idx in range(len(s2)):
    if random.random() < 0.05:
        s2[idx] += (random.random() - 0.5) / 2
d, paths = dtw.warping_paths(s1, s2, window=25, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(s1, s2, paths, best_path)

distance = dtw.distance(s1, s2)
print(distance)