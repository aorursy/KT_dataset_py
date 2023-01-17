from IPython.display import YouTubeVideo

YouTubeVideo('Uz3JIp6EvIg', width=800, height=600 )
import numpy as np

import pandas as pd



X_present = np.array([[200], [120], [180]])

X_present
states = ['A', 'B', 'C']

P = np.array([[.8, .2, .1], [.1, .7, .3], [.1 , .1, .6]])

print(pd.DataFrame(P, index = states, columns  = states))
X_1 = np.dot(P, X_present)

X_1
X_2 = np.dot(P, X_1)

X_2
X_3 = np.dot(P, X_2)

X_3