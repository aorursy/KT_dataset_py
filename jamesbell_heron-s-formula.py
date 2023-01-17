import numpy as np



def Heron(a, b, c):

    s = np.sum(np.array([a, b, c])) / 2

    return np.sqrt(s * (s - a) * (s - b) * (s - c)) 
Heron(29, 43, 19)