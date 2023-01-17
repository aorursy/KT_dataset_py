import numpy as np
from scipy.sparse import csc_matrix

def pageRank(G, s = .85, maxerr = .0001):
    n = G.shape[0]

    a = csc_matrix(G,dtype=np.float)
    sums = np.array(a.sum(1))[:,0]
    r, c = a.nonzero()
    a.data /= sums[r]

    sink = sums==0
    
    ro, rk = np.zeros(n), np.ones(n)
    while np.sum(np.abs(rk-ro)) > maxerr:
        ro = rk.copy()

        for i in range(0,n):
            ai = np.array(a[:,i].todense())[:,0]
            di = sink / float(n)
            ei = np.ones(n) / float(n)

            rk[i] = ro.dot( ai*s + di*s + ei*(1-s) )

    return rk/float(sum(rk))




if name=='main':
    G = np.array([[0,0.5,0,0,0.5,0],
                  [0.3,0,0,0.3,0,0.3],
                  [0,0,0,1,0,0],
                  [0,0.5,0,0,0.5,0.5],
                  [0,0.5,0,0,0.5,0],
                  [0,0.5,0,0,0.5,0]])
pageRank(G,s=.86)