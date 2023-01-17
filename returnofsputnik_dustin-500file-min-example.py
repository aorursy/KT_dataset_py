import numpy as np
for i in range(2000):

    np.save(str(i)+'npy',np.array(i))