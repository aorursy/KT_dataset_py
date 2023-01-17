import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
num = np.random.random(40)
num
plt.pcolormesh(num.reshape(10,4))
plt.colorbar()
