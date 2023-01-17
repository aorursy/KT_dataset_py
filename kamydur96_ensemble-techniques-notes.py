import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# tossing coin in series of 10 attempts, for a total of 1000 times

heads_proba = 0.51



coin_tosses = (np.random.rand(10000,10)< heads_proba).astype(np.int32)

cummulative_head_ratio = np.cumsum(coin_tosses,axis=0) / np.arange(1,10001).reshape(-1,1)
plt.figure(figsize=(10,5))

plt.plot(cummulative_head_ratio)

plt.plot([0,10000],[.51,.51],'k--',linewidth =2,label='51%')

plt.plot([0,10000],[.50,.50],'k-',linewidth =2,label='50%')

plt.xlabel("No. of coin tosses", fontsize=12)

plt.ylabel("Heads Ratio", fontsize=12)

plt.axis([0, 10000, 0.42, 0.59])

plt.show()
import warnings

warnings.simplefilter(action='ignore',category='FutureWarning')
from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split



X,y = make_moons(n_samples=500,noise=0.3,random_state=42)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)