import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



village = pd.DataFrame({'Person':['Person_{}'.format(i) for i in range(1,11)], 

                        'Income_Village_1':[10]*10, 

                        'Income_Village_2':[5,5,5,10,10,10,10,15,15,15],

                        'Income_Village_3':[1,1,1,1,1,1,1,10,33,50]})

village['Cum_population'] =  np.cumsum(np.ones(10)/10)

village['Cum_Income_Village_1'] =  np.cumsum(village['Income_Village_1']/100)

village['Cum_Income_Village_2'] =  np.cumsum(village['Income_Village_2']/100)

village['Cum_Income_Village_3'] =  np.cumsum(village['Income_Village_3']/100)

village = village.iloc[:, [3,4,0,5,1,6,2,7]]

village
plt.figure(figsize = (8,8))

Gini=[]



for i in range(1,4):

    X_k = village['Cum_population'].values

    X_k_1 = village['Cum_population'].shift().fillna(0).values

    Y_k = village['Cum_Income_Village_{}'.format(i)].values

    Y_k_1 = village['Cum_Income_Village_{}'.format(i)].shift().fillna(0).values

    Gini.append(1 - np.sum((X_k - X_k_1) * (Y_k + Y_k_1)))

    plt.plot(np.insert(X_k,0,0), np.insert(village['Cum_Income_Village_{}'.format(i)].values,0,0),

             label='Village {} (Gini = {:0.2f})'.format(i, Gini[i-1]))

        

plt.title('Коэффициент Джини')

plt.xlabel('Кумулятивная доля населения')

plt.ylabel('Кумулятивная доля дохода')

plt.legend(loc="upper left")

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.show()
curve_area = np.trapz(np.insert(village['Cum_Income_Village_3'].values,0,0), np.insert(village['Cum_population'].values,0,0))

S = (0.5 - curve_area) / 0.5



plt.figure(figsize = (8,8))

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')

plt.plot(np.insert(village['Cum_population'].values,0,0), np.insert(village['Cum_Income_Village_3'].values,0,0),

             label='Village {} (Gini = {:0.2f})'.format(i, Gini[i-1]),lw = 2,color = 'green')



plt.fill_between(np.insert(X_k,0,0), np.insert(X_k,0,0), y2=np.insert(village['Cum_Income_Village_3'].values,0,0), alpha=0.5)

plt.text(0.45,0.27,'S = {:0.2f}'.format(S),fontsize = 28)

plt.title('Коэффициент Джини')

plt.xlabel('Кумулятивная доля населения')

plt.ylabel('Кумулятивная доля дохода')

plt.legend(loc="upper left")

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.show()