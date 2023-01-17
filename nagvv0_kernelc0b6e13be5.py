import pandas as pd

import numpy as np

%pylab inline

plt.style.use('seaborn-dark')

import warnings

warnings.filterwarnings("ignore") # отключение варнингов

pd.set_option('display.max_columns', None) # pd.options.display.max_columns = None 

# pd.set_option('display.max_rows', None) # не прятать столбцы при выводе дата-фреймов

import matplotlib.pyplot as plt

import matplotlib as mpl

plt.rc('font', size=14)
npdata = pd.read_csv('../input/train.csv').values
def week(day):

    return (day - 1)%7 + 1
def P(i, l):

    #return (l/2+i)/(l*l)

    #return i / ( l*l*2) +1/l - 1/(4*l)

    return i*2/(l*l)
def getM2(s):

    l = len(s)

    probs = [0]*8

    for i, day in enumerate(s):

        probs[day] += P(i, l)

    mv = max(probs)

    return probs.index(mv)
sol = []

for i, _ in enumerate(npdata):

    t = list(map(int, npdata[i, 1].split()))

    t = list(map(week, t))

    sol.append(getM2(t))

    
solw = array(sol)
solw = solw.transpose()
np.savetxt("mode.csv", np.dstack((np.arange(1, solw.size+1),solw))[0],"%d, %d",header="id,nextvisit") #добавляет лишний "# "