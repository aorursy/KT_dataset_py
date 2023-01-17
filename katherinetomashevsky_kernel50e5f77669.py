# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as pl
import scipy.stats
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import plotly.graph_objects as go
import seaborn as sns
import math
data = pd.read_csv("../input/covid19dataset/COVID_Data.csv")
france = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,3562)], nrows=55)
france2 = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,3592)], nrows=25)
#france = france[france.Confirmed != 0]
#france[france.Confirmed != 0.0]
russia = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,8129)], nrows=19)
iran = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,4668)], nrows=30)
turkey = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,15869)], nrows=34)
US = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,10129)], nrows=59)
US2 = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,10164)], nrows=24)
kuwait = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,5524)], nrows=24)
Malaysia = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,6045)], nrows=54)
sk = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,5351)], nrows=59)
italy = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,4990)], nrows=28)
israel = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,4911)], nrows=27)
china = pd.read_csv("../input/covid19dataset/COVID_Data.csv", skiprows=[i for i in range(1,1988)], nrows=96)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
france.head()
plt.scatter(france.index, france['Confirmed'])
plt.scatter(US.index, US['Confirmed'])
def linBF(ds):

    V = ds['Confirmed']
#print(V)

    Vlin = []


    for x in V:
        if x != 0:
            Vlin.append(math.log(x))
        else:
            Vlin.append(0)

    plt.scatter(ds.index, Vlin)


  
    

    bf = np.polyfit(ds.index, Vlin, 1)
    bf2 = np.polyfit(ds.index, Vlin, 1)
    print(bf)

    bfx = [0,len(Vlin)]
    bfy = [bf[1], bfx[1] * bf[0] + bf[1]]

#print("slope of best fit line: "+str(bf[0]))


    plt.plot(bfx, bfy)
linBF(france)
linBF(france2)
linBF(russia)
linBF(US)
linBF(US2)
linBF(iran)
linBF(kuwait)
linBF(Malaysia)
linBF(china)
linBF(israel)
#GDP pc
plt.scatter([11373,33994,5628,62794,11289,41463,31363,34483,41715], [.078,.11,.314,.241,.247,.288,.18,.27,.257])
#GDP
plt.scatter([358.6,140.6,454,20540,1658,2778,1619,2084,370.6], [.078,.11,.314,.241,.247,.288,.18,.27,.257])
#pop density
plt.scatter([95,239,51,36,8.4,117,503,206,400], [.078,.11,.314,.241,.247,.288,.18,.27,.257])