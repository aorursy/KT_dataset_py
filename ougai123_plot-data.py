import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



read_data = pd.read_csv("../input/FX_USDJPY_201701till01312259_Train - USDJPY_201701till01312259 (1).csv",header=None)

read_data
plot_data = read_data[[2,3,4,5]]

plot_data.index = read_data[0]+","+read_data[1]

plot_data.plot()
nomal_data = plot_data/np.max(plot_data)

nomal_data.plot()
from pandas.tools.plotting import scatter_matrix

scatter_matrix(nomal_data)
plot_data.corr()
import matplotlib.pyplot as plt

from statsmodels.tsa import stattools



for i in plot_data.columns:

    acf = stattools.acf(plot_data[i], len(plot_data))

    plt.bar(range(len(acf)), acf, width = 1)