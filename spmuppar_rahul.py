# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/rahul.csv")
data.head()
data
#plt.scatter(data["WFS"][:3], data["Bead Width"][:3])
fig, axs = plt.subplots(3,3, figsize=(12, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .0025, wspace=.001)

axs = axs.ravel()
for i in range(3, 28, 3):
    axs[int(i/3-1)].plot(data["WFS"][i-3:i], data["Bead Width"][i-3:i], '-o')
    axs[int(i/3)-1].set_ylim([0,17])
    #axs[int(i/3)-1].errorbar(data["WFS"][i-3:i], data["Bead Width"][i-3:i], yerr=[yerr_lower, 2*yerr], xerr=xerr, fmt='o', ecolor='g', capthick=2)
    if i<20:
        axs[int(i/3)-1].set_xticks([])
    if i<10:
        axs[int(i/3)-1].set_title("Voltage"+ str(data["Voltage"][i-1]))
    if i != 3 and i!=12 and i!=21  :
        axs[int(i/3)-1].set_yticks([])
    if i == 9 or i==18 or i==27  :
        axs[int(i/3)-1].text(600,10, "TS "+str(data["TS"][i-1]))
        
data_vg.head()
