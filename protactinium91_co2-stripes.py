import pandas as pd



import os, io, requests



import functools



import numpy as np



import matplotlib.pyplot as plt



from matplotlib.colors import Normalize



import seaborn as sns



data = pd.read_csv("../input/archive.csv")



savename = 'co2_stripes'



ydata_temp=data

## Yearly mean

#ydata_temp = data.groupby(np.arange(len(data))//12).mean()



ydata = ydata_temp.iloc[:,[0,3]]

ydata
co2data =ydata.iloc[:,1]

co2data
co2data = co2data[~np.isnan(co2data)]

co2data
co2data = co2data.reset_index(drop=True)

co2data
co2_normed = ((co2data - co2data.min(0)) / co2data.ptp(0)) * (len(co2data) - 1)

co2_normed




elements = len(co2data)







x_lbls = np.arange(elements)



y_vals = co2_normed / (len(co2data) - 1)



y_vals2 = np.full(elements, 1)



bar_wd  = 1







my_cmap = plt.cm.Spectral_r #choose colormap to use for bars



norm = Normalize(vmin=0, vmax=elements - 1)
def colorval(num):



    return my_cmap(norm(num))







fig=plt.figure(figsize=(12,5))



plt.axis('off')



plt.axis('tight')







#Plot co2 stripes. 



plt.bar(x_lbls, y_vals2, color = list(map(colorval, co2_normed)), width=1.0)







#Plot co2 line. Comment out to only plot stripes



plt.plot(x_lbls, y_vals - 0.002, color='white', linewidth=3)







plt.xticks( x_lbls + bar_wd, x_lbls)



plt.ylim(0, 1)



fig.subplots_adjust(bottom = 0)



fig.subplots_adjust(top = 1)



fig.subplots_adjust(right = 1.005)



fig.subplots_adjust(left = 0)



fig.savefig(savename+'.png', dpi=300)