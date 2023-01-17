#  Packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

from matplotlib import pyplot as plt

from sklearn.neighbors import KDTree

from scipy.stats.stats import pearsonr

from scipy.stats import rankdata

#  Data

train = pd.read_csv('../input/train_2016_v2.csv')

props = pd.read_csv('../input/properties_2016.csv',low_memory=False)

train = train.merge(props,how='left',on='parcelid')

train = train[['parcelid','longitude','latitude','logerror']]

train.dropna(inplace=True)  #

del props  # delete redundant data

gc.collect()  # Free up memory

print("DataFrame sample:")

print("***************************************************")

train.head()

print("***************************************************")

print("shape = ",train.shape)
def get_pde(train,bw):

    x = train['longitude'].values

    y = train['latitude'].values

    xy = np.vstack([x,y])

    X = np.transpose(xy)

    tree = KDTree(X,leaf_size = 20 )     

    parcelDensity = tree.kernel_density(X, h=bw,kernel='gaussian',rtol=0.00001)

    return parcelDensity
parcelDensity30000 = get_pde(train,30000)

parcelDensity10000 = get_pde(train,10000)

parcelDensity3000 = get_pde(train,3000)

parcelDensity1000 = get_pde(train,1000)

parcelDensity300 = get_pde(train,300)

plt.figure(figsize=(14,14))

plt.axis("off")

plt.title("Gaussian Parcel Density Estimate at bandwidth 30,000")

plt.scatter(train['longitude'].values, train['latitude'].values, c=parcelDensity30000,cmap='inferno', s=1, edgecolor='')

rankScaled30000 = 100*rankdata(parcelDensity30000)/len(parcelDensity30000)

rankScaled10000 = 100*rankdata(parcelDensity10000)/len(parcelDensity10000)

rankScaled3000 = 100*rankdata(parcelDensity3000)/len(parcelDensity3000)

rankScaled1000 = 100*rankdata(parcelDensity1000)/len(parcelDensity1000)

rankScaled300 = 100*rankdata(parcelDensity300)/len(parcelDensity300)
fig = plt.figure(figsize=(15,15))



ax1 = fig.add_subplot(221)

ax1.set_title('bandwidth = 10,000')

ax1.set_axis_off()

ax1.scatter(train['longitude'].values, train['latitude'].values, c=rankScaled10000,cmap='inferno', s=1, edgecolor='')



ax2 = fig.add_subplot(222)

ax2.set_title('bandwidth = 3,000')

ax2.set_axis_off()

ax2.scatter(train['longitude'].values, train['latitude'].values, c=rankScaled3000,cmap='inferno', s=1, edgecolor='')



ax3 = fig.add_subplot(223)

ax3.set_title('bandwidth = 1,000')

ax3.set_axis_off()

ax3.scatter(train['longitude'].values, train['latitude'].values, c=rankScaled1000,cmap='inferno', s=1, edgecolor='')



ax4 = fig.add_subplot(224)

ax4.set_title('bandwidth = 300')

ax4.set_axis_off()

ax4.scatter(train['longitude'].values, train['latitude'].values, c=rankScaled300,cmap='inferno', s=1, edgecolor='')

abs_logerrors = np.abs(train['logerror'].values)

def moving_average(a, n=3) :

    ret = np.cumsum(a, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n
fig = plt.figure(figsize=(15,15))

fig.suptitle("Parcel Density Vs. Logerror")

ax1, ax2= fig.add_subplot(221),fig.add_subplot(222)

x1,x2 = parcelDensity30000, rankScaled30000

index1, index2 = x1.argsort(), x2.argsort()

x1 = x1[index1[::-1]]

x2 = x2[index2[::-1]]

y = abs_logerrors

y = y[index1[::-1]]

y_av = moving_average(y,n=250)

y_av = [0]*249 + list(y_av)



m, b = np.polyfit(x1,y , 1)

ax1.plot(x1, y, '.',alpha=0.5,color='skyblue')

ax1.plot(x1,y_av,linewidth=6,color="steelblue")

ax1.plot(x1, m*x1 + b, '--',linewidth=3,color='red')



ax1.set_xlim([-0.000000001,0.00000025])

ax1.set_ylim([-0.0,2])

ax1.set_ylabel("logerror",fontsize='large')

ax1.set_xlabel("PDE",fontsize='large')



m, b = np.polyfit(x2,y , 1)

ax2.plot(x2, y, '.',alpha=0.5,color='skyblue')

ax2.plot(x2,y_av,linewidth=6,color="steelblue")

ax2.plot(x2, m*x2 + b, '--',linewidth=3,color='red')

ax2.set_xlabel("PDE - ranked",fontsize='large')

ax2.set_ylabel("Logerror",fontsize='large')

ax2.set_xlim([0,100])

ax2.set_ylim([-0.0,2])



corrCoef_30000_1, p_twoTailed_30000_1 = pearsonr(x1,y)

corrCoef_30000_2, p_twoTailed_30000_2 = pearsonr(x2,y)

p_oneTailed_30000_1 = 1 - (p_twoTailed_30000_1/2)



print("Result for bandwidth 30,000:")

print("*******************************************************************")

print("Correlation Coefficient: ",corrCoef_30000_1)

print("Two tailed_p: ",p_twoTailed_30000_1)

print("One tailed p for negative correlation: ",p_oneTailed_30000_1)

print("*******************************************************************")
fig = plt.figure(figsize=(15,15))

x1,x2,x3,x4 = parcelDensity10000, parcelDensity3000, parcelDensity1000,parcelDensity300

y = abs_logerrors



index1, index2,index3,index4 = x1.argsort(), x2.argsort(), x3.argsort(), x4.argsort()

x1, x2, x3, x4 = x1[index1[::-1]],  x2[index2[::-1]],  x3[index3[::-1]], x4[index4[::-1]]

x1 = x1 - min(x1)

x2 = x2 - min(x2)

x3 = x3 - min(x3)

x4 = x4 - min(x4)



y1, y2, y3, y4 = y[index1[::-1]], y[index2[::-1]],y[index3[::-1]],y[index4[::-1]]

y_av1 = moving_average(y1,n=100)

y_av1 = [0]*99 + list(y_av1)

y_av2 = moving_average(y2,n=100)

y_av2 = [0]*99 + list(y_av2)

y_av3 = moving_average(y3,n=100)

y_av3 = [0]*99 + list(y_av3)

y_av4 = moving_average(y4,n=100)

y_av4 = [0]*99 + list(y_av4)



ax1 = fig.add_subplot(221)

m, b = np.polyfit(x1,y1 , 1)

ax1.plot(x1, y1, '.',alpha=0.5,color='skyblue')

ax1.plot(x1,y_av1,linewidth=6,color="steelblue")

ax1.plot(x1, m*x1 + b, '--',linewidth=3,color='red')

ax1.set_xlim([0,0.0000005])

ax1.set_ylim([-0.0,2])

ax1.set_ylabel("logerror",fontsize='large')

ax1.set_xlabel("PDE",fontsize='large')



ax2 = fig.add_subplot(222)

m, b = np.polyfit(x2,y2 , 1)

ax2.plot(x2, y2, '.',alpha=0.5,color='skyblue')

ax2.plot(x2,y_av2,linewidth=4,color="steelblue")

ax2.plot(x2, m*x2 + b, '--',linewidth=3,color='red')

ax2.set_xlim([0,0.0000015])

ax2.set_ylim([-0.0,2])

ax2.set_ylabel("logerror",fontsize='large')

ax2.set_xlabel("PDE",fontsize='large')



ax3 = fig.add_subplot(223)

m, b = np.polyfit(x3,y3 , 1)

ax3.plot(x3, y3, '.',alpha=0.5,color='skyblue')

ax3.plot(x3,y_av3,linewidth=4,color="steelblue")

ax3.plot(x3, m*x3 + b, '--',linewidth=3,color='red')

ax3.set_xlim([-0,0.000005])

ax3.set_ylim([-0.0,2])

ax3.set_ylabel("logerror",fontsize='large')

ax3.set_xlabel("PDE",fontsize='large')



ax4 = fig.add_subplot(224)

m, b = np.polyfit(x4,y4 , 1)

ax4.plot(x4, y, '.',alpha=0.5,color='skyblue')

ax4.plot(x4,y_av4,linewidth=4,color="steelblue")

ax4.plot(x4, m*x4 + b, '--',linewidth=3,color='red')

ax4.set_xlim([-0.000000001,0.00004])

ax4.set_ylim([-0.0,2])

ax4.set_ylabel("logerror",fontsize='large')

ax4.set_xlabel("PDE",fontsize='large')



corrCoef_10000, p_twoTailed_10000 = pearsonr(x1,y1)

p_oneTailed_10000 = p_twoTailed_10000/2

corrCoef_3000, p_twoTailed_3000 = pearsonr(x2,y2)

p_oneTailed_3000 = p_twoTailed_3000/2

corrCoef_1000, p_twoTailed_1000 = pearsonr(x3,y3)

p_oneTailed_1000 = p_twoTailed_1000/2

corrCoef_300, p_twoTailed_300 = pearsonr(x4,y4)

p_oneTailed_300 = p_twoTailed_300/2



print("For BW 10,000, Correlation Coefficient: ",corrCoef_10000)

print("For BW 10,000, One tailed_p: ",p_oneTailed_10000)

print("**********************************************************")

print("For BW 3,000, Correlation Coefficient: ",corrCoef_3000)

print("For BW 3,000, One tailed_p: ",p_oneTailed_3000)

print("**********************************************************")

print("For BW 1,000, Correlation Coefficient: ",corrCoef_1000)

print("For BW 1,000, One tailed_p: ",p_oneTailed_1000)

print("**********************************************************")

print("For BW 500, Correlation Coefficient: ",corrCoef_300)

print("For BW 500, One tailed_p: ",p_oneTailed_300)

#  Data

train = pd.read_csv('../input/train_2017.csv')

props = pd.read_csv('../input/properties_2017.csv',low_memory=False)

train = train.merge(props,how='left',on='parcelid')

train = train[['parcelid','longitude','latitude','logerror']]

train.dropna(inplace=True)  #

del props  # delete redundant data

gc.collect()  # Free up memory

parcelDensity30000 = get_pde(train,30000)

parcelDensity10000 = get_pde(train,10000)

parcelDensity3000 = get_pde(train,3000)

parcelDensity1000 = get_pde(train,1000)

parcelDensity300 = get_pde(train,300)
rankScaled30000 = 100*rankdata(parcelDensity30000)/len(parcelDensity30000)

rankScaled10000 = 100*rankdata(parcelDensity10000)/len(parcelDensity10000)

rankScaled3000 = 100*rankdata(parcelDensity3000)/len(parcelDensity3000)

rankScaled1000 = 100*rankdata(parcelDensity1000)/len(parcelDensity1000)

rankScaled300 = 100*rankdata(parcelDensity300)/len(parcelDensity300)
logerrors = train['logerror'].values

y = logerrors
fig = plt.figure(figsize=(15,15))

fig.suptitle("Parcel Density Vs. Logerror")

ax1, ax2= fig.add_subplot(221),fig.add_subplot(222)

x1,x2 = parcelDensity30000, rankScaled30000

index1, index2 = x1.argsort(), x2.argsort()

x1 = x1[index1[::-1]]

x2 = x2[index2[::-1]]

y = y[index1[::-1]]

y_av = moving_average(y,n=250)

y_av = [0]*249 + list(y_av)



m, b = np.polyfit(x1,y , 1)

ax1.plot(x1, y, '.',alpha=0.5,color='skyblue')

ax1.plot(x1,y_av,linewidth=6,color="steelblue")

ax1.plot(x1, m*x1 + b, '--',linewidth=3,color='red')



ax1.set_xlim([-0.000000001,0.00000025])

ax1.set_ylim([-2.0,2])

ax1.set_ylabel("logerror",fontsize='large')

ax1.set_xlabel("PDE",fontsize='large')



m, b = np.polyfit(x2,y , 1)

ax2.plot(x2, y, '.',alpha=0.5,color='skyblue')

ax2.plot(x2,y_av,linewidth=6,color="steelblue")

ax2.plot(x2, m*x2 + b, '--',linewidth=3,color='red')

ax2.set_xlabel("PDE - ranked",fontsize='large')

ax2.set_ylabel("Logerror",fontsize='large')

ax2.set_xlim([0,100])

ax2.set_ylim([-2.0,2])



corrCoef_30000_1, p_twoTailed_30000_1 = pearsonr(x1,y)



print("Result for bandwidth 30,000:")

print("*******************************************************************")

print("Correlation Coefficient: ",corrCoef_30000_1)

print("Two tailed_p: ",p_twoTailed_30000_1)
fig = plt.figure(figsize=(15,15))

x1,x2,x3,x4 = parcelDensity10000, parcelDensity3000, parcelDensity1000,parcelDensity300

y = logerrors



index1, index2,index3,index4 = x1.argsort(), x2.argsort(), x3.argsort(), x4.argsort()

x1, x2, x3, x4 = x1[index1[::-1]],  x2[index2[::-1]],  x3[index3[::-1]], x4[index4[::-1]]

x1 = x1 - min(x1)

x2 = x2 - min(x2)

x3 = x3 - min(x3)

x4 = x4 - min(x4)



y1, y2, y3, y4 = y[index1[::-1]], y[index2[::-1]],y[index3[::-1]],y[index4[::-1]]

y_av1 = moving_average(y1,n=100)

y_av1 = [0]*99 + list(y_av1)

y_av2 = moving_average(y2,n=100)

y_av2 = [0]*99 + list(y_av2)

y_av3 = moving_average(y3,n=100)

y_av3 = [0]*99 + list(y_av3)

y_av4 = moving_average(y4,n=100)

y_av4 = [0]*99 + list(y_av4)



ax1 = fig.add_subplot(221)

m, b = np.polyfit(x1,y1 , 1)

ax1.plot(x1, y1, '.',alpha=0.5,color='skyblue')

ax1.plot(x1,y_av1,linewidth=6,color="steelblue")

ax1.plot(x1, m*x1 + b, '--',linewidth=3,color='red')

ax1.set_xlim([0,0.0000005])

ax1.set_ylim([-02.0,2])

ax1.set_ylabel("logerror",fontsize='large')

ax1.set_xlabel("PDE",fontsize='large')



ax2 = fig.add_subplot(222)

m, b = np.polyfit(x2,y2 , 1)

ax2.plot(x2, y2, '.',alpha=0.5,color='skyblue')

ax2.plot(x2,y_av2,linewidth=4,color="steelblue")

ax2.plot(x2, m*x2 + b, '--',linewidth=3,color='red')

ax2.set_xlim([0,0.0000015])

ax2.set_ylim([-02.0,2])

ax2.set_ylabel("logerror",fontsize='large')

ax2.set_xlabel("PDE",fontsize='large')



ax3 = fig.add_subplot(223)

m, b = np.polyfit(x3,y3 , 1)

ax3.plot(x3, y3, '.',alpha=0.5,color='skyblue')

ax3.plot(x3,y_av3,linewidth=4,color="steelblue")

ax3.plot(x3, m*x3 + b, '--',linewidth=3,color='red')

ax3.set_xlim([-0,0.000005])

ax3.set_ylim([-2.0,2])

ax3.set_ylabel("logerror",fontsize='large')

ax3.set_xlabel("PDE",fontsize='large')



ax4 = fig.add_subplot(224)

m, b = np.polyfit(x4,y4 , 1)

ax4.plot(x4, y, '.',alpha=0.5,color='skyblue')

ax4.plot(x4,y_av4,linewidth=4,color="steelblue")

ax4.plot(x4, m*x4 + b, '--',linewidth=3,color='red')

ax4.set_xlim([-0.000000001,0.00004])

ax4.set_ylim([-02.0,2])

ax4.set_ylabel("logerror",fontsize='large')

ax4.set_xlabel("PDE",fontsize='large')



corrCoef_10000, p_twoTailed_10000 = pearsonr(x1,y1)

corrCoef_3000, p_twoTailed_3000 = pearsonr(x2,y2)

corrCoef_1000, p_twoTailed_1000 = pearsonr(x3,y3)

corrCoef_300, p_twoTailed_300 = pearsonr(x4,y4)

print("For BW 10,000, Correlation Coefficient: ",corrCoef_10000)

print("For BW 10,000,  2-tailed_p: ",p_twoTailed_10000)

print("**********************************************************")

print("For BW 3,000, Correlation Coefficient: ",corrCoef_3000)

print("For BW 3,000,  2-tailed_p: ",p_twoTailed_3000)

print("**********************************************************")

print("For BW 1,000, Correlation Coefficient: ",corrCoef_1000)

print("For BW 1,000, 2-tailed_p: ",p_twoTailed_1000)

print("**********************************************************")

print("For BW 500, Correlation Coefficient: ",corrCoef_300)

print("For BW 500, tailed_p: ",p_twoTailed_300)
