import matplotlib

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

import scipy.interpolate
data1=np.load('/kaggle/input/depression-rest-eeg-features/507_Depression_REST-epo-feat-v1.npy',allow_pickle=True)

type(data1)
data1.shape,data1[0].shape,data1[0][0].shape
Features ="Min, Max, STD, Mean, Median, Activity, Mobility, Complexity, Kurtosis, 2nd Difference Mean, 2nd Difference Max, 1st Difference Mean, 1st Difference Max, Coeffiecient of Variation, Skewness, Wavelet Approximate Mean, Wavelet Approximate Std Deviation, Wavelet Detailed Mean, Wavelet Detailed Std Deviation, Wavelet Approximate Energy, Wavelet Detailed Energy, Wavelet Approximate Entropy, Wavelet Detailed Entropy, Mean of Vertex to Vertex Slope, Var of Vertex to Vertex Slope, FFT Delta Max Power, FFT Theta Max Power, FFT Alpha Max Power, FFT Beta Max Power, Delta/Alpha, Delta/Theta"

print(len(Features.split(", ")))

features=Features.split(", ")
chs = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2']

len(chs)
df0=pd.DataFrame(data1[0][0],columns=features,index=chs)

df0.shape
df=df0.reset_index().copy()

df.rename(columns={'index':'channel'},inplace=True)

df.head()
BioSemi64 = pd.read_csv("../input/channel-loc/BioSemi64.csv",header=None,usecols=None)

BioSemi64.columns=['num','x','y','channel']

BioSemi64.head()
mn=BioSemi64.merge(df[['channel','Mean']],on='channel')

mn.shape

# mn
df[(~df.channel.isin(mn.channel))] #elctrodes not specified in BioSemi64 file
BioSemi64[(~BioSemi64.channel.isin(mn.channel))] #Electrodes not specified in the data
#changed the electrode position to cartesian coordinate. Though there is a direct way to plot from polar coordinates

def pol2cart(x, y):

    xx=[]

    yy=[]

    for i in range(0,58):

#         print(i)

        xx.append(y[i] * np.cos(np.radians(x[i])))

        yy.append(y[i] * np.sin(np.radians(x[i])))

    return(xx,yy)

xx,yy=pol2cart(mn['x'].tolist(), mn['y'].tolist())
N=300

z = mn['Mean']*100000



xi = np.linspace(np.min(xx), np.max(xx), N)

yi = np.linspace(np.min(yy), np.max(yy), N)

zi = scipy.interpolate.griddata((xx, yy), z, (xi[None,:], yi[:,None]), method='cubic')
xy_center = [0,0]   # center of the plot

radius =0.45          # radius



# set points > radius to not-a-number. They will not be plotted.

# the dr/2 makes the edges a bit smoother

dr = xi[1] - xi[0]

for i in range(N):

    for j in range(N):

        r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)

        if (r - dr/2) > radius:

            zi[j,i] = "nan"



# make figure

fig = plt.figure()



# set aspect = 1 to make it a circle

ax = fig.add_subplot(111, aspect = 1)



# use different number of levels for the fill and the lines

CS = ax.contourf(xi, yi, zi, 60, cmap = plt.cm.jet, zorder = 1)

ax.contour(xi, yi, zi, 15, colors = "grey", zorder = 2)



# make a color bar

cbar = fig.colorbar(CS, ax=ax)



# add the data points

# I guess there are no data points outside the head...

ax.scatter(xx, yy, marker = 'o', c = 'b', s = 15, zorder = 3)

for i, txt in enumerate(mn['channel'].tolist()):

    ax.annotate(txt, (xx[i], yy[i]))



# Add some body parts. Hide unwanted parts by setting the zorder low

# add two ears

circle = matplotlib.patches.Ellipse(xy = [0,-0.45], width = 0.1, height = 0.05, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)

ax.add_patch(circle)

circle = matplotlib.patches.Ellipse(xy = [0,0.45], width = 0.1, height = 0.05, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)

ax.add_patch(circle)

# add a nose

# xy = [[-0.05,0.425], [0,0.475],[0.05,0.425]]

xy = [[0.425,-0.05], [0.475,0.0],[0.425,0.05]]

polygon = matplotlib.patches.Polygon(xy = xy,edgecolor = "k", facecolor = "w", zorder = 0)

ax.add_patch(polygon) 

plt.show()

from IPython.display import Image

Image("../input/electrode-location-fig/Electrode_location.jpg",height=300,width=300)