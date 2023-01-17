import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scaleogram as scg

import matplotlib.pyplot as plt



Xtr = pd.read_csv("../input/X_train.csv") # row_id series_id measurement_number orientation_X orientation_Y orientation_Z orientation_W 

ytr = pd.read_csv("../input/y_train.csv") #series_id group_id surface

Xtr.head(1)

ytr.head(1)

def plot_serie(df, id):

    surface = ytr[ytr.series_id==id].surface.values[0]

    fig = plt.figure(figsize=(16, 4))

    ax = plt.subplot(1, 3, 1)

    for orient in [ 'X', 'Y', 'Z', 'W']:

        X = df[df.series_id==id]['orientation_'+orient].values

        plt.plot(X - np.mean(X), label='o'+orient)

    plt.legend(); plt.title("Orientation (%s)" %(surface)); 

    

    plt.subplot(1, 3, 2)

    for orient in ['X', 'Y', 'Z']:

        X = df[df.series_id==id]['angular_velocity_'+orient].values

        plt.plot(X,label=orient)

    plt.legend(); plt.title("Angular velocity (%s)" %(surface))



    plt.subplot(1, 3, 3)

    for orient in ['X', 'Y', 'Z']:

        X = df[df.series_id==id]['linear_acceleration_'+orient].values

        plt.plot(X,label=orient)

    plt.legend(); plt.title("Linear acceleration (%s)" %(surface))



    fig = plt.figure(figsize=(18,9))

    for i, orient in enumerate(['X', 'Y', 'Z', 'W']):

        ax=plt.subplot(4, 3, 1+3*i)

        X = df[df.series_id==id]['orientation_'+orient].values

        scales = scg.periods2scales(10**np.linspace(np.log10(1), np.log10(60), 50))

        scg.cws(X-X.mean(), scales=scales, ax=ax, yscale='log', cscale='log', clim=(1e-6, 1e-4))

        i == 0 and ax.set_title('Orientation (%s)'%(surface))

    for i, orient in enumerate(['X', 'Y', 'Z']):

        ax=plt.subplot(4, 3, 2+3*i)

        X = df[df.series_id==id]['angular_velocity_'+orient].values

        scales = scg.periods2scales(10**np.linspace(np.log10(1), np.log10(60), 50))

        scg.cws(X-X.mean(), scales=scales, ax=ax, yscale='log', clim=(0, 0.1))

        i == 0 and ax.set_title('Angular Velocity (%s)'%(surface))

    for i, orient in enumerate(['X', 'Y', 'Z']):

        ax=plt.subplot(4, 3, 3+3*i)

        scales = scg.periods2scales(10**np.linspace(np.log10(1), np.log10(60), 50))

        X = df[df.series_id==id]['linear_acceleration_'+orient].values

        scg.cws(X-X.mean(), scales=scales, ax=ax, yscale='log', clim=(0,3))

        i ==0 and ax.set_title('Linear Acceleration (%s)' %(surface))

    

plot_serie(Xtr,0)

    
plot_serie(Xtr,1)

plot_serie(Xtr,2)

plot_serie(Xtr,3)

plot_serie(Xtr,4)
plot_serie(Xtr,5)

plot_serie(Xtr,6)

plot_serie(Xtr,7)

plot_serie(Xtr,8)

plot_serie(Xtr,9)

plot_serie(Xtr,10)
