# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



train_df = pd.read_csv('../input/conways-reverse-game-of-life-2020/train.csv')

test_df = pd.read_csv("../input/conways-reverse-game-of-life-2020/test.csv")
import matplotlib.pyplot as plt

import matplotlib as mpl

import numpy as np
def sampling(xs):

  for x in xs:

    #extraction pixel values for start and stop images

    start_x = train_df.loc[x, train_df.columns.str.startswith('start')]

    stop_x = train_df.loc[x, train_df.columns.str.startswith('stop')]

    #reshap array en matrice 25x25

    start_x = np.asarray(start_x).reshape(25, 25)

    stop_x = np.asarray(stop_x).reshape(25, 25)

    

    #calculation

    Invariant=start_x+stop_x

    Invariant[(Invariant<2) & (Invariant>0)]=0

    #print(Invariant)

    change=start_x-stop_x

    #print(change)

    variation=Invariant+change



    #iteration number extract

    delta=train_df.loc[x,'delta']

    

    #Display

    colors = ['red','white','blue','black']

    cmap = mpl.colors.ListedColormap(colors)



    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    ax[0] = plt.subplot2grid((1,3), (0,0), colspan=1)

    ax[1] = plt.subplot2grid((1,3), (0,1), colspan=1)

    ax[2] = plt.subplot2grid((1,3), (0,2), colspan=1)

    

    clipped01=ax[0].imshow(2*(start_x),cmap=cmap,vmin=-1,vmax=2,interpolation='none')

    ax[0].set_title("Start Board {0}".format(x))



    clipped00=ax[1].imshow(2*(stop_x),cmap=cmap,vmin=-1,vmax=2,interpolation='none')

    ax[1].set_title("Board after {0} time(s) step".format(delta))

       

    clipped02=ax[2].imshow(variation,cmap=cmap,vmin=-1,vmax=2,interpolation='none')

    ax[2].set_title("variation")

    plt.colorbar(clipped02,ax=ax[2])

    plt.show()

    

    print ("RED:Add, BLUE:Remove, BLACK: No change")

sampling([1])