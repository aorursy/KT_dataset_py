import scipy.io
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px

file="/kaggle/input/trends-assessment-prediction/fMRI_train/12857.mat"
f = h5py.File(file, 'r')
f['SM_feature']
NUM=33
plt.imshow(f['SM_feature'][1,:,:,:][NUM])
f['SM_feature'][1,:,:,:].shape
def plot_brain_plotly(Brain, NUM=1):
    X=[]
    Y=[]
    Z=[]
    COl=[]
    for z, image in tqdm(enumerate(Brain[NUM,:,:,:])):
        xx, yy = np.meshgrid(np.linspace(0,image.shape[1],image.shape[1]), np.linspace(0,image.shape[0],image.shape[0]))


        zz=np.ones(xx.shape)*z

        xx = xx[[image!=0][0]]
        yy = yy[[image!=0][0]]
        zz=np.ones(xx.shape)*z
        X+=list(xx)
        Y+=list(yy)
        Z+=list(zz)
        COl+=list(image[[image!=0][0]])

    fig = px.scatter_3d(x=X, y=Y, z=Z,color=COl,size=np.ones(len(X))*1, opacity = 1)
    fig.show()


Brain=f['SM_feature']

plot_brain_plotly(Brain)
def plot_brain_matplotlib(Brain, NUM=1):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X=[]
    Y=[]
    Z=[]
    COl=[]

    for z, image in tqdm(enumerate(Brain[NUM,:,:,:])):
        xx, yy = np.meshgrid(np.linspace(0,image.shape[1],image.shape[1]), np.linspace(0,image.shape[0],image.shape[0]))


        zz=np.ones(xx.shape)*z

        xx = xx[[image!=0][0]]
        yy = yy[[image!=0][0]]
        zz=np.ones(xx.shape)*z
        X+=list(xx)
        Y+=list(yy)
        Z+=list(zz)
        COl+=list(image[[image!=0][0]])

    ax.scatter(X,Y,Z,c = plt.cm.gist_heat(COl), s=2, alpha=0.4)



    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
Brain=f['SM_feature']
plot_brain_matplotlib(Brain)