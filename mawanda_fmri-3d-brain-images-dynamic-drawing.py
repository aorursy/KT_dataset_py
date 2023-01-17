import h5py

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from ipywidgets import interact_manual  # interaction for sliders

import ipywidgets as widgets  # slider widgets
file="/kaggle/input/trends-assessment-prediction/fMRI_train/12857.mat"

f = h5py.File(file, 'r')
brain = f['SM_feature']  # this is the only key the header has
# Extract the shape of the image

max_channels, max_height, max_width, max_depth = brain.shape



print("Dimensions of the brain: \t\nchannels: {}\t\nheigth: {}\t\nwidth: {}\t\ndepth: {}".format(max_channels, max_height, max_width, max_depth))



channel = 32

height = 32

width = np.arange(max_width)

depth = np.arange(max_depth)



print("Picture dimension : {}".format(brain[channel][height].shape))



plt.imshow(brain[channel][height][width])
def create_coordinates(C, X_r, Y_r, Z_r):

    """

    Useful function that extract the coordinates that are going to be scattered inside the 3D plot. 

    :param C: the selected channel

    :param X_r: (x_min, x_max) range that is going to be represented

    :param Y_r: (y_min, y_max) range that is going to be represented

    :param X_r: (x_min, x_max) range that is going to be represented

    """

    # Prepare the meshgrid information

    X=[]

    Y=[]

    Z=[]

    heatmap=[]

    brain_slice = brain[C, X_r[0]:X_r[1], Y_r[0]:Y_r[1], Z_r[0]:Z_r[1]]

    for z, image in tqdm(enumerate(brain_slice), desc='Loading...'):

        # Iterate over the first dimension (vertical view). In this way, we analyze each slice of the image along the first dimension, and create 

        # a heatmap (of the actual values of the image) of the selected layer of dimension (width, depth).

        xx, yy = np.meshgrid(np.linspace(Z_r[0],Z_r[1],Z_r[1] - Z_r[0]), np.linspace(Y_r[0],Y_r[1],Y_r[1] - Y_r[0]))

        z = z + X_r[0]

        zz=np.ones(xx.shape)*z

        xx = xx[[image!=0][0]]

        yy = yy[[image!=0][0]]

        zz=np.ones(xx.shape)*z

        X+=list(xx)

        Y+=list(yy)

        Z+=list(zz)

        heatmap+=list(image[[image!=0][0]])

    

    return X, Y, Z, heatmap
def plot_brain_matplotlib(C, X_r, Y_r, Z_r):

    """

    This function is called at each interaction of the widget. It is responsible of reading the image, calculating the graph and showing it.

    

    """

    # TODO: put the plot declaration outside this function so it is not loaded at every slider selection.

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111, projection='3d')



    ax.set_xlabel('x')

    ax.set_xlim([0, X_dim])

    ax.set_ylabel('y')

    ax.set_ylim([0, Y_dim])

    ax.set_zlabel('z')

    ax.set_zlim([0, Z_dim])

    

    X, Y, Z, heatmap = create_coordinates(C, X_r, Y_r, Z_r)

    ax.scatter(X, Y, Z, c = plt.cm.gist_heat(heatmap), s=1, alpha=1)
C_dim, X_dim, Y_dim, Z_dim = brain.shape  # load shape to set the limits



interact_manual(

    plot_brain_matplotlib, 

    C=widgets.Dropdown(options=np.arange(C_dim), description='Channel selection'),

    X_r=widgets.IntRangeSlider(value=[0, X_dim-1], min=0, max=X_dim-1, step=1, description='X axis'),

    Y_r=widgets.IntRangeSlider(value=[0, Y_dim-1], min=0, max=Y_dim-1, step=1, description='Y axis'),

    Z_r=widgets.IntRangeSlider(value=[24, 28], min=0, max=Z_dim-1, step=1, description='Z axis')

)