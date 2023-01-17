import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

%config InlineBackend.figure_format = 'svg' # using svg makes plots clearer

%matplotlib inline
plt.rcParams["figure.figsize"] = (13.0,8.0) # cell must be called in a different cell than the import for it to take effect
# loading data

def load_data(h = False):

    if h:   

        temp = np.load('/kaggle/input/alphabet-characters-fonts-dataset/character_fonts (with handwritten data).npz')

        return temp['images'], temp['labels']

    else: 

        temp = np.load('/kaggle/input/alphabet-characters-fonts-dataset/character_font.npz')

        return temp['images'], temp['labels']
images, labels = load_data() # loading data without handwritten digits

print(f'Images shape: {images.shape}')

print(f'Labels shape: {labels.shape}')
images, labels = load_data(True) # loading data with handwritten digits

print(f'Images shape: {images.shape}')

print(f'Labels shape: {labels.shape}')
# showing data

# # of rows, # of columns, organzied, tight layout

# set organized to true to get an alphabet for each row

def show_images(nrows=10,ncols = 10, o = True, t= False):

    # organized, random, same

    plt.close()

    fig = plt.figure(figsize=(13,13)) # Notice the equal aspect ratio

    ax = [plt.subplot(nrows,ncols,i+1) for i in range(nrows*ncols)]

    if not o:

        temp = np.random.choice(images.shape[0],nrows*ncols,replace=False)

        for i,a in enumerate(ax):

            a.imshow(images[temp[i]])

            a.axis('off')

            a.set_aspect('equal')

    else:

        assert nrows <= 26

        temp = np.random.choice(np.arange(26),ncols,replace=False)

        count = -1

        l = []

        for i in temp:

            l.append(np.random.choice(np.argwhere(labels == i).ravel(),ncols, replace=False))

        for i,a in enumerate(ax):

            if i%(ncols) == 0:

                count+=1

            a.imshow(images[l[count][i%ncols]])

            a.axis('off')

            a.set_aspect('equal')

    if t:

        fig.tight_layout()

    fig.subplots_adjust(wspace=0.0, hspace=0.05)

    fig.show()

show_images()