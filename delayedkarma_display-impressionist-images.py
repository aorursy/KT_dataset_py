import os

import random



%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import warnings

warnings.filterwarnings('ignore')
# Root directory



root = '/kaggle/input/impressionist-classifier-data/'
# Creates artist-specific directory names, and files storing the image names for each artist



artist_lst = ['Cezanne', 'Degas', 'Gauguin', 'Hassam', 'Matisse', 

              'Monet', 'Pissarro', 'Renoir', 'Sargent', 'VanGogh']



for artist in artist_lst:

    exec(f"train_{artist}_dir = os.path.join(root, 'training/training', '{artist}')")

    exec(f"train_{artist}_filenames = os.listdir(train_{artist}_dir)")

    exec(f"valid_{artist}_dir = os.path.join(root, 'validation/validation', '{artist}')")

    exec(f"valid_{artist}_filenames = os.listdir(valid_{artist}_dir)")
def plot_imgs(artist, nrows=4, ncols=4, num_imgs=8):

    """

    Function to plot random sample images for each artist in a num_rows x num_cols grid

    :param artist: Artist name

    :type artist: str

    :param nrows: Number of rows in grid

    :type nrows: int

    :param ncols: Number of columns

    :type ncols: int

    :param num_imgs: Number of sample images to plot

    :type num_imgs: int

    :return: None

    """

    

    pic_idx = 0

    

    fig = plt.gcf()

    fig.set_size_inches(ncols * 6, nrows * 6)



    pic_idx += num_imgs



    train_dir = eval(f"train_{artist}_dir")

    filenames = eval(f"train_{artist}_filenames")

    filenames = random.sample(filenames, len(filenames))

    

    next_pix = [os.path.join(train_dir, fname) 

                    for fname in filenames[pic_idx-num_imgs: pic_idx]]





    for i, img_path in enumerate(next_pix):

        plt.suptitle(f"{artist}", fontsize=24)

        sp = plt.subplot(nrows, ncols, i + 1)

        sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)

        plt.imshow(img)
plot_imgs(artist_lst[0])
plot_imgs(artist_lst[1])
plot_imgs(artist_lst[2])
plot_imgs(artist_lst[3])
plot_imgs(artist_lst[4])
plot_imgs(artist_lst[5])
plot_imgs(artist_lst[6])
plot_imgs(artist_lst[7])
plot_imgs(artist_lst[8])
plot_imgs(artist_lst[9])