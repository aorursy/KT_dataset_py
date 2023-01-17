# All imports

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,7)

import numpy as np

import skimage.io



#Set up base levels

slide_dir = "../input/prostate-cancer-grade-assessment/train_images/"

annotation_dir = "../input/prostate-cancer-grade-assessment/train_label_masks/"

example_id = "0032bfa835ce0f43a92ae0bbab6871cb"

example_slide = f"{slide_dir}{example_id}.tiff"



# Open slide on lowest resolution

low_res_lvl = -1

img_low = skimage.io.MultiImage(example_slide)[low_res_lvl]

plt.imshow(img_low)

plt.show()
def plot_tiles(tile_list):

    """

    Description

    ----------

    Plot a list of tissue tiles four wide

    Credit: https://stackoverflow.com/a/11172032



    Parameters

    ----------

    tile_list: list 

        List of tiles to be ploted

    Returns(0)

    ----------

    """

    

    nrows = -(-len(tile_list)//4)

    ncols = 4

    figures = {f"Tile: {i+1}":tile for i,tile in enumerate(tile_list)}

    

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)

    for ind, title in enumerate(figures):

        axeslist.ravel()[ind].imshow(figures[title], aspect="auto")

        axeslist.ravel()[ind].set_title(title)

    plt.tight_layout()

    plt.show()

    return 
def tile(img):

    shape = img.shape

    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                 constant_values=255)

    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

    if len(img) < N:

        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

    img = img[idxs]

    return img



N=12

sz=128

tiles = tile(img_low)

plot_tiles(tiles)
def tile_plot(base_image, N=12, sz=128, plot=True):

    """

    Description

    __________

    Tilizer module made by @iafoss that can be found in the notebook:

    https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-inference

    Takes a base image and returns the N tiles with the largest differnce

    from a white backgound each with a given square size of input-sz.

    

    Parameters

    __________

    base_image: numpy array

        Image array to split into tiles and plot

    N: int

        This is the number of tiles to split the image into

    sz: int

        This is the size for each side of the square tiles

    plot: bool

        True to show plot of choosen tiles, False for silent return

    

    Returns

    __________

    - List of size N with each item being a numpy array tile.

    """

    

    #Get the shape of the input image

    shape = base_image.shape

    

    #Find the padding such that the image divides evenly by the desired size

    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

    

    #Pad the image with blank space to reach the above found tagrgets

    base_img = np.pad(base_image,[[pad0//2,pad0-pad0//2],

                                  [pad1//2,pad1-pad1//2],[0,0]],

                                     constant_values=255)

    

    #Reshape and Transpose to get the images into tiles

    all_tiles = base_img.reshape(base_img.shape[0]//sz,sz,base_img.shape[1]//sz,sz,3)

    all_tiles = all_tiles.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

    

    #If there are not enough tiles to meet desired N pad again

    if len(all_tiles) < N:

        all_tiles = np.pad(all_tiles,[[0,N-len(all_tiles)],[0,0],[0,0],[0,0]],

                                       constant_values=255) 

    

    #Sort the images by those with the lowest sum (i.e the least white)

    #Return indexes to the lowest N 

    idxs = np.argsort(all_tiles.reshape(all_tiles.shape[0],-1).sum(-1))[:N]

    

    #Slect by index those returned from the above funtion

    tissue_tiles = all_tiles[idxs]

    

    if plot:

        #Funciton for plotting

        line_color=[0,255,255]

        line_sz=5

        #Get the deminsions in terms of slides

        tile_cnt_size = [base_img.shape[0]//sz, base_img.shape[1]//sz]

        

        #Iterate through all images and change the boarder on the slected images

        prod_tiles = []

        for i,img in enumerate(all_tiles):

            if i in idxs:

                #If image is in the slected slides change the..

                #Left

                img[:,:line_sz,:] =  [[line_color]*line_sz]*sz

                #Right

                img[:,-line_sz:,:] =  [[line_color]*line_sz]*sz

                #Top

                img[:line_sz] =  [[line_color]*sz]*line_sz

                #Bottom

                img[-line_sz:] =  [[line_color]*sz]*line_sz

                #... boarders to the specified color

            prod_tiles.append(img)

            

        #Piece the tiles back into one image

        #Split the array of tiles into a list of rows

        rows = np.array_split(prod_tiles,tile_cnt_size[0])

        #Horizontally combine rows

        row_combine = [np.hstack(tiles) for tiles in rows]

        #Vertically stack rows back into base image

        prod_image = np.vstack(row_combine)

        #Display image

        plt.imshow(prod_image)

        plt.show()

    return tissue_tiles



tiles = tile_plot(img_low, plot=True)