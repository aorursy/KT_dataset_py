import numpy as np

import os

img_file_path = "../input/guodian-xingzimingchu/"

img_files = sorted(os.listdir(img_file_path))

print("List of image files: ", img_files)



from skimage import io, dtype_limits



def load_image(file_path_name):

    img = io.imread(file_path_name)

    return img



img_number = 5 # there are six images for XZMC

source_img = load_image(img_file_path + img_files[img_number]) # load one image from list of files

source_img_markup = source_img.copy() # make copy



import matplotlib.pyplot  as plt





fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(source_img, cmap='gray')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))    

                                            # ARGS: 1/ number of rows in the grid; 2/ number of cols in each row

                                            # RETURNS a figure obj and an (array of) axes obj(s)



ax[0].hist(source_img.ravel(),256,(0,255)) # first arg = unravelled image array

                                         # second arg = number of bins (or ranges of indivual bins if a sequence)

                                         # third arg = range (from, to)

ax[0].set(title="full range")



ax[1].hist(source_img.ravel(),256,(0,200))

ax[1].set(title="darker end, < 200")

plt.show()


def get_white_bands(img): 

    '''

    Returns a list of left and right vertical boundaries of bands of white space flanking strips. 

    There should be one more such band than the number of strips, and twice that number of boundaries in the list.   

    

    ASSUMPTIONS

    * strips are close to straight and vertical

    * A vertical straight band of non-negligible thickness (> 5 pixels)

    can be drawn between any pair of adjacent strips, without touching either of them.

    * strips fall into straight vertical bands > 5 pixels wide

    * The "white" background of the page is lighter than the strips "almost everywhere"

    '''

    ## IDENTIFY ARRAY COLUMNS THAT ARE 90% "WHITE" 

    rows, cols = img.shape # number of rows and columns in the image.

    white_mask = img > 200    # numpy bool array, choose threshold that distinguishes white background from strip gray.

    column_white_count = white_mask.sum(axis = 0)    # list of counts of "white" pixels in each column of image 

    white_threshold = rows * 0.90    # 90% of of pixels should be "white" for a column to count as part of a "white" band between strips.

    white_cols = column_white_count > white_threshold # 1D boolean array



    ## GET BOUNDARIES OF THE WHITE BANDS BETWEEN STRIPS

    band_boundaries = [] # list of 2-tuples of start and finish col of band

    band_minimum = 5 # minimum width of band

    seeking_start = True # start by seeking start of first band, then alternate seeking start and end (i.e. left and right) of band

    for col in range(cols):

        found_boundary = False

        if seeking_start and white_cols[col]: # possible start of band (= rhs of strip)

            # check width sufficient to be band between strips

            found_boundary = True

            for i in range(1, band_minimum):

                if col+i < cols:

                    if not white_cols[col+i]:

                        found_boundary = False

                        break

        elif not seeking_start and not white_cols[col]: # possible end of band (= lhs of strip)

            # check width sufficient to be band between strips

            found_boundary = True

            for i in range(1, band_minimum):

                if col + i < cols: 

                    if white_cols[col+i]:

                        found_boundary = False

                        break

        if found_boundary:

            seeking_start = not seeking_start #alternate seeking lhs and rhs of bands

            band_boundaries.append(col)

    band_boundaries.append(cols - 1) # end of last band 

    print(band_boundaries)

    return band_boundaries



band_boundaries = get_white_bands(source_img)

number_of_strips = len(band_boundaries)//2 - 1

print("Number of strips = ", number_of_strips)



strips = []

for i in range(len(band_boundaries) - 3, 1, -2):

    strip = dict()

    strip['img_file'] = img_files[img_number]

    strip['right_bound'] = band_boundaries[i + 1]

    strip['right_outer_bound'] = (band_boundaries[i + 1] + band_boundaries[i + 2]) // 2

    strip['left_bound'] = band_boundaries[i]

    strip['left_outer_bound'] = (band_boundaries[i] + band_boundaries[i - 1]) // 2

    strips.append(strip)

    

def get_strip_img(img, strips, index, narrow=False):

    if narrow:

        return img[:, strips[index]['left_bound']:strips[index]['right_bound']]

    return img[:, strips[index]['left_outer_bound']:strips[index]['right_outer_bound']]



# show boundary positions on copy of source image

for boundary in band_boundaries:

    source_img_markup[:,boundary-1:boundary+1] = 0

    

plt.figure(figsize=(40, 40))

plt.imshow(source_img_markup, cmap='gray')

plt.show()
from skimage.morphology import binary_erosion, binary_dilation, disk, rectangle, square



strip_img = get_strip_img(source_img, strips, 4, True)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 30))    

                                            # ARGS: 1/ number of rows in the grid; 2/ number of cols in each row

                                            # RETURNS a figure obj and an (array of) axes obj(s)

ax[0].imshow(strip_img, cmap='gray')



ink_map = strip_img < 120



# erosion to remove noise

selem = square(3)

ink_erosion = binary_erosion(ink_map, selem)



# dilation to merge disjoint regions of same graph

selem = rectangle(25, ink_erosion.shape[1]*2)

ink_dilation = binary_dilation(ink_erosion, selem)



ax[1].imshow(ink_map)

ax[2].imshow(ink_erosion)

ax[3].imshow(ink_dilation)
fig, ax = plt.subplots(nrows=1, ncols=len(strips)*2, figsize=(40, 40))    

                                            # ARGS: 1/ number of rows in the grid; 2/ number of cols in each row

                                            # RETURNS a figure obj and an (array of) axes obj(s)

        

for strip_index in range(len(strips)):

    strip_img = get_strip_img(source_img, strips, strip_index, True)

    ink_map = strip_img < 120



    # erosion to remove noise

    selem = square(3)

    ink_erosion = binary_erosion(ink_map, selem)



    # dilation to merge disjoint regions of same graph

    selem = rectangle(25, ink_erosion.shape[1]*2)

    ink_dilation = binary_dilation(ink_erosion, selem)

    ax[strip_index*2 + 1].imshow(strip_img, cmap='gray')

    ax[strip_index*2].imshow(ink_dilation)



plt.show()
from skimage.measure import label, regionprops

import statistics



label_strip = label(ink_dilation)

boundaries = []

for region in regionprops(label_strip):

    minr, minc, maxr, maxc = region.bbox

    boundaries.append(minr)

    boundaries.append(maxr)



boundaries.sort()

print(boundaries)

sizes = []

spacings = []

prev_midpoint = None

for i in range(0, len(boundaries), 2):

    sizes.append(boundaries[i+1] - boundaries[i])

    midpoint = (boundaries[i+1] - boundaries[i]) // 2 + boundaries[i]

    if prev_midpoint:

        spacings.append(midpoint - prev_midpoint)

    prev_midpoint = midpoint

print(sizes)

print(spacings)



median_graph_size = int(statistics.median(sizes))

median_spacing = int(statistics.median(spacings))

print(median_graph_size)

print(median_spacing)



# break up merged graphs

boundaries_to_add = []

for i in range(len(sizes)):

    if sizes[i] > median_graph_size + median_spacing:

        for nth_new_boundary in range(sizes[i] // (median_graph_size + median_spacing)):

            new_boundary = boundaries[i*2] + nth_new_boundary * (median_graph_size + median_spacing)

            boundaries_to_add.append(int(new_boundary))

            boundaries_to_add.append(int(new_boundary))

print(boundaries_to_add)