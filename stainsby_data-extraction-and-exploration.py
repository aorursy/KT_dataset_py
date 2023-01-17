import pandas

import numpy as np

import matplotlib.pyplot as plt

from scipy import misc



INPUT_IMAGE_SIZE = 300 # pixels; we will resize and embed all input images into a square of this size
train_metadata = pandas.read_csv('../input/train.csv')

test_metadata = pandas.read_csv('../input/test.csv')

train_metadata
train_ids = train_metadata['id']

train_species = train_metadata['species']



species = list(set(train_species))

species.sort()

num_species = len(species)



print('Read %i training samples of %i species.' % (len(train_ids), num_species))



test_ids = test_metadata['id']



print('Read %i testing samples.' % len(test_ids))



# some useful maps



species_name_2_species_id_map = {}

species_name_2_sample_ids_map = {}

sample_id_2_species_id_map = {}

sample_id_2_species_name_map = {}



for id, name in zip(range(len(species)), species):

    species_name_2_species_id_map[name] = id



for i in range(len(train_ids)):

    sample_id = train_ids[i]

    species_name = train_species[i]

    species_id = species_name_2_species_id_map[species_name]

    if not species_name_2_sample_ids_map.get(species_name, None):

        species_name_2_sample_ids_map[species_name] = []

    species_name_2_sample_ids_map[species_name].append(sample_id)

    sample_id_2_species_id_map[sample_id] = species_id

    sample_id_2_species_name_map[sample_id] = species_name
species_counts = [len(species_name_2_sample_ids_map[name]) for name in species]

print('Distinct counts per species: %s' % set(np.unique(species_counts)))
sample_ids = list(set(train_ids).union(set(test_ids)))

sample_id_to_image_map = {}





# Creates an image in a fixed squared shape with normalised pixel values.

def standardise_image(im, size = INPUT_IMAGE_SIZE):

    

    # Resize the original image if necessary

    

    major_axis = np.max(im.shape)

    if major_axis > size:

        resize_factor = size/major_axis

        im = misc.imresize(im, resize_factor, interp='bilinear')

    im_height, im_width = im.shape

    offset_y = (size - im_height)//2

    offset_x = (size - im_width)//2

    im_sqr = np.zeros((size, size), dtype=np.float)

    im_sqr[offset_y:(offset_y + im_height), offset_x:(offset_x + im_width)] = im

    im = im_sqr

    

    # Normalise - this is essential the same as tensoflow's tf.image.per_image_standardization

    

    mean = np.mean(im)

    stddev = np.std(im)

    adjusted_stddev = np.max([stddev, 1.0/np.sqrt(im.size)])

    im = (im - mean)/adjusted_stddev

    

    # Add a channel dimension of size 1

    

    im = im.reshape(size, size, 1)

    

    return im





def read_images():

    np.random.shuffle(sample_ids)

    print('Number of images to read:', len(sample_ids))

    sample_id_to_image_map = {}

    is_first = True

    image_read_count = 0

    for id in sample_ids:

        im0 = misc.imread('../input/images/' + str(id) + '.jpg', mode='L')

        image_read_count += 1

        # ensure properly thresholded

        im = 1*(im0 >= 128)

        if is_first:

            is_first = False

            # these input images are not properly thesholded - this will show it:

            dirty_pixels = 1*(im0 > 0)*(im0 < 255)

            im_dirty = im0*dirty_pixels

            print('Showing a random example:')

            print('Shape of image:', im.shape, '; id =', id)

            f, ((dx1, dx2), (ax1, ax2)) = plt.subplots(2, 2, figsize=(9, 9))

            dx1.imshow(dirty_pixels, cmap='gray')

            dx1.set_title('locations of values not zero or 255')

            dx2.hist(im_dirty.flatten(), bins=64, edgecolor='red', facecolor='red')

            dx2.set_title('value histogram')

            ax1.imshow(im, cmap='gray')

            ax1.set_title('thresholded image')

            _, _, bars = ax2.hist(im.flatten(), bins=2)

            bars[0].set_facecolor('black')

            bars[-1].set_facecolor('white')

            ax2.set_title('value histogram')

            plt.show()

            print('Reading in remaining images ...')



        sample_id_to_image_map[id] = standardise_image(im)



    print('Finished reading images - read %d images.' % image_read_count)

    assert image_read_count == len(sample_ids), 'read the wrong number of images'

    

    return sample_id_to_image_map





sample_id_to_image_map = read_images()
# Just a test

def test_original_images():

    print('An input image chosen at random:')

    im = sample_id_to_image_map[np.random.choice(sample_ids)]

    plt.figure(figsize=(300/90, 300/90), dpi=90)

    # we need to squeeze out the channel dimanesion that we added

    plt.imshow(np.squeeze(im), cmap='gray')

    plt.show()

    print('Some more input images:')

    rows = 6

    for k in range(rows):

        _, axs = plt.subplots(1, 9, figsize=(9, 2))

        for i, ax in zip(range(len(axs)), axs):

            im = sample_id_to_image_map[sample_ids[i + k*rows]]

            ax.imshow(np.squeeze(im), cmap='gray')

            ax.get_xaxis().set_ticks([])

            ax.get_yaxis().set_ticks([])

        plt.show()





test_original_images() # just a test