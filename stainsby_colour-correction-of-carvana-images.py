import glob

import numpy as np

from scipy import ndimage

from matplotlib import pyplot as plt



PROJECT_PATH = '..'

INPUT_PATH = PROJECT_PATH + '/input'

TRAIN_IMAGE_PATH = INPUT_PATH + '/train'

TEST_IMAGE_PATH = INPUT_PATH + '/test'
# Use the pixel-wise mean of top strip of an image to calculate the background colour.

# A strip 32 pixels high seems to work well.



def extract_mean_backgound_colour(image, top_strip_width = 32):

    top_strip = image[:top_strip_width, : , :]

    return np.mean(top_strip, axis=(0,1))
# Now do this for all images (or a limited number of images by setting max_images to a value

# less than the total number of images).



def find_mean_background_colour(max_images = 1000*1000):

    

    image_paths = glob.glob(TRAIN_IMAGE_PATH + '/*.jpg') + glob.glob(TEST_IMAGE_PATH + '/*.jpg')

    image_paths = image_paths[:max_images]



    print('Finding images ...')

    image_count = 0

    num_images = len(image_paths)

    mean_bg_colour = np.asarray([0.0, 0.0, 0.0])

    print('\nStarting ...\n')

    for image_path in image_paths:

        image = ndimage.imread(image_path, mode = 'RGB')

        mean_bg_colour = mean_bg_colour + extract_mean_backgound_colour(image)

        image_count += 1

        if image_count % 1000 == 0:

            print('  .. completed', image_count, 'of', num_images, 'images: mean bg',

                    mean_bg_colour/image_count, ' ..')

    mean_bg_colour = mean_bg_colour/image_count

    print('\nDone.')

    print('Mean background colour:', mean_bg_colour)

    return mean_bg_colour
# THIS TAKES A LONG TIME SO WE WON'T DO IT HERE.

# MEAN_BACKGROUND_COLOUR = find_mean_background_colour()



# Instead we'll use the answer calculated previously:

MEAN_BACKGROUND_COLOUR = np.asarray((241.84525443, 240.75213576, 238.65245818))
# A numpy implementation of rudimentary colour correction.

# I actually use a Tensorflow implementation of this in training (appended to this notebook),

# which doesn't use clipping since there is all sort of normalisation later.



def colour_correct_image(image):

    mean_bg_colour = extract_mean_backgound_colour(image)

    colour_correction_factor = mean_bg_colour/MEAN_BACKGROUND_COLOUR

    corrected_image = np.round(image/colour_correction_factor)

    corrected_image = np.clip(corrected_image, 0.0, 255.0)

    return corrected_image.astype(np.uint8)
# Somes tests/examples:



def test_colour_correction():

    image_file_names = [

        '0d53224da2b7_05.jpg',

        '0d3adbbc9a8b_14.jpg',

        '1a17a1bd648b_15.jpg',

        '2ea62c1beee7_15.jpg',

        '11fcda0a9e1c_04.jpg'

    ]

    image_paths = [TRAIN_IMAGE_PATH + '/' + file_name for file_name in image_file_names]

    images = [ndimage.imread(path, mode = 'RGB') for path in image_paths]

    count = 0

    for image in images:

        count += 1; print('\n--------\nImage #' + str(count))

        plt.figure(figsize=(12, 10))

        plt.subplot(221); plt.title('Original Image'); plt.imshow(image)

        plt.subplot(223); plt.title('Original RBG'); plt.hist(image.flatten(), bins=100); 

        corrected_image = colour_correct_image(image)

        plt.subplot(222); plt.title('Corrected Image'); plt.imshow(corrected_image)

        plt.subplot(224); plt.title('Corrected RBG'); plt.hist(corrected_image.flatten(), bins=100)

        plt.show()

        





test_colour_correction()
# My Tensorflow implemenation for those interested (NOTE: I did the inverse of the

# corrected_rgb_image that I used in the numpy version, not that it makes any difference):



def tf_colour_correction(rgb_image, top_strip_width = 32):

    global_mean_bg_colour = tf.constant(MEAN_BACKGROUND_COLOUR/255.0, dtype = tf.float32)

    mean_bg_colour = tf.reduce_mean(rgb_image[:, :top_strip_width, : , :], axis = (1, 2), keep_dims=True)

    colour_correction_factor = global_mean_bg_colour/mean_bg_colour

    corrected_rgb_image = colour_correction_factor*rgb_image

    return corrected_rgb_image