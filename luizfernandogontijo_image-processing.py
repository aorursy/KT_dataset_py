from matplotlib import pyplot as plt



def show_image(image, title='Image', cmap_type='gray'): 

  plt.imshow(image, cmap=cmap_type)

  plt.title(title)

  plt.axis('off')

  plt.show()
# show a image of a rocket

from skimage import data 

rocket_image = data.rocket()

show_image(rocket_image, 'A rocket')
from skimage import color

rocket_image_gray = color.rgb2gray(rocket_image)

show_image(rocket_image_gray)
import numpy as np



flipped_rocket_image = np.flipud(rocket_image)

show_image(flipped_rocket_image, 'Flipped rocket')
red = rocket_image[:,:,0]

green = rocket_image[:,:,1]

blue = rocket_image[:,:,2]



show_image(red, 'Red channel')

show_image(green, 'Green channel')

show_image(blue, 'Blue channel')
rocket_image.shape # the third argument in parenthesis indicates that there are three channels in that image
rocket_image.size # the size of a image is the total count of pixels: 427x640
red = rocket_image[:, :, 0] # using the red channel of the rocket image.



plt.hist(red.ravel(), bins=256) # plot its histogram with 256 bins, the number of possible values of a pixel.

plt.title('Red Histogram')

plt.show
cameraman_image = data.camera()

show_image(cameraman_image, 'Original cameraman')



thresh = 120 # set a random thresh value



binary_high = cameraman_image > thresh

binary_low = cameraman_image <= thresh



show_image(binary_high, 'Tresholded high values')

show_image(binary_low, 'Tresholded low values')
text_image = data.page()



show_image(text_image, 'Text image')
from skimage.filters import try_all_threshold



fig, ax = try_all_threshold(text_image, verbose=False)
from skimage.filters import threshold_otsu



thresh = threshold_otsu(text_image)



text_binary_otsu = text_image > thresh



show_image(text_binary_otsu, 'Otsu algorithm')
from skimage.filters import threshold_local



block_size = 35 # define the size of the region to apply the binarization



local_thresh = threshold_local(text_image, block_size, offset=10) # apply the function



binary_local = text_image > local_thresh



show_image(binary_local)
def plot_comparison(original, filtered, title_filtered):

  fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6), sharex=True, sharey=True)

  ax1.imshow(original, cmap=plt.cm.gray) 

  ax1.set_title('original') 

  ax1.axis('off')

  ax2.imshow(filtered, cmap=plt.cm.gray) 

  ax2.set_title(title_filtered) 

  ax2.axis('off')
from skimage.filters import sobel



coins_image = data.coins() # use the coins image as an example



edge_coins_image = sobel(coins_image) # apply the filter



plot_comparison(coins_image, edge_coins_image, 'Edge coins')
from skimage.filters import gaussian



cat_image = data.chelsea()



smmooth_cat_image = gaussian(cat_image, multichannel=True) # you have to specify the multichannel



plot_comparison(cat_image, smmooth_cat_image, 'Smooth cat')
from skimage import exposure



moon_image = data.moon()



equalized_image_moon = exposure.equalize_hist(moon_image)



plot_comparison(moon_image, equalized_image_moon, 'Histogram equalization')
from skimage import exposure



adapthits_image_moon = exposure.equalize_adapthist(moon_image)



plot_comparison(moon_image, adapthits_image_moon, 'Adaptive Histogram equalization')
from skimage.transform import rotate



coffe_image = data.coffee()



rotate_coffe_image = rotate(coffe_image, -90) # rotate clockwise



show_image(rotate_coffe_image)
from skimage.transform import rescale



rescale_coffe_image = rescale(coffe_image, 1/2, anti_aliasing=True, multichannel=True) # rescaling by 1/2 of the original image



show_image(coffe_image)

print(coffe_image.shape)



show_image(rescale_coffe_image)

print(rescale_coffe_image.shape)



from skimage.transform import rescale



rescale_coffe_image = rescale(coffe_image, 5, anti_aliasing=True, multichannel=True) # rescaling by 1/2 of the original image



show_image(coffe_image)

print(coffe_image.shape) # the shape of the original image



show_image(rescale_coffe_image)

print(rescale_coffe_image.shape) # the shape of the rescaled image
from skimage.transform import resize



height = 600 

width = 1000



resized_coffe_image = resize(coffe_image, (height, width), anti_aliasing=True)



show_image(coffe_image)

print(coffe_image.shape) # the shape of the original image



show_image(resized_coffe_image)

print(resized_coffe_image.shape) # the shape of the resized image
from skimage import morphology



horse_image = data.horse()



eroded_horse_image = morphology.binary_erosion(horse_image)



plot_comparison(horse_image, eroded_horse_image, 'Eroded image')
dilated_horse_image = morphology.binary_dilation(horse_image)



plot_comparison(horse_image, dilated_horse_image, 'Dilated image')
from skimage.util import random_noise



cat_image = data.chelsea()



noisy_cat_image = random_noise(cat_image)



plot_comparison(cat_image, noisy_cat_image, 'Noisy image')
from skimage.restoration import denoise_tv_chambolle



denoised_cat_image = denoise_tv_chambolle(noisy_cat_image, weight=0.1,multichannel=True)



plot_comparison(noisy_cat_image, denoised_cat_image, 'Denoised image')
from skimage.restoration import denoise_bilateral



denoised_cat_image = denoise_bilateral(noisy_cat_image, multichannel=True)



plot_comparison(noisy_cat_image, denoised_cat_image, 'Denoised image')
coins_image = data.coins()



thresh = 90 # set a random thresh value



segmented_coins_image = coins_image > thresh



plot_comparison(coins_image, segmented_coins_image, 'Segmented image')
from skimage.segmentation import slic # import te slic function



from skimage.color import label2rgb # import te label2rgb function



segments_400 = slic(coffe_image, n_segments = 400) # segmentation with 400 regions



segmented_image_400 = label2rgb(segments_400, coffe_image, kind='avg')



segments_40 = slic(coffe_image, n_segments = 40) # segmentation with 400 regions



segmented_image_40 = label2rgb(segments_40, coffe_image, kind='avg')



show_image(segmented_image_400, '400 regions of segmentation')



show_image(segmented_image_40, '40 regions of segmentation')
# A function to show the contour of the image 

def show_image_contour(image, contours):

    plt.figure()

    for n, contour in enumerate(contours):

        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)

    plt.imshow(image, interpolation='nearest', cmap='gray_r')

    plt.title('Contours')

    plt.axis('off')

    plt.show()
from skimage import measure



horse_image = data.horse()



contours_horse_image = measure.find_contours(horse_image, 0.8)



show_image_contour(horse_image, contours_horse_image)
from PIL import Image

from numpy import asarray



dices_image = Image.open('/content/drive/My Drive/Notebooks/dices.png')



dices_image = asarray(dices_image)



show_image(dices_image)
from skimage import filters



dices_image_gray = color.rgb2gray(dices_image)



thresh = filters.threshold_otsu(dices_image_gray)



binary_dices_image = dices_image_gray > thresh



contours = measure.find_contours(binary_dices_image, 0.8)



show_image_contour(binary_dices_image, contours)
from skimage.feature import canny



coins_image = data.coins()



coins_image_gray = color.rgb2gray(coins_image)



canny_coins_image = canny(coins_image_gray)



show_image(canny_coins_image)
bulding_image = Image.open('/content/drive/My Drive/Notebooks/corners_building_top.jpg')



bulding_image = asarray(bulding_image)



show_image(bulding_image)
from skimage.feature import corner_harris, corner_peaks



bulding_image_gray = color.rgb2gray(bulding_image)



corner_bulding_image_gray = corner_harris(bulding_image_gray) 



show_image(corner_bulding_image_gray) 



coords = corner_peaks(corner_bulding_image_gray, min_distance=5) # find de coordinates of the corners



print("A total of", len(coords), "corners were detected.")
def show_image_with_corners(image, coords, title="Corners detected"): 

  plt.imshow(image, interpolation='nearest', cmap='gray') 

  plt.title(title)

  plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=15) 

  plt.axis('off')

  plt.show()
show_image_with_corners(corner_bulding_image_gray, coords)
from skimage.feature import Cascade



trained_file = data.lbp_frontal_face_cascade_filename() # load the training file



detector = Cascade(trained_file) # initialize the detector cascade.
astronaut_image = data.astronaut()



# Apply detector on the image

detected = detector.detect_multi_scale(img=astrounaut_image, 

                                       scale_factor=1.2,

                                        step_ratio=1,

                                        min_size=(10, 10), max_size=(200, 200))



print(detected)
from matplotlib import patches



def show_detected_face(result, detected, title="Face image"):

    plt.figure()

    plt.imshow(result)

    img_desc = plt.gca()

    plt.set_cmap('gray')

    plt.title(title)

    plt.axis('off')



    for patch in detected:

        

        img_desc.add_patch(

            patches.Rectangle(

                (patch['c'], patch['r']),

                patch['width'],

                patch['height'],

                fill=False,

                color='r',

                linewidth=2)

        )

    plt.show()
show_detected_face(astronaut_image, detected)
from PIL import Image

from numpy import asarray



flamengo_image = Image.open('/content/drive/My Drive/Notebooks/flamengo.jpg')



flamengo_image = asarray(flamengo_image)



detected = detector.detect_multi_scale(img=flamengo_image, 

                                       scale_factor=1.2,

                                        step_ratio=1,

                                        min_size=(10, 10), max_size=(200, 200))



show_detected_face(flamengo_image, detected)

image = data.astronaut()



def getFaceRectangle(d):

    ''' Extracts the face from the image using the coordinates of the detected image '''

    # X and Y starting points of the face rectangle

    x, y  = d['r'], d['c']

    

    # The width and height of the face rectangle

    width, height = d['r'] + d['width'],  d['c'] + d['height']

    

    # Extract the detected face

    face = image[ x:width, y:height]

    return face
detected = detector.detect_multi_scale(img=image, 

                                       scale_factor=1.2,

                                        step_ratio=1,

                                        min_size=(10, 10), max_size=(200, 200))


def mergeBlurryFace(original, gaussian_image):

     # X and Y starting points of the face rectangle

    x, y  = d['r'], d['c']

    # The width and height of the face rectangle

    width, height = d['r'] + d['width'],  d['c'] + d['height']

  

    original[x:width, y:height] =  gaussian_image

    return original

from skimage.filters import gaussian



#image.setflags(write=1)



for d in detected: 

  face = getFaceRectangle(d)

  

  gaussian_face = gaussian(face, multichannel=True, sigma = 1) # Apply gaussian filter to extracted face

  

  resulting_image = mergeBlurryFace(image, gaussian_face) # Merge this blurry face to our final image and show it



show_image(resulting_image, "Blurred faces")

 