import cv2

import numpy as np

from matplotlib import pyplot as plt

from collections import defaultdict



def show_image(img, label='image'):

    plt.imshow(img)

#     cv2.waitKey(0)





def smooth(img, filter_type):

    if filter_type == "mean":

        return cv2.blur(img, (5,5))

    if filter_type == "gaussian":

        return cv2.GaussianBlur(img, (5,5), 0)

    if filter_type == "median":

        return cv2.medianBlur(img, 5)

    if filter_type == "bilateral":

        return cv2.bilateralFilter(img, 9, 75, 75)

    return bilateral_filter

    

    # return the mode pixel of the image

def get_mode(img, xdim, ydim):

    # split into color channels

    [B,G,R] = cv2.split(img)

    blue = B.astype(float)

    green = G.astype(float)

    red = R.astype(float)

      

    # count the number of times each triple shows up

    d = defaultdict(int)

    for i in range(xdim):

        for j in range(ydim):

            d[(B[i,j], G[i,j], R[i,j])] += 1



    # return the triple which shows up most often

    maxval = 0

    returnval = (0,0,0)

    for k,v in d.items():

        if v > maxval:

            returnval = k

            maxval = v

    return returnval

    

def plot_histogram(img):

  color = ('b','g','r')

  for i,col in enumerate(color):

      histr = cv2.calcHist([img],[i],None,[256],[0,256])

      plt.plot(histr,color = col)

      plt.xlim([0,256])

  plt.show()

  

# import cv22 as cv2

# import numpy as np

# import utils



def detect(img, xdim, ydim):

	# convert to grayscale

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	show_image(gray, 'gray')

	

	# threshold to convert to binary image

	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	show_image(thresh, 'threshold')



	# erode image to isolate the sure foreground

	kernel = np.ones((3,3),np.uint8)

	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations=3)

	show_image(opening, 'opening')



	# get the median pixel value (should be background)

	mode = get_mode(img, xdim, ydim)

	

	# replace the foreground (trees) with the median pixel

	for i in range(xdim):

		for j in range(ydim):

    		# if it's white in the eroded image, then it's vegetation

			if opening[i,j] == 255:

    			# set to black

				img[i,j] = mode

				

	show_image(img, 'color-overlay')

	return img

	

	# sure background area

	# sure_bg = cv2.dilate(opening2,kernel,iterations=3)

	# show_image(sure_bg, 'sure_bg')



	# Finding sure foreground area

	# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

	# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)



	# Finding unknown region

	# sure_fg = np.uint8(sure_fg)

	# unknown = cv2.subtract(sure_bg,sure_fg)



# if __name__ == "__main__":

# 	main()



# import cv22 as cv2

# #import pymeanshift as ms

# import numpy as np

# from matplotlib import pyplot as plt

# import utils



def vegetationMask(im, xdim, ydim):

  # compute color invariant

  [B,G,R] = cv2.split(im)

  red = R.astype(float)

  blue = B.astype(float)

  green = G.astype(float)



  colInvarIm = np.zeros(shape=(xdim, ydim))



  # iterate over the image

  for i in range(xdim):

    for j in range(ydim):

      # if there are no blue or green at thix pixel, turn it black

      if (green[i,j] + blue[i,j]) < np.finfo(float).eps:

        colInvarIm[i,j] = 2

      else:

        if blue[i,j] > 130 and blue[i,j] < 150:

          im[i,j] = blue[i,j] #(4./np.pi)*np.arctan((blue[i,j] - green[i,j])/(green[i,j] + blue[i,j]))

        else:

          im[i,j] = 2



  show_image(im, 'blue threshold')

  # normalize to [0,255]

  colInvarIm += abs(colInvarIm.min())

  colInvarIm *= 255.0/colInvarIm.max()

  colInvarIm = colInvarIm.astype('uint8')



  # threshold to detect vegetation

  thresh, vegetation = cv2.threshold(colInvarIm, 0, 255, cv2.THRESH_OTSU)



  cv2.imshow('color invariant image', colInvarIm)

  cv2.waitKey(0)

  cv2.imshow('vegetation', vegetation)

  cv2.waitKey(0)

  #cv2.destroyAllWindows()



  cinvar_fname = fname[:-4] + '-col-invar.png'

  #cv2.imwrite(cinvar_fname, colInvarIm)

  mask_fname = fname[:-4] + '-veg-mask.png'

  #cv2.imwrite(mask_fname, vegetation)



  return vegetation



def mask(img, xdim, ydim):



  plot_histogram(img)



  [B,G,R] = cv2.split(img)

  blue = B.astype(float)

  green = G.astype(float)

  red = R.astype(float)



  meanR = np.mean(red)

  stdR = np.std(red)

  print(meanR + 1.6 * stdR)

  meanB = np.mean(blue)

  stdB = np.std(blue)

  print(meanB + 1.1 * stdB)



  mode_pixel = get_mode(img, xdim, ydim)



  # separate into roads and houses

  for i in range(xdim):

    for j in range(ydim):

      # road: red value is at least 2 std above the mean

      if red[i,j] > meanR + 1.6 * stdR: # red[i,j] > 180

        img[i,j] = mode_pixel

      # houses: blue value is at least 1 std above the mean

      if blue[i,j] > meanB + 1.1 * stdB: # 182: #and blue[i,j] <= 238:

        img[i,j] = (0,0,0)



  show_image(img, 'mask')



  return img



  #seg, labels, num_regions = \

    # ms.segment(im, spatial_radius=6, range_radius=4.5, min_density=50)

  #seg2 = np.copy(seg)



  #veg = vegetationMask(im, xdim, ydim)



  #hists = np.bincount(np.reshape(labels, xdim*ydim))

  #hmean = np.mean(hists)

  #hstd = np.std(hists)



  #for i in xrange(num_regions):

  #  if hists[i] < 15 or hists[i] > 2*hstd + hmean:

  #    seg2[labels == i, :] = 0





  

  #cv2.imshow('segmented1', seg)

  #cv2.waitKey(0)



  #fout = fname[:-4] + '-seg.png'

  #cv2.imwrite(fout, seg)



# if __name__ == '__main__':

#   main()



# import cv2

# import numpy as np

# import utils



# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# WINDOW_NAME = "win"



def detect(segmented, original, xdim, ydim):



  # morphological opening and closing

  kernel = np.ones((3,3), np.uint8)

  img = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)

  img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)



  show_image(img, 'open-close')



  imgcopy = img.copy()

  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)



  num_buildings = 0



  for i in range(255):

    # threshold the grayscale image at that value

    binary = np.zeros((xdim, ydim), np.uint8)

    ret, binary = cv2.threshold(gray, dst=binary, thresh=i, maxval=255, type=cv2.THRESH_OTSU)

    #binary[gray == i] = 255

    # utils.show_image(binary, 'binary')



    # find contours, fit to polygon, and determine if rectangular

    contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)



    for c in contours:

      poly = cv2.approxPolyDP(np.array(c), 0.07*cv2.arcLength(c,True), True)

      carea = cv2.contourArea(c)

      polyarea = cv2.contourArea(poly)

      hull = cv2.convexHull(c)

      hullarea = cv2.contourArea(hull)



      # bounding box

      rect = cv2.minAreaRect(c)

      box = cv2.boxPoints(rect)

      box = np.int0(box)

#       cv2.drawContours(frame,[box], 0, (0, 0, 255), 2)

      if polyarea > 30 and carea > 30:

        cv2.drawContours(img, [c], 0, (0,0,255), 1)

      if len(poly) < 6 and carea > 100: #and carea > 5: #\

          #and abs(polyarea/carea - 1) < 0.25:

        num_buildings += 1

        cv2.drawContours(imgcopy, [poly], 0, (0,0,255), 1)

        cv2.drawContours(original, [poly], 0, (0,0,255), 1)



  # show images

  show_image(img, 'all bounding boxes')

  show_image(imgcopy, 'with some filtering')

  show_image(original, 'onto original')

  print(num_buildings)

  return original
# import cv2

# import utils

# import matplotlib.pyplot as plt

# import detectvegetation

# import segmentcolor

# import detectpolygon



#def main():

fname = '../input/chibombo1.png'

original = cv2.imread(fname)

show_image(original,'ori')

# show_image(original, 'original')

img = smooth(original, 'mean')

show_image(img, 'bilateral')



# get image dimensions

# xdim, ydim, nchannels = img.shape

# print(xdim,ydim,nchannels)



# # veg_to_background = detect(img, xdim, ydim)



# segmented = mask(img, xdim, ydim)



# detect = detect(segmented, img, xdim, ydim)

# plt.imshow(detect)