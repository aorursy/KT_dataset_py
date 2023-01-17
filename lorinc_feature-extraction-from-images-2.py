import numpy as np



import scipy as sp

import scipy.ndimage as ndi

from scipy.signal import argrelextrema



from skimage import measure

from sklearn import metrics



import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches



%matplotlib inline

from pylab import rcParams

rcParams['figure.figsize'] = (6, 6)
# ----------------------------------------------------- I/O ---



def read_img(img_no):

    """reads image from disk"""

    return mpimg.imread('../input/images/' + str(img_no) + '.jpg')





def get_imgs(num):

    """convenience function, yields random sample from leaves"""

    if type(num) == int:

        imgs = range(1, 1584)

        num = np.random.choice(imgs, size=num, replace=False)

        

    for img_no in num:

        yield img_no, preprocess(read_img(img_no))





# ----------------------------------------------------- preprocessing ---



def threshold(img, threshold=250):

    """splits img to 0 and 255 values at threshold"""

    return ((img > threshold) * 255).astype(img.dtype)





def portrait(img):

    """makes all leaves stand straight"""

    y, x = np.shape(img)

    return img.transpose() if x > y else img

    



def resample(img, size):

    """resamples img to size without distorsion"""

    ratio = size / max(np.shape(img))

    return sp.misc.imresize(img, ratio, mode='L', interp='nearest')



    

def fill(img, size=500, tolerance=0.95):

    """extends the image if it is signifficantly smaller than size"""

    y, x = np.shape(img)



    if x <= size * tolerance:

        pad = np.zeros((y, int((size - x) / 2)), dtype=int)

        img = np.concatenate((pad, img, pad), axis=1)



    if y <= size * tolerance:

        pad = np.zeros((int((size - y) / 2), x), dtype=int)

        img = np.concatenate((pad, img, pad), axis=0) 

    

    return img





# ----------------------------------------------------- postprocessing ---



def standardize(arr1d):

    """move mean to zero, 1st SD to -1/+1"""

    return (arr1d - arr1d.mean()) / arr1d.std()





def coords_to_cols(coords):

    """from x,y pairs to feature columns"""

    return coords[::,1], coords[::,0]





def get_contour(img):

    """returns the coords of the longest contour"""

    return max(measure.find_contours(img, .8), key=len)





def get_center(img):

    """so that I do not have to remember the function ;)"""

    return ndi.measurements.center_of_mass(img)





# ----------------------------------------------------- feature engineering ---



def extract_shape(img):

    """

    Expects prepared image, returns leaf shape in img format.

    The strength of smoothing had to be dynamically set

    in order to get consistent results for different sizes.

    """

    size = int(np.count_nonzero(img)/1000)

    brush = int(5 * size/size**.75)

    return ndi.gaussian_filter(img, sigma=brush, mode='nearest') > 200





# TODO: optimize - do not search the whole array for lmin, just the near0 parts

def near0_lmin_ix(timeseries):

    """finds near-zero local *flat* minima in time-series"""

    lmin = argrelextrema(timeseries[0], np.less_equal, order=3)

    near0 = np.where(timeseries[0] < 3)

    return np.intersect1d(lmin, near0)  # returns indices





def extend_index(ix, radius=4):  # 3 is good, 4 is safe

    """extends near0_lmin_ix results by radius"""

    result = []

    for ix in ix:

        result += list(range(ix-radius, ix+radius))

    return np.unique(result)





def dist_line_line(src_arr, tgt_arr):

    """

    returns 2 tgt_arr length arrays, 

    1st is distances, 2nd is src_arr indices

    """

    return np.array(sp.spatial.cKDTree(src_arr).query(tgt_arr))





def dist_line_point(src_arr, point):

    """returns 1d array with distances from point"""

    point1d = [[point[0], point[1]]] * len(src_arr)

    return metrics.pairwise.paired_distances(src_arr, point1d)





def index_diff(kdt_output_1):

    """

    Shows pairwise distance between all n and n+1 elements.

    Useful to see, how the dist_line_line maps the two lines.

    """

    return np.diff(kdt_output_1)





# ----------------------------------------------------- wrapping functions ---



# wrapper function for all preprocessing tasks    

def preprocess(img, do_portrait=True, do_resample=500, 

               do_fill=True, do_threshold=250):

    """ prepares image for processing"""

    if do_portrait:

        img = portrait(img)

    if do_resample:

        img = resample(img, size=do_resample)

    if do_fill:

        img = fill(img, size=do_resample)

    if do_threshold:

        img = threshold(img, threshold=do_threshold)

        

    return img
# exploring solution before building it as function



# img, shape

title, img = list(get_imgs([709]))[0]

blur = extract_shape(img)



# img contour, shape contour  

blade = get_contour(img)

shape = get_contour(blur)



# img distance, shape distance  

shape_y, shape_x = get_center(blur)

blade_dist = dist_line_line(shape, blade)

shape_dist = dist_line_point(shape, [shape_x, shape_y])



# finding minima near 0 on the edge

blade_poi_ix = extend_index(near0_lmin_ix(blade_dist))
# the points to be checked

blade_x, blade_y = coords_to_cols(blade)

shape_x, shape_y = coords_to_cols(shape)



section_x = blade_x[blade_poi_ix]

section_y = blade_y[blade_poi_ix]



plt.plot(blade_x, blade_y, linewidth=.3, c='r')

plt.plot(shape_x, shape_y, linewidth=.3, c='g')

plt.scatter(section_x, section_y, 

            marker='x', linewidth=.3, s=10, alpha=.5)

plt.show()