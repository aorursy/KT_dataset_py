import numpy as np



import scipy as sp

import scipy.ndimage as ndi

from scipy.signal import argrelextrema



import pandas as pd



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





def downsample_contour(coords, bins=512):

    """splits the array to ~equal bins, and returns one point per bin"""

    edges = np.linspace(0, coords.shape[0], 

                       num=bins).astype(int)

    for b in range(bins-1):

        yield [coords[edges[b]:edges[b+1],0].mean(), 

               coords[edges[b]:edges[b+1],1].mean()]





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





def near0_ix(timeseries_1d, radius=5):

    """finds near-zero values in time-series"""

    return np.where(timeseries_1d < radius)[0]





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





# wrapper function for feature extraction tasks

def get_std_contours(img):

    """from image to standard-length countour pairs"""

    

    # shape in boolean n:m format

    blur = extract_shape(img) 

    

    # contours in [[x,y], ...] format

    blade = np.array(list(downsample_contour(get_contour(img))))

    shape = np.array(list(downsample_contour(get_contour(blur))))

    

    # flagging blade points that fall inside the shape contour

    # notice that we are loosing subpixel information here

    blade_y, blade_x = coords_to_cols(blade)

    blade_inv_ix = blur[blade_x.astype(int), blade_y.astype(int)]

    

    # img and shape centers

    shape_cy, shape_cx = get_center(blur)

    blade_cy, blade_cx = get_center(img)

    

    # img distance, shape distance (for time series plotting)

    blade_dist = dist_line_line(shape, blade)

    shape_dist = dist_line_point(shape, [shape_cx, shape_cy])



    # fixing false + signs in the blade time series

    blade_dist[0, blade_inv_ix] = blade_dist[0, blade_inv_ix] * -1

    

    return {'shape_img': blur,

            'shape_contour': shape, 

            'shape_center': (shape_cx, shape_cy),

            'shape_series': [shape_dist, range(len(shape_dist))],

            'blade_img': img,

            'blade_contour': blade,

            'blade_center': (blade_cx, blade_cy),

            'blade_series': blade_dist,

            'inversion_ix': blade_inv_ix}

    
title, img = list(get_imgs([968]))[0]

features = get_std_contours(img)



plt.subplot(121)

plt.plot(*coords_to_cols(features['shape_contour']))

plt.plot(*coords_to_cols(features['blade_contour']))

#plt.axis('equal')



plt.subplot(122)

plt.plot(*features['shape_series'])

plt.plot(*features['blade_series'])

plt.show()
# determining eigenvalues and eigenvectors for the leaves

# and drawing 2SD ellipse around its center as a 3rd descriptor



from matplotlib.patches import Ellipse



standard_deviations = 2

x_imgsize, y_imgsize = features['shape_img'].shape



# generating rnd coords

x_rnd = np.random.randint(x_imgsize, size=2048)

y_rnd = np.random.randint(y_imgsize, size=2048)

rnd_coords = np.array([y_rnd, x_rnd])



# checking rnd coords against shape, keep only the ones inside

shape_mask = features['shape_img'][x_rnd, y_rnd]

sampled_coords = rnd_coords[0, shape_mask], rnd_coords[1, shape_mask]



# this is actually a PCA, visualized as an ellipse

covariance_matrix = np.cov(sampled_coords)

eigenvalues, eigenvectors = pd.np.linalg.eigh(covariance_matrix)

order = eigenvalues.argsort()[::-1]

eigenvectors = eigenvectors[:,order]

theta = pd.np.rad2deg(pd.np.arctan2(*eigenvectors[0]) % (2 * pd.np.pi))

width, height = 2 * standard_deviations * pd.np.sqrt(eigenvalues)
# visualization

ellipse = Ellipse(xy=features['shape_center'],

                  width=width, height=height, angle=theta, 

                  fc='k', color='none', alpha=.2)



ax = plt.subplot(111)



ax.add_artist(ellipse)

ax.set_title(title)

ax.plot(*coords_to_cols(features['shape_contour']))

ax.plot(*coords_to_cols(features['blade_contour']))

ax.axis('equal')

plt.show()