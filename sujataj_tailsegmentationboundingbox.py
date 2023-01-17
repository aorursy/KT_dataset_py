# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from skimage.util import montage as montage2d

from skimage.color import rgb2hsv, gray2rgb, rgb2gray, label2rgb

from skimage.feature import greycomatrix, greycoprops

from skimage.io import imread

from skimage import img_as_ubyte

from skimage.filters import threshold_otsu

from skimage.measure import regionprops

import matplotlib.patches as mpatches

from skimage.morphology import label

from skimage.segmentation import slic

from collections import namedtuple

from skimage import measure

import regex as re

import os

from glob import glob



import warnings  

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



n_samples = 50



base_img_dir = os.path.join('..', 'input')

all_tails = glob(os.path.join(base_img_dir, '*', 'train', 'train', '*.jpg'))



# Any results you write to the current directory are saved as output.
def create_binary_image_with_threshold(image):

    threshold = threshold_otsu(image)

    return (image < threshold).astype(np.float32)



def add_box(labelled_image, color='red'):

        maxArea = 0

        for region in regionprops(labelled_image):

            if region.area > maxArea:

                maxArea = region.area

                maxRegion = region

            

        minr, minc, maxr, maxc = maxRegion.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,

                          fill=False, edgecolor=color, linewidth=2)

        return rect

    

def image_name_parser(c_path):

    file_name_pattern = re.compile(r'\.\.\/input\/whale-categorization-playground\/train/train\/(?P<name>.+\.jpg)')

    match = file_name_pattern.match(c_path)

    if match is not None:

        return match.group('name')



def flatten_superpixel_image(color_image, superpixel_image):

    flat_image = color_image.copy()

    for s_idx in np.unique(superpixel_image.ravel()):

            flat_image[superpixel_image == s_idx] = np.mean(

            flat_image[superpixel_image == s_idx])

    return rgb2gray(flat_image.copy())



def create_hsv_dict(hsv_image):

    hsv_dict = {}

    hsv_dict['h'] = hsv_image[:, :, 0]

    hsv_dict['s'] = hsv_image[:, :, 1]

    hsv_dict['v'] = hsv_image[:, :, 2]

    return hsv_dict



def create_binary_with_hsv_threshold(hsv_dict):

    try:

        value_threshold = threshold_otsu(hsv_dict['v'])

        hue_threshold = threshold_otsu(hsv_dict['h'])

        binary_image = 1.0*((hsv_dict['v'] < value_threshold) | (hsv_dict['h'] < hue_threshold))

        return binary_image

    except Exception as e:

        print(e)

        return None

def bounding_rectangle(list):

    x0, y0 = list[0]

    x1, y1 = x0, y0

    for x,y in list[1:]:

        x0 = min(x0, x)

        y0 = min(y0, y)

        x1 = max(x1, x)

        y1 = max(y1, y)

    return x0,y0,x1,y1



def bb_intersection_over_union(rectA, rectB):

    if rectB is None or rectA is None:

        return 0

    boxA = rectA.get_bbox().get_points().flatten()

    boxB = rectB.get_bbox().get_points().flatten()

    # determine the (x, y)-coordinates of the intersection rectangle

    xA = max(boxA[0], boxB[0])

    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])

    yB = min(boxA[3], boxB[3])

 

    # compute the area of intersection rectangle

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

 

    # compute the area of both the prediction and ground-truth

    # rectangles

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)

    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

 

    # compute the intersection over union by taking the intersection

    # area and dividing it by the sum of prediction + ground-truth

    # areas - the interesection area

    iou = interArea / float(boxAArea + boxBArea - interArea)

    

    # return the intersection over union value

    return iou



def check_array_for_all_zeros(hsv_dict):

    return not np.any(hsv_dict['v']) or not np.any(hsv_dict['h']) or not np.any(hsv_dict['s'])



def crop(image, rect):

    return image[rect.get_y():rect.get_y()+rect.get_height() , rect.get_x():rect.get_x()+rect.get_width(), :]



def montage_nd(in_img):

    if len(in_img.shape) > 3:

        return montage2d(np.stack([montage_nd(x_slice) for x_slice in in_img], 0))

    elif len(in_img.shape) == 3:

        return montage2d(in_img)

    else:

        warn('Input less than 3d image, returning original', RuntimeWarning)

        return in_img

    

def calc_coomatrix(in_img):

    return greycomatrix(image=in_img,

                        distances=dist_list,

                        angles=angle_list,

                        levels=4)





def coo_tensor_to_df(x): return pd.DataFrame(

    np.stack([x.ravel()]+[c_vec.ravel() for c_vec in np.meshgrid(range(x.shape[0]),

                                                                 range(

                                                                     x.shape[1]),

                                                                 dist_list,

                                                                 angle_list,

                                                                 indexing='xy')], -1),

    columns=['E', 'i', 'j', 'd', 'theta'])



def calculate_glcm_properties(image):

    grayco_prop_list = ['contrast', 'dissimilarity',

                    'homogeneity', 'energy',

                    'correlation', 'ASM']

    

    out_row = {}

    glcm = greycomatrix(image, [5], [0], 256, symmetric=True, normed=True)

    for c_prop in grayco_prop_list:

        out_row[c_prop] = greycoprops(glcm, c_prop)[0, 0]

    out_row['bw_ratio'] = np.mean(image)

    return out_row

    

Detection = namedtuple("Detection", ["image_name","image", "ml", "color", "hsv", "iou_color", "iou_hsv"])
all_tails_dict = {}

ml_box_dict = {}

for c_path in all_tails:

    image_name = image_name_parser(c_path)

    all_tails_dict[image_name] = c_path



with open('../input/humpback-whale-identification-fluke-location/cropping.txt', 'rt') as f: data = f.read().split('\n')[:-1]



data = [line.split(',') for line in data]

data = [(p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) for p,*coord in data]

for filename, coordinates in data:

    x0,y0,x1,y1 = bounding_rectangle(coordinates)

    box = mpatches.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, color='green')

    ml_box_dict[filename] = box
detection_list = []

out_df_list = []





for image_name in np.random.choice(list(ml_box_dict.keys()), size=n_samples):

#for image_name in list(ml_box_dict.keys()):

    c_path = all_tails_dict[image_name]

    image = imread(c_path)

    if len(np.shape(image)) == 3:

        pass

    else:

        image = gray2rgb(image)

    # bounding box analysis

    working_img = label(create_binary_image_with_threshold(flatten_superpixel_image(image, slic(image, n_segments=250))))

    color_box = add_box(working_img)

    hsv_dict = create_hsv_dict(rgb2hsv(image)[:, :, :])



    if check_array_for_all_zeros(hsv_dict):

        hsv_box = None

    else: 

        hsv_box = add_box(label(create_binary_with_hsv_threshold(hsv_dict)), 'orange')



    detect = Detection(image_name, 

                                image, 

                                ml_box_dict[image_name], 

                                color_box, 

                                hsv_box, 

                                bb_intersection_over_union(ml_box_dict[image_name], color_box), 

                                bb_intersection_over_union(ml_box_dict[image_name], hsv_box))







#image analaysis

    image_ml = img_as_ubyte(rgb2gray(crop(detect.image, detect.ml)))

    image_color = img_as_ubyte(rgb2gray(crop(detect.image, detect.color)))





    

    ml_out_row = calculate_glcm_properties(image_ml)

    ml_out_row['type'] = 'ML'

    ml_out_row['iou'] = 1

    out_df_list += [ml_out_row]





    color_out_row = calculate_glcm_properties(image_color)

    color_out_row['type'] = 'Color'

    color_out_row['iou'] = detect.iou_color

    out_df_list += [color_out_row]







out_df = pd.DataFrame(out_df_list)

sns.pairplot(out_df,

             x_vars=['contrast', 'dissimilarity',

                    'homogeneity', 'energy',

                    'correlation', 'ASM', 'bw_ratio'],

             y_vars=['iou'],

             hue='type',

             kind="reg")
detection_list = []

out_df_list = []

image_list = []

for image_name in np.random.choice(list(ml_box_dict.keys()), size=5):

#for image_name in list(ml_box_dict.keys()):

    c_path = all_tails_dict[image_name]

    image = imread(c_path)

    if len(np.shape(image)) == 3:

        pass

    else:

        image = gray2rgb(image)

    image_list.append(image)

    x_list=[]

    y_list=[]

    # bounding box analysis

    for segments in np.arange(50,1000,50):

        working_img = label(create_binary_image_with_threshold(flatten_superpixel_image(image, slic(image, n_segments=segments))))

        color_box = add_box(working_img)

        hsv_dict = create_hsv_dict(rgb2hsv(image)[:, :, :])



        if check_array_for_all_zeros(hsv_dict):

            hsv_box = None

        else: 

            hsv_box = add_box(label(create_binary_with_hsv_threshold(hsv_dict)), 'orange')



        detect = Detection(image_name, 

                                    image, 

                                    ml_box_dict[image_name], 

                                    color_box, 

                                    hsv_box, 

                                    bb_intersection_over_union(ml_box_dict[image_name], color_box), 

                                    bb_intersection_over_union(ml_box_dict[image_name], hsv_box))

        

        image_color = img_as_ubyte(rgb2gray(crop(detect.image, detect.color)))

        #print(detect.image_name, detect.iou_color, np.mean(image_color))

        

        x_list.append(segments)

        y_list.append(detect.iou_color)

    

    plt.plot(x_list, y_list)
for image in image_list: 

    fig, ax = plt.subplots()

    ax.imshow(image, cmap='bone')
    #if detect.hsv is not None:

    #    image_hsv = img_as_ubyte(rgb2gray(crop(detect.image, detect.hsv)))

    #    hsv_out_row = calculate_glcm_properties(image_hsv)

    #    hsv_out_row['type'] = 'HSV'

    #    hsv_out_row['iou'] = detect.iou_hsv

    #    out_df_list += [hsv_out_row]



    

    #fig, ax = plt.subplots()

    #ax.imshow(image_hsv, cmap='bone')