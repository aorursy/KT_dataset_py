#All imports

%matplotlib inline

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 5]

import numpy

import pandas as pd

import numpy as np

import cv2

from skimage import morphology

import openslide



#Setup code

slide_dir = "../input/prostate-cancer-grade-assessment/train_images/"

annotation_dir = "../input/prostate-cancer-grade-assessment/train_label_masks/"

train_data_df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")

sample_id_list = list(train_data_df["image_id"].sample(5))

sample_slides = [f"{slide_dir}{slide_id}.tiff" for slide_id in sample_id_list]

sample_annotations = [f"{annotation_dir}{slide_id}_mask.tiff" for slide_id in sample_id_list]



def get_disk_size(numpy_image):

    """ Returns size in MB of numpy array on disk."""

    return (numpy_image.size * numpy_image.itemsize) / 1000000



def plot_figures(figures, nrows = 1, ncols=1):

    #https://stackoverflow.com/a/11172032

    """Plot a dictionary of figures.



    Parameters

    ----------

    figures : <title, figure> dictionary

    ncols : number of columns of subplots wanted in the display

    nrows : number of rows of subplots wanted in the figure

    """



    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)

    for ind,title in enumerate(figures):

        axeslist.ravel()[ind].imshow(figures[title], aspect='auto')

        axeslist.ravel()[ind].set_title(title)

    plt.tight_layout()

    return plt



print("^ All imports and setup code in above hidden code block. ^")
def downsample(wsi, downsampling_factor=16):

    #Select the min downsampling factor 

    #Between the input value and the available values from the slide

    downsampling_factor = min(wsi.level_downsamples, 

                              key=lambda x: abs(x - downsampling_factor))

    

    #Set the level of the slide by the downsampling

    level = wsi.level_downsamples.index(downsampling_factor)

    

    #Read and convert to numpy array

    slide = wsi.read_region((0, 0), level, wsi.level_dimensions[level])

    numpy_slide = np.array(slide)[:, :, :3]

    

    return numpy_slide



#Set up example slide

example_id = "037504061b9fba71ef6e24c48c6df44d"

example_slide = f"{slide_dir}{example_id}.tiff"



#Open slide as wsi

wsi = openslide.open_slide(example_slide)



def display_downsample(factor):

    display_slide = downsample(wsi, factor)

    return (f"Factor:\n{factor}\nSize (MB):\n{get_disk_size(display_slide)}",display_slide)



potential_lvls = wsi.level_downsamples

downsample_results = [display_downsample(factor) for factor in potential_lvls]

downsample_dict = {k:v for k,v in downsample_results}

plt = plot_figures(downsample_dict, 1, len(downsample_dict))

plt.show()
def otsu_filter(channel, gaussian_blur=True):

    """Otsu filter."""

    if gaussian_blur:

        channel = cv2.GaussianBlur(channel, (5, 5), 0)

    channel = channel.reshape((channel.shape[0], channel.shape[1]))



    return cv2.threshold(

        channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



def detect_tissue(wsi, sensitivity = 3000, downsampling_factor=64):

    

    """

    Find RoIs containing tissue in WSI.

    Generate mask locating tissue in an WSI. Inspired by method used by

    Wang et al. [1]_.

    .. [1] Dayong Wang, Aditya Khosla, Rishab Gargeya, Humayun Irshad, Andrew

    H. Beck, "Deep Learning for Identifying Metastatic Breast Cancer",

    arXiv:1606.05718

    

    Parameters

    ----------

    wsi: OpenSlide/AnnotatedOpenSlide class instance

        The whole-slide image (WSI) to detect tissue in.

    downsampling_factor: int

        The desired factor to downsample the image by, since full WSIs will

        not fit in memory. The image's closest level downsample is found

        and used.

    sensitivity: int

        The desired sensitivty of the model to detect tissue. The baseline is set

        at 5000 and should be adjusted down to capture more potential issue and

        adjusted up to be more agressive with trimming the slide.

        

    Returns

    -------

    -Binary mask as numpy 2D array, 

    -RGB slide image (in the used downsampling level, in case the user is visualizing output examples),

    -Downsampling factor.

    """

    

    # Get a downsample of the whole slide image (to fit in memory)

    downsampling_factor = min(

        wsi.level_downsamples, key=lambda x: abs(x - downsampling_factor))

    level = wsi.level_downsamples.index(downsampling_factor)



    slide = wsi.read_region((0, 0), level, wsi.level_dimensions[level])

    slide = np.array(slide)[:, :, :3]



    # Convert from RGB to HSV color space

    slide_hsv = cv2.cvtColor(slide, cv2.COLOR_BGR2HSV)



    # Compute optimal threshold values in each channel using Otsu algorithm

    _, saturation, _ = np.split(slide_hsv, 3, axis=2)



    mask = otsu_filter(saturation, gaussian_blur=True)



    # Make mask boolean

    mask = mask != 0



    mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)

    mask = morphology.remove_small_objects(mask, min_size=sensitivity)



    mask = mask.astype(np.uint8)

    mask_contours, tier = cv2.findContours(

        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    return mask_contours, tier, slide, downsampling_factor



def draw_tissue_polygons(mask, polygons, polygon_type,

                              line_thickness=None):

        """

        Plot as numpy array detected tissue.

        Modeled WSIPRE github package

        

        Parameters

        ----------

        mask: numpy array 

            This is the original image represented as 0's for a starting canvas

        polygons: numpy array 

            These are the identified tissue regions

        polygon_type: str ("line" | "area")

            The desired display type for the tissue regions

        polygon_type: int

            If the polygon_type=="line" then this parameter sets thickness



        Returns

        -------

        Nunmpy array of tissue mask plotted

        """

        

        tissue_color = 1



        for poly in polygons:

            if polygon_type == 'line':

                mask = cv2.polylines(

                    mask, [poly], True, tissue_color, line_thickness)

            elif polygon_type == 'area':

                if line_thickness is not None:

                    warnings.warn('"line_thickness" is only used if ' +

                                  '"polygon_type" is "line".')



                mask = cv2.fillPoly(mask, [poly], tissue_color)

            else:

                raise ValueError(

                    'Accepted "polygon_type" values are "line" or "area".')



        return mask

    

#Base Example

tissue_contours, tier, downsampled_slide, downsampling_factor = detect_tissue(wsi, 3000,64)

base_slide_mask = np.zeros(downsampled_slide.shape[:2])

tissue_slide = draw_tissue_polygons(base_slide_mask, tissue_contours,'line', 2)

base_size = get_disk_size(downsampled_slide)

plt.imshow(tissue_slide)

plt.show()

print("^ Code hidden above for Tissue Detection Algorithm. ^")
#Sensitivity Tests

def sensitivity_test(sensitivity):

    """Take in a given sensitivity and return tissue_slide"""

    tissue_contours, tier, downsampled_slide, downsampling_factor = detect_tissue(wsi, sensitivity,64)

    base_slide_mask = np.zeros(downsampled_slide.shape[:2])

    tissue_slide = draw_tissue_polygons(base_slide_mask, tissue_contours,'line', 2)

    return (f"Sensitivity:\n{sensitivity}",tissue_slide)



to_test = [i for i in range(0,7500,1500)]

sensitvity_results = [sensitivity_test(sensitivity) for sensitivity in to_test]

sensitvity_dict = {k:v for k,v in sensitvity_results}

sensitvity_dict["Basic Slide"] = downsampled_slide

plt = plot_figures(sensitvity_dict, 1, len(sensitvity_dict))

plt.show()
def tissue_cutout(tissue_slide, tissue_contours, slide):

    #https://stackoverflow.com/a/28759496

    crop_mask = np.zeros_like(tissue_slide) # Create mask where white is what we want, black otherwise

    cv2.drawContours(crop_mask, tissue_contours, -1, 255, -1) # Draw filled contour in mask

    tissue_only = np.zeros_like(slide) # Extract out the object and place into output image

    tissue_only[crop_mask == 255] = slide[crop_mask == 255]

    return tissue_only



tissue_only_slide = tissue_cutout(tissue_slide, tissue_contours, downsampled_slide)

plt.imshow(tissue_only_slide)

plt.show()



current_size = get_disk_size(tissue_only_slide)

current_pct = current_size / base_size

print(f"Slide Size on Disk: {current_size:.2f}MB")

print(f"% of original image: {current_pct*100:.2f}%")
min_rect_bounding = tissue_only_slide.copy()

for c in tissue_contours:

    rect = cv2.minAreaRect(c)

    box = cv2.boxPoints(rect)

    box = np.int0(box)

    cv2.drawContours(min_rect_bounding,[box],0,(0,255,255),4)

plt.imshow(min_rect_bounding)

plt.show()
simple_rect_bound = min_rect_bounding.copy()

boxes = []

for c in tissue_contours:

    (x, y, w, h) = cv2.boundingRect(c)

    boxes.append([x,y, x+w,y+h])



boxes = np.asarray(boxes)

left = np.min(boxes[:,0])

top = np.min(boxes[:,1])

right = np.max(boxes[:,2])

bottom = np.max(boxes[:,3])



cv2.rectangle(simple_rect_bound, (left,top), (right,bottom), (255, 0, 0), 4)



plt.imshow(simple_rect_bound)

plt.show()
smart_bounding_boxes = min_rect_bounding.copy()

all_bounding_rect = cv2.minAreaRect(np.concatenate(tissue_contours))

all_bounding_box = cv2.boxPoints(all_bounding_rect)

all_bounding_box = np.int0(all_bounding_box)

cv2.drawContours(smart_bounding_boxes,[all_bounding_box],0,(255,0,0),4)

plt.imshow(smart_bounding_boxes)

plt.show()
simple_crop = smart_bounding_boxes.copy()

crop_mask = np.zeros_like(simple_crop)

(y, x) = np.where(tissue_slide == 1)

(topy, topx) = (np.min(y), np.min(x))

(bottomy, bottomx) = (np.max(y), np.max(x))

simple_crop = simple_crop[topy:bottomy+1, topx:bottomx+1]

plt.imshow(simple_crop)

plt.show()



current_size = get_disk_size(simple_crop)

current_pct = current_size / base_size

print(f"Slide Size on Disk: {current_size:.2f}MB")

print(f"% of original image: {current_pct*100:.2f}%")
def getSubImage(rect, src_img):

    width = int(rect[1][0])

    height = int(rect[1][1])

    box = cv2.boxPoints(rect)



    src_pts = box.astype("float32")

    dst_pts = np.array([[0, height-1],

                        [0, 0],

                        [width-1, 0],

                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(src_img, M, (width, height))

    return warped



smart_bounding_crop = smart_bounding_boxes.copy()

smart_bounding_crop = getSubImage(all_bounding_rect,smart_bounding_crop)

plt.imshow(smart_bounding_crop)

plt.show()



current_size = get_disk_size(smart_bounding_crop)

current_pct = current_size / base_size

print(f"Slide Size on Disk: {current_size:.2f}MB")

print(f"% of original image: {current_pct*100:.2f}%")
row_not_blank =  [row.all() for row in ~np.all(smart_bounding_crop == [255,   0,   0],axis=1)]

col_not_blank =  [col.all() for col in ~np.all(smart_bounding_crop == [255,   0,   0],axis=0)]

bounded_cut = smart_bounding_crop[row_not_blank,:]

bounded_cut = bounded_cut[:,col_not_blank]

plt.imshow(bounded_cut)

plt.show()



current_size = get_disk_size(bounded_cut)

current_pct = current_size / base_size

print(f"Slide Size on Disk: {current_size:.2f}MB")

print(f"% of original image: {current_pct*100:.2f}%")
def detect_and_crop(image_location:str, sensitivity:int=3000, 

                    downsample_rate:int=16, show_plots:str="simple"):

    

    #Set-up dictionary for plotting

    verbose_plots = {}

    

    #Open Slide

    wsi = openslide.open_slide(image_location)

    

    #Get returns from detect_tissue()

    (tissue_contours, tier, 

     downsampled_slide, 

     downsampling_factor) = detect_tissue(wsi,

                                          sensitivity,downsample_rate)

    #Add Base Slide to verbose print

    verbose_plots[f"Base Slide\n{get_disk_size(downsampled_slide):.2f}MB"] = downsampled_slide

    

    #Get Tissue Only Slide

    base_slide_mask = np.zeros(downsampled_slide.shape[:2])

    tissue_slide = draw_tissue_polygons(base_slide_mask, tissue_contours,'line', 5)

    base_size = get_disk_size(downsampled_slide)

    tissue_only_slide = tissue_cutout(tissue_slide, tissue_contours, downsampled_slide)

    #Add Tissue Only to verbose print

    verbose_plots[f"Tissue Detect\nNo Change"] = tissue_slide

    

    #Get minimal bounding rectangle for all tissue contours

    if len(tissue_contours) == 0:

        img_id = image_location.split("/")[-1]

        print(f"No Tissue Contours - ID: {img_id}")

        return None, 1.0

    

    all_bounding_rect = cv2.minAreaRect(np.concatenate(tissue_contours))

    #Crop with getSubImage()

    smart_bounding_crop = getSubImage(all_bounding_rect,tissue_only_slide)

    #Add Bounding Boxes to verbose print

    verbose_plots[f"Bounding Boxes\n{get_disk_size(smart_bounding_crop):.2f}MB"] = smart_bounding_crop



    #Crop empty space

    #Remove by row

    row_not_blank =  [row.all() for row in ~np.all(smart_bounding_crop == [255,0,0],

                                                   axis=1)]

    space_cut = smart_bounding_crop[row_not_blank,:]

    #Remove by column

    col_not_blank =  [col.all() for col in ~np.all(smart_bounding_crop == [255,0,0],

                                                   axis=0)]

    space_cut = space_cut[:,col_not_blank]

    #Add Space Cut Boxes to verbose print

    verbose_plots[f"Space Cut\n{get_disk_size(space_cut):.2f}MB"] = space_cut

    

    #Get size change

    start_size = get_disk_size(downsampled_slide)

    final_size = get_disk_size(space_cut)

    pct_change = final_size / start_size

    

    if show_plots == "simple":

        print(f"Percent Reduced from Base Slide to Final: {(1- pct_change)*100:.2f}")

        plt.imshow(space_cut)

        plt.show() 

    elif show_plots == "verbose":

        print(f"Percent Reduced from Base Slide to Final: {(1- pct_change)*100:.2f}")

        plt = plot_figures(verbose_plots, 1, len(verbose_plots))

        plt.show()

    elif show_plots == "none":

        pass

    else:

        pass

    return space_cut, (1-pct_change)

    

numpy_result, pct_change = detect_and_crop(image_location=example_slide, show_plots="verbose")
for sample in sample_slides:

    detect_and_crop(image_location=sample, show_plots="verbose")
from statistics import mean 

from multiprocessing import Pool

from tqdm.notebook import tqdm

import gc

baseline_pct = .05

baseline_count = int(baseline_pct*len(train_data_df))

baseline_slide_ids = list(train_data_df["image_id"].sample(baseline_count))

baseline_slide_locs = [f"{slide_dir}{slide_id}.tiff" for slide_id in baseline_slide_ids]

#Nested in funciton to not take up more memory and allow mulitprocessing

def baseline_check(image_id):

    numpy_result, pct_change = detect_and_crop(image_location=image_id, show_plots="none")

    del numpy_result

    gc.collect()

    return pct_change



with Pool(processes=4) as pool:

    avg_pct_reduced = list(

        tqdm(pool.imap(baseline_check, baseline_slide_locs), total = len(baseline_slide_locs))

    )



print(f"The averge size reduced reduced from a {baseline_pct:.0%} sample of slides is {mean(avg_pct_reduced):.2%}")