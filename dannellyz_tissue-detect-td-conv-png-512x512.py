#All imports

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 5]

import numpy

import pandas as pd

import numpy as np

import cv2

from skimage import morphology

import openslide

import time



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

    #For timing

    time_stamps = {}

    time_stamps["start"] = time.time()

    

    # Get a downsample of the whole slide image (to fit in memory)

    downsampling_factor = min(

        wsi.level_downsamples, key=lambda x: abs(x - downsampling_factor))

    level = wsi.level_downsamples.index(downsampling_factor)



    slide = wsi.read_region((0, 0), level, wsi.level_dimensions[level])

    slide = np.array(slide)[:, :, :3]

    time_stamps["1"] = time.time()

    # Convert from RGB to HSV color space

    slide_hsv = cv2.cvtColor(slide, cv2.COLOR_BGR2HSV)

    time_stamps["2"] = time.time()

    # Compute optimal threshold values in each channel using Otsu algorithm

    _, saturation, _ = np.split(slide_hsv, 3, axis=2)



    mask = otsu_filter(saturation, gaussian_blur=True)

    time_stamps["3"] = time.time()

    # Make mask boolean

    mask = mask != 0



    mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)

    mask = morphology.remove_small_objects(mask, min_size=sensitivity)

    time_stamps["4"] = time.time()

    mask = mask.astype(np.uint8)

    mask_contours, tier = cv2.findContours(

        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    time_stamps["5"] = time.time()

    time_stamps = {key:(value-time_stamps["start"]) * 1000 for key,value in time_stamps.items()}

    return mask_contours, tier, slide, downsampling_factor, time_stamps



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



def tissue_cutout(tissue_slide, tissue_contours, slide):

    #https://stackoverflow.com/a/28759496

    crop_mask = np.zeros_like(tissue_slide) # Create mask where white is what we want, black otherwise

    cv2.drawContours(crop_mask, tissue_contours, -1, 255, -1) # Draw filled contour in mask

    tissue_only = np.zeros_like(slide) # Extract out the object and place into output image

    tissue_only[crop_mask == 255] = slide[crop_mask == 255]

    return tissue_only



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



def detect_and_crop(image_location:str, sensitivity:int=3000, 

                    downsample_rate:int=16, show_plots:str="simple"):

    

    #For timing

    time_stamps = {}

    time_stamps["start"] = time.time()

    

    #Open Slide

    wsi = openslide.open_slide(image_location)

    time_stamps["open"] = time.time()

    

    #Get returns from detect_tissue()

    (tissue_contours, tier, 

     downsampled_slide, 

     downsampling_factor,

     time_stamps_detect) = detect_tissue(wsi,

                                          sensitivity,downsample_rate)

    time_stamps["tissue_detect"] = time.time()

    

    #Get Tissue Only Slide

    base_slide_mask = np.zeros(downsampled_slide.shape[:2])

    tissue_slide = draw_tissue_polygons(base_slide_mask, tissue_contours,'line', 5)

    base_size = get_disk_size(downsampled_slide)

    tissue_only_slide = tissue_cutout(tissue_slide, tissue_contours, downsampled_slide)

    time_stamps["tissue_trim"] = time.time()

    #Get minimal bounding rectangle for all tissue contours

    if len(tissue_contours) == 0:

        img_id = image_location.split("/")[-1]

        print(f"No Tissue Contours - ID: {img_id}")

        return None, 1.0

    

    all_bounding_rect = cv2.minAreaRect(np.concatenate(tissue_contours))

    #Crop with getSubImage()

    smart_bounding_crop = getSubImage(all_bounding_rect,tissue_only_slide)

    time_stamps["crop"] = time.time()

    

    #Crop empty space

    #Remove by row

    row_not_blank =  [row.all() for row in ~np.all(smart_bounding_crop == [255,0,0],

                                                   axis=1)]

    space_cut = smart_bounding_crop[row_not_blank,:]

    #Remove by column

    col_not_blank =  [col.all() for col in ~np.all(smart_bounding_crop == [255,0,0],

                                                   axis=0)]

    space_cut = space_cut[:,col_not_blank]

    time_stamps["cut"] = time.time()

    

    #Get size change

    start_size = get_disk_size(downsampled_slide)

    final_size = get_disk_size(space_cut)

    pct_change = final_size / start_size

    

    if show_plots == "simple":

        print(f"Percent Reduced from Base Slide to Final: {(1- pct_change)*100:.2f}")

        plt.imshow(space_cut)

        plt.show() 

    elif show_plots == "verbose":

        #Set-up dictionary for plotting

        verbose_plots = {}

        #Add Base Slide to verbose print

        verbose_plots[f"Base Slide\n{get_disk_size(downsampled_slide):.2f}MB"] = downsampled_slide

        #Add Tissue Only to verbose print

        verbose_plots[f"Tissue Detect\nNo Change"] = tissue_slide

        #Add Bounding Boxes to verbose print

        verbose_plots[f"Bounding Boxes\n{get_disk_size(smart_bounding_crop):.2f}MB"] = smart_bounding_crop

        #Add Space Cut Boxes to verbose print

        verbose_plots[f"Space Cut\n{get_disk_size(space_cut):.2f}MB"] = space_cut

        print(f"Percent Reduced from Base Slide to Final: {(1- pct_change)*100:.2f}")

        plt = plot_figures(verbose_plots, 1, len(verbose_plots))

        plt.show()

    elif show_plots == "none":

        pass

    else:

        pass

    time_stamps["all"] = time.time()

    time_stamps = {key:(value-time_stamps["start"]) * 1000 for key,value in time_stamps.items()}

    return space_cut, (1-pct_change), time_stamps



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



#Set up example slide

slide_dir = "../input/prostate-cancer-grade-assessment/train_images/"

annotation_dir = "../input/prostate-cancer-grade-assessment/train_label_masks/"

example_id = "0032bfa835ce0f43a92ae0bbab6871cb"

example_slide = f"{slide_dir}{example_id}.tiff"

numpy_result, pct_change, time_stamps = detect_and_crop(image_location=example_slide, downsample_rate=16, show_plots="verbose")

print(time_stamps)
small_img = cv2.resize(numpy_result, (512, 512))

plt.imshow(small_img)

plt.show() 
import os

import skimage.io

from multiprocess import Pool

from statistics import mean

from tqdm.notebook import tqdm

save_dir = "/kaggle/train_images/"

os.makedirs(save_dir, exist_ok=True)

train_data_df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")



## 5 Image sample for example *******************

image_ids = list(train_data_df.image_id.sample(5))

## Uncomment for call code *******************

#image_ids = list(train_data_df.image_id)

def make_images(image_id):

    load_path = slide_dir + image_id + '.tiff'

    save_path = save_dir + image_id + '.png'

    

    biopsy, pct_change, time_stamps = detect_and_crop(load_path, downsample_rate=16, show_plots="none")

    if biopsy is None: return 0

    img = cv2.resize(biopsy, (512, 512))

    cv2.imwrite(save_path, img)

    return pct_change

        



with Pool(processes=4) as pool:

    avg_pct_reduced = list(

        tqdm(pool.imap(make_images, image_ids), total = len(image_ids))

    )



print(f"The averge size reduced reduced is {mean(avg_pct_reduced):.2%}")
!tar -czf train_images.tar.gz ../train_images/*.png