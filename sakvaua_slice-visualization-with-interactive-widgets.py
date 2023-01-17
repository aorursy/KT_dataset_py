%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import matplotlib.pyplot as plt



from ipywidgets import interact, interactive, fixed

import ipywidgets as widgets



from scipy.ndimage.morphology import binary_dilation

from skimage.morphology import ball

from skimage import measure
# Borrowed directly from https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    

    slice_thickness = slices[0].SliceLocation - slices[1].SliceLocation

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

      

    intercept = scans[0].RescaleIntercept

    image += int(intercept)

    

    return np.array(image)



def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))

    spacing = (np.array(list(spacing)))

    if spacing[0]<0:

        spacing[0]=-spacing[0]

        #print ('Image',image.shape)

        image=image[::-1,:,:]

        #print ('Image',image.shape)

    #print('Spacing ', spacing,new_spacing)



    resize_factor = spacing / new_spacing

    #print ('Resize factor',resize_factor)

    

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    #print ('New shape',new_shape)

    

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    #print(image.shape, real_resize_factor)

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    

    return image, new_spacing



def largest_label_volume(im, bg=-1):

    label_values = list(np.unique(im))

    if bg in label_values:

        label_values.remove(bg)

    biggest = label_values[np.argmax([np.sum(im == l) for l in label_values])]

    return biggest



def segment_lung_mask(image, fill_lung_structures=True):

    

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -300, dtype=np.int8)+1

    labels = measure.label(binary_image)

    

    # Pick the pixel in the very corner to determine which label is air.

    #   Improvement: Pick multiple background labels from around the patient

    #   More resistant to "trays" on which the patient lays cutting the air 

    #   around the person in half

    background_label = labels[0,0,0]

    

    #Fill the air around the person

    binary_image[background_label == labels] = 2

    

    # Method of filling the lung structures (that is superior to something like 

    # morphological closing)

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)



            binary_image[i][labeling != l_max] = 1



        

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

    

    # Remove other air pockets insided body

    labels = measure.label(binary_image, background=0)

    l = largest_label_volume(labels, bg=0)

    binary_image[labels != l] = 0

 

    return binary_image
path= 'C:/DSB2017/data/stage1/'

path= '../input/sample_images/'



def plot_slice(sl,pat):

    f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(9,3))

    ax1.imshow(slices[sl], cmap=plt.cm.bone)

    ax2.imshow(mask[sl])

    ax3.imshow(slices[sl]*mask[sl], cmap=plt.cm.bone)

    plt.axis('off')

    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.show()



def load_patient(patient):

    global slices, mask

    

    scan = load_scan(path+patient)

    slices = get_pixels_hu(scan)

    mask=segment_lung_mask(slices, fill_lung_structures=True)

    mask=binary_dilation(mask,structure=selem,iterations=5)





slices=np.zeros((1,1,1))

mask=np.zeros((1,1,1))

selem=ball(radius=2)#structure element to dilate the mask



patients=os.listdir(path)

load_patient(patients[1])

wslider=widgets.IntSlider(min=0,max=slices.shape[0]-1,step=1,value=100)

wlist=widgets.Select(options=patients)



def update_slider(*args):

    load_patient(args[0]['new'])

    wslider.max = slices.shape[0]-1

    if wslider.value>wslider.max:

        wslider.value=wslider.max

        

wlist.observe(update_slider, 'value')



interact(plot_slice, sl=wslider,pat=wlist);