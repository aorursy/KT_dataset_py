##### PACKAGES

import os

from pathlib import Path

import pydicom as dcm  # great library for working with dicom

from collections import defaultdict

import matplotlib.pyplot as plt
data_path = Path('../input/rsna-str-pulmonary-embolism-detection/train/759a5963508b')

dcm_paths = list(data_path.glob('**/*.dcm'))



print(f"First 10 dicom paths: \n\n{dcm_paths[:10]}\n\n... out of {len(dcm_paths)} total dcms")
"""Utiliy functions for displaying dcms clearly"""





def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == dcm.multival.MultiValue: return int(x[0])

    else: return int(x)

    

def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]



def metadata_window(img, print_ranges=False):

    # Get data from dcm

    window_center, window_width, intercept, slope = get_windowing(img)

    img = img.pixel_array

    

    # Window based on dcm metadata

    img = img * slope + intercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    if print_ranges:

        print(f'Window Level: {window_center}, Window Width:{window_width}, Range:[{img_min} {img_max}]')

    img[img < img_min] = img_min

    img[img > img_max] = img_max

    

    # Normalize

    img = (img - img_min) / (img_max - img_min)

    return img



def show_image_slices(dcms, title="dcms"):

    fig, axes = plt.subplots(1, len(dcms), figsize=(25,5))

    fig.suptitle(title, fontsize=16)

    for i, dcm in enumerate(dcms):

        axes[i].imshow(metadata_window(dcm), cmap="gray")

        
""" 

Note: slices are most reliably sorted by patient position. For axial slices this will be z index

as there is no gaurantee that the sequence numbers are different in a study so we can generate 

a unique key for each sequence (series number + position-axis[0] + position-axis[1]).

The first two positions (x, y) in ImagePositionPatient usually vary between Axial sequences 

"""



# dictionary of keys to series

dcm_series = defaultdict(list)

for dcm_path in dcm_paths:

    d = dcm.read_file(dcm_path) 

    # key = series number + first axis, most likely unique across axial series.

    dcm_series[f"Series #:{d.SeriesNumber}, x:{d.ImagePositionPatient[0]}, y:{d.ImagePositionPatient[1]}"].append(d)

    

    

print("printing series by series-keys with lengths...")

for key, series in dcm_series.items():

    print(f"series key [{key}] has:  {len(series)} dcms")

for key, series in dcm_series.items():

    

    show_image_slices(series[:30:5], title=key+" UNSORTED")

    
for key, series in dcm_series.items():

    

    show_image_slices(sorted(series[:30:5], key=lambda dcm:dcm.ImagePositionPatient[2]), title=key+" SORTED")   # ImagePositionPatient[2] is the Z axis

    