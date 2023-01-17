import pandas as pd

import rasterio as rio

import os



s5p_file = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20190501T161114_20190507T174400.tif'

gldas_file = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20180702_1500.tif'

gfs_file = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2019011212.tif'



def preview_meta_data(file_name):

    with rio.open(file_name) as img_filename:

        print('Metadata for: ',file_name)

        print('Bounding Box:',img_filename.bounds)

        print('Resolution:',img_filename.res)

        print('Tags:',img_filename.tags())

        print('More Tags:',img_filename.tags(ns='IMAGE_STRUCTURE'))

        print('Number of Channels =',img_filename.count,'\n')



def return_bands(file_name):

    # adapted from 

    # https://www.kaggle.com/gpoulain/ds4g-eda-bands-understanding-and-gee

    image = rio.open(file_name)

    for i in image.indexes:

        desc = image.descriptions[i-1]

        print(f'{i}: {desc}')
preview_meta_data(s5p_file)

preview_meta_data(gldas_file)

preview_meta_data(gfs_file)
print('S5P: ','\n')

return_bands(s5p_file)

print('\nGLDAS: ','\n')

return_bands(gldas_file)

print('\nGFS: ','\n')

return_bands(gfs_file)