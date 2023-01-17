import os



import rasterio as rio

from rasterio.plot import show, show_hist

import matplotlib.pyplot as plt

from matplotlib import animation



from datetime import datetime



import numpy as np

import pandas as pd
def load_sp5():

    # Get the data filenames

    no2_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/'

    no2_files = [no2_path + f for f in os.listdir(no2_path)]

    

    data = []

    

    print(f'Reading {len(no2_files)} files')

    for f in no2_files:

        raster = rio.open(f)

        data += [{

            'tif': raster,

            'filename': f.split('/')[-1],

            'measurement': 'no2',

            **raster.meta

        }]

        raster.close()

        

    # Get dates

    for d in data:

        d.update({'datetime': datetime.strptime(d['filename'][:23], 's5p_no2_%Y%m%dT%H%M%S')})



    for d in data:

        d['date'] = d['datetime'].date()

        d['hour'] = datetime.strftime(d['datetime'], '%H')

        d['weekday'] = d['datetime'].weekday()  # Mon = 0



    return data



data = load_sp5()
data[0]
df = pd.DataFrame(data)
# Get the affine transformation values

aff = pd.DataFrame(df['transform'].values.tolist())

aff.head()
for col in aff.columns:

    print(aff[col].value_counts(), '\n')
df = pd.merge(df, aff, left_index=True, right_index=True, how='inner')
def plot_average_raster(rasters, band=1, output_file='tmp.tif', avg='mean'):

    all_no2s = []

    print(f'Processing {len(rasters)} files')

    for r in rasters:

        if r.closed:

            r = rio.open(r.name)

        all_no2s += [r.read()[band-1, :, :]]

        r.close()

    temporal_no2 = np.stack(all_no2s)

    

    if avg == 'mean':

        avg_no2 = np.nanmean(temporal_no2, axis=(0))

    else:

        avg_no2 = np.nanmedian(temporal_no2, axis=(0))



    raster = rasters[0]

    

    new_dataset = rio.open(

        output_file,

        'w',

        driver=raster.driver,

        height=raster.height,

        width=raster.width,

        count=1,

        dtype=avg_no2.dtype,

        crs=raster.crs,

        transform=raster.transform,

    )

    

    new_dataset.write(avg_no2, 1)

    new_dataset.close()

    

    tmp = rio.open(output_file)

    

    print('Ranges from {:.2E} to {:.2E}'.format(np.nanmin(tmp.read(1)),

                                                np.nanmax(tmp.read(1))))

    

    # https://rasterio.readthedocs.io/en/latest/topics/plotting.html

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))



    show(tmp, transform=tmp.transform, ax=ax1)

    

    show((tmp, 1), cmap='Greys_r', interpolation='none', ax=ax2)

    show((tmp, 1), contour=True, ax=ax2)



    plt.show()

    

    return tmp
print('All files, mean', '\n')



tmp = plot_average_raster(df.tif.tolist(), avg='mean')
print('No translation files, mean', '\n')



tmp = plot_average_raster(df.query('f == 17.901140352016327 and c == -67.32431391288841').tif.tolist(),

                          avg='mean')
print('All files', '\n')



tmp = plot_average_raster(df.tif.tolist(), avg='median')