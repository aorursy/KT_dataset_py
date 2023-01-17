import os



import rasterio as rio

from rasterio.plot import show, show_hist

import matplotlib.pyplot as plt

from matplotlib import animation

import seaborn as sns



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

            'path': no2_path + f.split('/')[-1],

            'measurement': 'no2',

            'band1_mean': np.nanmean(raster.read(1)),

            **raster.meta

        }]

        raster.close()

        

    # Get dates

    for d in data:

        d.update({'datetime': datetime.strptime(d['filename'][:23], 's5p_no2_%Y%m%dT%H%M%S')})



    for d in data:

        d['date'] = d['datetime'].date()

        d['hour'] = int(datetime.strftime(d['datetime'], '%H'))

        d['weekday'] = d['datetime'].weekday()  # Mon = 0



    return data



def load_gfs():

    # Get the data filenames

    no2_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/'

    no2_files = [no2_path + f for f in os.listdir(no2_path)]

    

    data = []

    

    print(f'Reading {len(no2_files)} files')

    for f in no2_files:

        raster = rio.open(f)

        data += [{

            'tif': raster,

            'filename': f.split('/')[-1],

            'path': no2_path + f.split('/')[-1],

            'measurement': 'weather',

            'band1_mean': np.nanmean(raster.read(1)),

            **raster.meta

        }]

        raster.close()

        

    # Get dates

    for d in data:

        d.update({'datetime': datetime.strptime(d['filename'], 'gfs_%Y%m%d%H.tif')})



    for d in data:

        d['date'] = d['datetime'].date()

        d['hour'] = int(datetime.strftime(d['datetime'], '%H'))

        d['weekday'] = d['datetime'].weekday()  # Mon = 0



    return data



data = load_sp5() + load_gfs()

# data.append(load_gfs)

data[0]
df = pd.DataFrame(data)

df.head()
# Add total wind magnitude

for d in data:

    if d['measurement'] == 'weather':

        r = rio.open(d['path'])

        d.update({

            'avg_wind_magnitude': 

            np.nanmean(np.sqrt(np.square(r.read(4)) + np.square(r.read(5)))),

            'max_wind_magnitude': 

            np.nanmax(np.sqrt(np.square(r.read(4)) + np.square(r.read(5))))

        })

        r.close()

plt.hist([d['max_wind_magnitude'] for d in data if d['measurement'] == 'weather'])

plt.hist([d['avg_wind_magnitude'] for d in data if d['measurement'] == 'weather'])
def raster_median(rasters, band=1):

    tmp = []

    for r in rasters:

        r = rio.open(r.name)

        tmp += [r.read(band)]

        r.close()

    return [np.nanmedian(np.stack(tmp), axis=(0))]
df = pd.DataFrame(data)

df_daily = df.groupby('date').agg({

    'avg_wind_magnitude': 'median',

    'max_wind_magnitude': 'max',

    'tif': raster_median,

    'measurement': 'first'

}).rename(columns={'tif': 'values'})

df_daily.plot(kind='line')
quartile25 = df_daily['avg_wind_magnitude'].quantile(0.25)

quartile75 = df_daily['avg_wind_magnitude'].quantile(0.75)



# df_daily.plot(kind='line')

print('Low Wind Days')

df_daily.query('avg_wind_magnitude < @quartile25').plot(kind='line', marker='.', linestyle='')

print("High Wind Days")

df_daily.query('avg_wind_magnitude > @quartile75').plot(kind='line', marker='.', linestyle='')
def plot_average_raster(rasters, band=1, output_file='tmp.tif', avg='mean'):

    all_no2s = []

    print(f'Processing {len(rasters)} files')

    for r in rasters:

        if r.closed:

            r = rio.open(r.name)

        all_no2s += [r.read()[band-1, :, :]]

        r.close()

    temporal_no2 = np.stack(all_no2s)

    print("Done")

    

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

    

    new_dataset.write(avg_no2*10**5, 1)

    new_dataset.close()

    

    tmp = rio.open(output_file)

    

    min_val = np.nanmin(tmp.read(1))

    max_val = np.nanmax(tmp.read(1))

    print('Ranges from {:.2E} to {:.2E}'.format(min_val, max_val))

    

    # https://rasterio.readthedocs.io/en/latest/topics/plotting.html

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))



    # Augment the data so that it can be plotted nicely

    # mult_fact = 10**(-round(np.log10(min_val))+1)

    #data = tmp.read(1)*10**mult_fact

    

    show(tmp, transform=tmp.transform, ax=ax1)

    

    show(tmp, cmap='Greys_r', interpolation='none', ax=ax2)

    show(tmp, contour=True, ax=ax2)



    plt.show()

    

    return tmp
def plot_diff_raster(rasters_1, rasters_2, band=1, output_file='tmp.tif'):

    all_no2s_1 = []

    all_no2s_2 = []

    # Read raster 1

    print(f'Processing {len(rasters_1)} files')

    for r in rasters_1:

        r = rio.open(r.name)

        all_no2s_1 += [r.read()[band-1, :, :]]

        r.close()

    temporal_no2_1 = np.stack(all_no2s_1)

    # Read raster 2

    print(f'Processing {len(rasters_2)} files')

    for r in rasters_2:

        r = rio.open(r.name)

        all_no2s_2 += [r.read()[band-1, :, :]]

        r.close()

    temporal_no2_2 = np.stack(all_no2s_2)

        

    # Calculate averages

    avg_no2_1 = np.nanmean(temporal_no2_1, axis=(0))

    avg_no2_2 = np.nanmean(temporal_no2_2, axis=(0))



    avg_no2 = avg_no2_2 - avg_no2_1

    

    raster = rasters_1[0]

    

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

    

    new_dataset.write(avg_no2*10**5, 1)

    new_dataset.close()

    

    tmp = rio.open(output_file)

    

    print('Ranges from {:.2E} to {:.2E}'.format(np.nanmin(tmp.read(1)),np.nanmax(tmp.read(1))))

    

    # https://rasterio.readthedocs.io/en/latest/topics/plotting.html

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    

    show(tmp, transform=tmp.transform, ax=ax1)

    

    show(tmp, cmap='Greys_r', interpolation='none', ax=ax2)

    show(tmp, contour=True, ax=ax2)



    plt.show()

    

    return tmp
print("Low Wind Days")

plot_average_raster(

    df[df.date.isin(

        df_daily.query('avg_wind_magnitude < @quartile25*0.9'

                      ).index.tolist())].query('measurement == "no2"').tif.tolist()

)

print("High Wind Days")

avg = plot_average_raster(

    df[df.date.isin(

        df_daily.query('avg_wind_magnitude > @quartile75'

                      ).index.tolist())].query('measurement == "no2"').tif.tolist()

)

plot_diff_raster(

    df[df.date.isin(

        df_daily.query('avg_wind_magnitude < @quartile25*0.9'

                      ).index.tolist())].query('measurement == "no2"').tif.tolist(),

    df[df.date.isin(

        df_daily.query('avg_wind_magnitude > @quartile75'

                      ).index.tolist())].query('measurement == "no2"').tif.tolist()    

)
corr_no2 = df.query('measurement == "no2"').corr()

corr_qgs = df.query('measurement == "weather"').corr()





# https://rasterio.readthedocs.io/en/latest/topics/plotting.html

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))



sns.heatmap(

    corr_no2, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True,

    ax=ax1

)

ax1.set_xticklabels(

    ax1.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

plt.title("NO2")



sns.heatmap(

    corr_qgs, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True,

    ax=ax2

)

ax2.set_xticklabels(

    ax2.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

plt.title('Weather')

plt.show()

print("Less wind values, avg. hour: {}".format(

df[df.date.isin(

    df_daily.query('avg_wind_magnitude < @quartile25*0.9'

                  ).index.tolist())].query('measurement == "no2"').hour.mean()

))



print("More wind values, avg. hour: {}".format(

df[df.date.isin(

    df_daily.query('avg_wind_magnitude > @quartile75'

                  ).index.tolist())].query('measurement == "no2"').hour.mean()

))