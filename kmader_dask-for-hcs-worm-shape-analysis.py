%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from skimage.util import montage as montage2d

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from skimage.io import imread

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
from multiprocessing.pool import ThreadPool

import dask

import dask.array as da

import dask.dataframe as ddf

import dask.bag as dbag
# for progress bars, debugging and visualization

import dask.diagnostics as diag

from bokeh.io import output_notebook

from bokeh.resources import CDN

output_notebook(CDN, hide_banner=True)
image_bag = dbag.from_sequence(glob('../input/BBBC010_v1_images/*_w1_*.tif'))

image_bag
def _get_light_path(in_path):

    """Convert the tif path into a light-field path"""

    w2_path='_w2_'.join(in_path.split('_w1_'))

    glob_str='_'.join(w2_path.split('_')[:-1]+['*.tif'])

    m_files=glob(glob_str)

    if len(m_files)>0:

        return m_files[0]

    else:

        return None

image_df = image_bag.map(lambda f: {'gfp_path': f, 'light_path': _get_light_path(f)}).to_dataframe()

image_df
image_df=image_df.dropna()

image_df['base_name']=image_df['gfp_path'].map(lambda x: os.path.splitext(os.path.basename(x))[0], meta=('base_name', 'str'))

image_df['plate_rc']=image_df['base_name'].map(lambda x: x.split('_')[6], meta=('plate_rc', 'str'))

image_df['row']=image_df['plate_rc'].map(lambda x: x[0:1], meta=('row', 'str'))

image_df['column']=image_df['plate_rc'].map(lambda x: int(x[1:]), meta=('column', 'int'))

image_df['treated']=image_df['column'].map(lambda x: 'ampicillin' if x<13 else 'negative control', meta=('treated', 'str'))

image_df['wavelength']=image_df['base_name'].map(lambda x: x.split('_')[7], meta=('wavelength', 'str'))



image_df['mask_path']=image_df['plate_rc'].map(lambda x: '../input/BBBC010_v1_foreground/{}_binary.png'.format(x))

print('Loaded',image_df.shape[0],'datasets')

image_df
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:

    with dask.config.set(pool = ThreadPool(4)):

        print('Loaded',image_df.shape[0].compute(),'datasets')
# show the first row

image_df.head(1)
from skimage.measure import label

from skimage.measure import regionprops

image_df['worm_image']=image_df['mask_path'].map(lambda x: imread(x)[:,:,0]>0, meta=('worm_image', np.object))

image_df['worm_labels']=image_df['worm_image'].map(lambda x: label(x), meta=('worm_labels', np.object))

image_df['worm_regions']=image_df['worm_labels'].map(lambda x: regionprops(x), meta=('worm_regions', np.object))

image_df['worm_volume_fraction']=image_df['worm_image'].map(lambda x: np.mean(x), meta=('worm_volume_fraction', 'float'))

image_df['worm_nsegments']=image_df['worm_labels'].map(lambda x: np.max(x), meta=('worm_nsegments', 'int'))

image_df['worm_p2a_ratio']=image_df['worm_regions'].map(lambda x: np.mean([rp.perimeter/rp.area for rp in x]), meta=('worm_p2a_ratio', 'float'))

image_df
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:

    with dask.config.set(pool = ThreadPool(4)):

        collected_df = image_df.drop(['column'],1).compute()
diag.visualize([prof, rprof])
collected_df.sample(3)
sns.pairplot(collected_df,hue='treated')