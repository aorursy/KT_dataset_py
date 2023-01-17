# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import rasterio as ras
import matplotlib.pyplot as plt
from rasterio import plot
%matplotlib inline
mappath = '/kaggle/input/senthinelv1/'

band5 = ras.open(mappath+'T43PFS_20200531T051701_B05_20m.jp2',driver='JP2openJPEG')
band4 = ras.open(mappath+'T43PFS_20200531T051701_B04_20m.jp2',driver='JP2openJPEG')
band3 = ras.open(mappath+'T43PFS_20200531T051701_B03_20m.jp2',driver='JP2openJPEG')
band2 = ras.open(mappath+'T43PFS_20200531T051701_B02_20m.jp2',driver='JP2openJPEG')




#Viewing a file in Python
plot.show(band2)
#Here we are plotting the files and comparing with the bands together
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(12,4))
plot.show(band2,ax=ax1,cmap='Blues')
plot.show(band3,ax=ax2,cmap='Greens')
plot.show(band4,ax=ax3,cmap='Reds')
plot.show(band5,ax=ax4,cmap='Purples')
fig.tight_layout


truecolor = ras.open('/kaggle/working/Truepic.tiff','w',driver='Gtiff',
                     width=band2.width,
                     height=band2.height,
                     count=4,
                     crs=band2.crs,
                     transform=band2.transform,
                     dtype='uint16')




truecolor.write(band2.read(1),3)
truecolor.write(band3.read(1),2)
truecolor.write(band4.read(1),1)
truecolor.write(band5.read(1),4)
truecolor.close()
#opening tiff file in Python
src = ras.open(r'/kaggle/working/Truepic.tiff',count=4)
plot.show(src)
src = ras.open(r'/kaggle/working/Truepic.tiff',count=4)
fig, (axrgb, axhist) = plt.subplots(1, 2, figsize=(14,7))
plot.show(src, ax=axrgb)
plot.show_hist(src, bins=50, histtype='stepfilled',lw=0.0, stacked=True, alpha=0.3, ax=axhist)
plt.show()