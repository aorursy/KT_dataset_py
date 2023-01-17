# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os



# There are two ways to load the data from the PANDA dataset:

# Option 1: Load images using openslide

import openslide

# Option 2: Load images using skimage (requires that tifffile is installed)

import skimage.io



# General packages

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

from IPython.display import Image, display



# Plotly for the interactive viewer (see last section)

import plotly.graph_objs as go

data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '0200fa88e8c01546663b9db1726936ac.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((19357, 9075), 0, (450, 450))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '04201ee15d33abe4b032317792cf1040.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((2820,22398), 0, (450, 450))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '019c1b40e6ec7410e8356c5d8d487954.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((5306, 11343), 0, (1100, 1500))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '00412139e6b04d1e1cee8421f38f6e90.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((4494, 11253), 0, (700, 700))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '00412139e6b04d1e1cee8421f38f6e90.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((3204, 16449), 0, (900, 1050))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '019c1b40e6ec7410e8356c5d8d487954.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((8714, 33011), 0, (600, 600))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '060f60eeecf1ab502526a34db9caaf8e.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((5603, 3150), 0, (300, 300))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '035587e63d72f8537b3fce5067bb18df.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((2131, 24216), 0, (600, 400))



# Display the image

display(patch)



# Close the opened slide after use

image.close()


# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, 'fe2e96fbf3e76e9983d1cc4d90ef957b.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((13305, 8016), 0, (1200, 625))



# Display the image

display(patch)



# Close the opened slide after use

image.close()

# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '060f60eeecf1ab502526a34db9caaf8e.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((5009, 6927), 0, (675, 500))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
#RBC



# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '035587e63d72f8537b3fce5067bb18df.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((2412, 21305), 0, (650, 650))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, '033e39459301e97e457232780a314ab7.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((15584, 1210), 0, (1500, 2500))



# Display the image

display(patch)



# Close the opened slide after use

image.close()
# Open the image (does not yet read the image into memory)

image = openslide.OpenSlide(os.path.join(data_dir, 'cd894e213f1522b7dfc41d457dc4899b.tiff'))



# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.

# At this point image data is read from the file and loaded into memory.

patch = image.read_region((13626, 4409), 0, (1500, 2000))



# Display the image

display(patch)



# Close the opened slide after use

image.close()