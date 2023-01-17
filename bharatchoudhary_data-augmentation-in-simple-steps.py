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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import numpy as np
import pandas as pd
print('Number of Parasitzed cell images',len(os.listdir('/kaggle/input/malaria-small-dataset/malaria/training/Parasitized/')))
print('Number of Uninfected cell images',len(os.listdir('/kaggle/input/malaria-small-dataset/malaria/training/Uninfected/')))
!pip install Augmentor
import Augmentor
# location in system from were images is use for data augumentation(here we use parasitzed images)
p = Augmentor.Pipeline("/kaggle/input/malaria-small-dataset/malaria/training/Parasitized/")
p.rotate(probability=0.3, max_left_rotation=10, max_right_rotation=10)
p.rotate90(probability=0.3)
p.rotate270(probability=0.3)
p.flip_left_right(probability=0.3)
p.flip_top_bottom(probability=0.3)
p.flip_random(probability=0.3)
p.crop_random(probability=.1, percentage_area=0.3)
p.resize(probability=1, width=100, height=100)
p.random_brightness(probability = 0.5, min_factor=0.4, max_factor=0.9)
p.random_color(probability=0.5, min_factor=0.4, max_factor=0.9)
p.random_contrast(probability=0.5, min_factor=0.9, max_factor=1.4)
p.random_distortion(probability=0.5, grid_width=7, grid_height=8, magnitude=9)
p.random_erasing(probability=0.5, rectangle_area=0.2)
p.zoom(probability=0.7, min_factor=1.1, max_factor=1.5)
p.shear(probability=0.3, max_shear_left=0.2, max_shear_right=0.2)
p.rotate(probability=0.3, max_left_rotation=0.2, max_right_rotation=0.2)
p.skew(probability=0.3)
p.sample(100)
