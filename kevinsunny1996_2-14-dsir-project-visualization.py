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
from matplotlib import pyplot as plt
fire_nrt_m6 = pd.read_csv('../input/fires-from-space-australia-and-new-zeland/fire_nrt_M6_96619.csv')

fire_nrt_v1 = pd.read_csv('../input/fires-from-space-australia-and-new-zeland/fire_nrt_V1_96617.csv')

fire_archive_m6 = pd.read_csv('../input/fires-from-space-australia-and-new-zeland/fire_archive_M6_96619.csv')

fire_archive_v1 = pd.read_csv('../input/fires-from-space-australia-and-new-zeland/fire_archive_V1_96617.csv')
fire_archive_m6.head()
percent_brightness = (fire_nrt_v1['bright_ti5']/fire_nrt_v1['bright_ti4'])*100
plt.scatter(fire_nrt_v1['longitude'],fire_nrt_v1['latitude'],s = percent_brightness,cmap='viridis',alpha=0.3)

plt.show()
from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://www.freeworldmaps.net/australia/australia-map.png")
percent_brightness_m6 = (fire_nrt_m6['brightness']/fire_nrt_m6['bright_t31'])*100
plt.scatter(fire_nrt_m6['longitude'],fire_nrt_m6['latitude'],s = percent_brightness_m6,cmap='viridis',alpha=0.3)

plt.show()
percent_brightness_m6_archive = (fire_archive_m6['brightness']/fire_archive_m6['bright_t31'])*100
plt.scatter(fire_archive_m6['longitude'],fire_archive_m6['latitude'],s = percent_brightness_m6_archive,cmap='viridis',alpha=0.3)

plt.show()
percent_brightness_v1_archive = (fire_archive_v1['bright_ti5']/fire_archive_v1['bright_ti4'])*100
plt.scatter(fire_archive_v1['longitude'],fire_archive_v1['latitude'],s = percent_brightness_v1_archive,cmap='viridis',alpha=0.3)

plt.show()
Image(url= "https://www.anbg.gov.au/aust-veg/aust-veg-map.gif")