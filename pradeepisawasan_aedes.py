import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# A image of Aedes aegypti (Linnaeus) 

img_array = np.array(Image.open('../input/aedes-mosquitos/aegypti/aegypti/aegypti_0/aegypti100a.jpg'))

plt.imshow(img_array)
# A image of Aedes albopictus (Skuse) 

img_array = np.array(Image.open('../input/aedes-mosquitos/albopictus/albopictus/albopictus_0/albopictus100a.jpg'))

plt.imshow(img_array)