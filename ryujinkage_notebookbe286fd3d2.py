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

import cv2

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

DATADIR = "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images"



for img in os.listdir(DATADIR):

    img_array = cv2.imread(os.path.join(DATADIR,img))

    plt.imshow(img_array)

    plt.show()

    break

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(42)

import os

import pickle

import collections

from PIL import Image

import plotly.express as px

import plotly.graph_objects as go

import re

from tqdm import notebook
IMAGES_DIR = '/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images'

image_filenames = os.listdir(IMAGES_DIR)

file_extentions = [filename.split('.')[-1] for filename in image_filenames]



images_paths = [os.path.join(IMAGES_DIR,filename) for filename in image_filenames]



REF_FILE = '/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/reference_df_pickle'

LABELS_FILE = '/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/labels_pd_pickle'



with open(REF_FILE, 'rb') as handle:

    reference_df_ = pickle.load(handle)



with open(LABELS_FILE, 'rb') as handle:

    labels_pd_ = pickle.load(handle)
image_formats = collections.Counter(file_extentions)

print(f'Num Images: {len(images_paths)}')



print('Image formats found: ', image_formats)

image_formats_df = pd.DataFrame.from_dict(image_formats, orient='index').reset_index()

image_formats_df
#for converting images 

from glob import glob                                                           

import cv2 

pngs = glob('/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/*.png')



for j in pngs:

    img = cv2.imread(j)

    cv2.imwrite(j[:-3] + 'jpg', img)

    

#for converting images 

from glob import glob                                                           

import cv2 

pngs = glob('/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/*.jpeg')



for j in pngs:

    img = cv2.imread(j)

    cv2.imwrite(j[:-3] + 'jpg', img)

    

#for converting images 

from glob import glob                                                           

import cv2 

pngs = glob('/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/*.PNG')



for j in pngs:

    img = cv2.imread(j)

    cv2.imwrite(j[:-3] + 'jpg', img)

    

#for converting images 

from glob import glob                                                           

import cv2 

pngs = glob('/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/*.JPG')



for j in pngs:

    img = cv2.imread(j)

    cv2.imwrite(j[:-3] + 'jpg', img)

    

#for converting images 

from glob import glob                                                           

import cv2 

pngs = glob('/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/*.jpe')



for j in pngs:

    img = cv2.imread(j)

    cv2.imwrite(j[:-3] + 'jpg', img)

    

#for converting images 

from glob import glob                                                           

import cv2 

pngs = glob('/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/*.bmp')



for j in pngs:

    img = cv2.imread(j)

    cv2.imwrite(j[:-3] + 'jpg', img)

    

image_formats = collections.Counter(file_extentions)

print(f'Num Images: {len(images_paths)}')



print('Image formats found: ', image_formats)

image_formats_df = pd.DataFrame.from_dict(image_formats, orient='index').reset_index()

image_formats_df
#for deleting the images

import glob, os, os.path



mydir = "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/"



filelist = glob.glob(os.path.join(mydir, "*.bmp"))

for f in filelist:

    os.remove(f)
#for deleting the images

import glob, os, os.path



mydir = "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/"



filelist = glob.glob(os.path.join(mydir, "*.jpe"))

for f in filelist:

    os.remove(f)
#for deleting the images

import glob, os, os.path



mydir = "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/"



filelist = glob.glob(os.path.join(mydir, "*.JPG"))

for f in filelist:

    os.remove(f)
#for deleting the images

import glob, os, os.path



mydir = "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/"



filelist = glob.glob(os.path.join(mydir, "*.PNG"))

for f in filelist:

    os.remove(f)
#for deleting the images

import glob, os, os.path



mydir = "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/"



filelist = glob.glob(os.path.join(mydir, "*.jpeg"))

for f in filelist:

    os.remove(f)
#for deleting the images

import glob, os, os.path



mydir = "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/"



filelist = glob.glob(os.path.join(mydir, "*.png"))

for f in filelist:

    os.remove(f)
#for resizing the images

import PIL

import os

import os.path

from PIL import Image



f = r'/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images/'

for file in os.listdir(f):

    f_img = f+"/"+file

    img = Image.open(f_img)

    img = img.resize((300,300))

    img.save(f_img)
BLUR = 21

CANNY_THRESH_1 = 10

CANNY_THRESH_2 = 200

MASK_DILATE_ITER = 10

MASK_ERODE_ITER = 10

MASK_COLOR = (0.0,0.0,1.0)
gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)



edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)

edges = cv2.dilate(edges, None)

edges = cv2.erode(edges, None)

plt.imshow(img_array)

plt.show()

plt.imshow(gray)

plt.show()

plt.imshow(edges)

plt.show()