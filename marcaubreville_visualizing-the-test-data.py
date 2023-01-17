# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from dicomslide import DicomSlide

dcmsl = DicomSlide('/kaggle/input/mitosis-wsi-ccmct-test-set/f26e9fcef24609b988be.dcm')

location=(69000,20500)

size=(1000,1000)

img = dcmsl.read_region(location=location, size=size, level=0 )

img
from sliderunnerdatabase import Database, ViewingProfile

DB = Database()

DB.open('/kaggle/input/mitosis-wsi-ccmct-test-set/MITOS_WSI_CCMCT_ODAEL_test.dcm.sqlite')

# look up slide in database

slideid = DB.findSlideWithFilename('f26e9fcef24609b988be.dcm','')

DB.loadIntoMemory(slideid)
imgarr = np.array(img)

vp=ViewingProfile()

vp.activeClasses = [0,0,1,0,0,0,1,1]

DB.annotateImage(imgarr, leftUpper=location, rightLower=[sum(x) for x in zip(location,size)], vp=vp, zoomLevel=1.0, selectedAnnoID=None)

Image.fromarray(imgarr)