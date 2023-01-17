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
# Hi, i am practising data science concepts and below is the Program to Extract the text 
# from multiple Pdfs and Images from a kaggle dataset created by me and display them accordingly.
# Below code snipet is run one time, in order to see all the outputs of a cell despite of 
# printing only last output of a cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
!pip install PyPDF2
import wand
import glob
import PyPDF2
import pytesseract
import pandas as pd
from PIL import Image as img1
from wand.image import Image


pdfs_path=glob.glob('../input/filesdata/*.pdf')
pdfs_info=[]
# Getting text from pdfs
for path in pdfs_path:
    with(Image(filename=path, resolution=120)) as source: 
        for i, image in enumerate(source.sequence):
            newfilename = "/kaggle/working/" + path.split("/")[-1][:-4] +  "_" + str(i + 1) + '.jpeg'
            Image(image).save(filename=newfilename)
            img=img1.open(newfilename)
            print(path.split("/")[-1], newfilename[-6])
            pdfs_info.append([path.split("/")[-1], newfilename[-6] , pytesseract.image_to_string(img)])

# DataFrame creation for Displaying the info. of pdf to text 
pd_pdfs=pd.DataFrame(pdfs_info,columns=["File Name","Page Number","Text"])
pd_pdfs

# Getting text from images
images_path=glob.glob("../input/filesdata/*['jpg','png','jpeg']")
image_count=0
images_info=[]
for path in images_path:
    img=img1.open(path)
    image_count+=1
    print(path.split("/")[-1],image_count)
    images_info.append([path.split("/")[-1],image_count, pytesseract.image_to_string(img)])

# DataFrame creation for Displaying the info. of images to text     
pd_images=pd.DataFrame(images_info,columns=["File Name","Image Number","Text"])
pd_images