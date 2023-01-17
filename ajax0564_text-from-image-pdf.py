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

!ls /kaggle/input/Test440.jpg 


pdfs_path=glob.glob('../input/*.pdf')

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
pd_pdfs=pd.DataFrame(pdfs_info,columns=["File Name","Page Number","Text"])
pd_pdfs
# Getting text from images

images_path=glob.glob("../input/*['jpg']")

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
