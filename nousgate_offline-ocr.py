import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install tika
from tika import parser



myPDFPath = '/kaggle/input/it-invoices/Fattura no. 4400811717.pdf'

#myPDFPath = '/kaggle/input/it-invoices/File PDF134_3.PDF'



raw = parser.from_file(myPDFPath)



if raw['content'] != None:

    

    """ File PDF vero non scanned images """

    

    extracted_text = (''.join([i.replace("\n"," ").replace("\n\n"," ") for i in raw['content']]))

    print(extracted_text)
if raw['content'] == None:

    

    """ File PDF con scanned images """

    

    import os

    import io

    from PIL import Image

    import pytesseract

    from wand.image import Image as wi

    import gc

    

    def Get_text_from_pdf(pdf_path):

        import pytesseract,io,gc

        from PIL import Image

        from wand.image import Image as wi

        import gc

        

        """ Extracting text content from PDF  """



        pdf=wi(filename=pdf_path,resolution=300)                                                                                                                

        pdfImg=pdf.convert('jpeg')                                                                                                                                                                              

        imgBlobs=[]

        extracted_text=[]

        try:        

            for img in pdfImg.sequence:

                page=wi(image=img)

                imgBlobs.append(page.make_blob('jpeg'))

                for i in range(0,5):

                    [gc.collect() for i in range(0,10)]



            for imgBlob in imgBlobs:

                im=Image.open(io.BytesIO(imgBlob))   

                

                """ in questo ambiente non ho la possibilità di utilizzare ita """

                text=pytesseract.image_to_string(im,lang='eng')  

                

                text = text.replace(r"\n", " ")

                extracted_text.append(text)

                for i in range(0,5):

                   [gc.collect() for i in range(0,10)]

            return (''.join([i.replace("\n"," ").replace("\n\n"," ") for i in extracted_text]))

            [gc.collect() for i in range(0,10)]

        finally:

            [gc.collect() for i in range(0,10)]

            img.destroy()

            

    

    returnText = Get_text_from_pdf(myPDFPath)

    print(returnText)
import os

from PIL import Image

import pytesseract

import argparse

import cv2



myIMGPath = '/kaggle/input/invoices-it/fattura_hotel_2.png'
# Convert to grayscale

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# preprocess the image

#if preprocess == "thresh": gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]



# blur the image to remove noise

#elif preprocess == "blur": gray = cv2.medianBlur(gray, 3)



# write the new grayscale image to disk 

#filename = "{}.png".format(os.getpid())

#cv2.imwrite(filename, gray)



# load the image as a PIL/Pillow image, apply OCR 

#text = pytesseract.image_to_string(Image.open(filename))

#print(text)
!pip install git+https://github.com/myhub/tr.git@master
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tr import *

from PIL import Image, ImageDraw, ImageFont



myIMGPath = '/kaggle/input/invoices-it/fattura_hotel_2.png'

img_pil = Image.open(myIMGPath)



MAX_SIZE = 2000

if img_pil.height > MAX_SIZE or img_pil.width > MAX_SIZE:

    scale = max(img_pil.height / MAX_SIZE, img_pil.width / MAX_SIZE)



    new_width = int(img_pil.width / scale + 0.5)

    new_height = int(img_pil.height / scale + 0.5)

    img_pil = img_pil.resize((new_width, new_height), Image.BICUBIC)



print(img_pil.width, img_pil.height)

img_pil
gray_pil = img_pil.convert("L")



rect_arr = detect(img_pil, FLAG_RECT)



img_draw = ImageDraw.Draw(img_pil)

colors = ['red', 'green', 'blue', "yellow", "pink"]



for i, rect in enumerate(rect_arr):

    x, y, w, h = rect

    img_draw.rectangle(

        (x, y, x + w, y + h),

        outline=colors[i % len(colors)],

        width=4)



img_pil
blank_pil = Image.new("L", img_pil.size, 255)

blank_draw = ImageDraw.Draw(blank_pil)



results = run(gray_pil)

for line in results:

    x, y, w, h = line[0]

    txt = line[1]

    font = ImageFont.truetype("/kaggle/input/ocr-test/msyh.ttf", max(int(h * 0.6), 14))

    blank_draw.text(xy=(x, y), text=txt, font=font)



blank_pil