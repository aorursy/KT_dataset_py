import cv2

cv2.__version__
!apt install tesseract-ocr-spa
!ls /usr/share/tesseract-ocr/tessdata/
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "/kaggle/input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/usr/share/tesseract-ocr/tessdata/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#!/usr/bin/env python

# coding: utf-8



# In[92]:





import cv2

import pytesseract

import json





class reciboOCR:



    def __init__(self, file_name):

        self.file_name = file_name

        # pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Jorge\AppData\Local\Tesseract-OCR\tesseract.exe'

        # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'





    def obtener_parrafos(self, file_name):

        img = cv2.imread(file_name)

        img = cv2.resize(img, (1200, 1200), interpolation=cv2.INTER_AREA)

        parrafos = []

        img_final = cv2.imread(file_name)

        img_final = cv2.resize(img_final, (1200, 1200))

        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





        ret, mascara = cv2.threshold(img_gris, 100, 255, cv2.THRESH_BINARY)

        image_final = cv2.bitwise_and(img_gris, img_gris, mask=mascara)

        ret, new_img = cv2.threshold(image_final, 100, 255, cv2.THRESH_BINARY_INV)



        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        dilated = cv2.dilate(new_img, kernel, iterations=10)

        contornos, __ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #print(contornos)



        # config = ('-l spa --oem 1 --psm 3')

        config = ('--tessdata-dir "/usr/share/tesseract-ocr/tessdata/" -l eng --oem 1 --psm 3')



        for contorno in contornos:

            [x, y, w, h] = cv2.boundingRect(contorno)



            #if w < 50 or h < 50:  # filtramos algunos falsos positivos.

            #    continue

                # if w>100 and h>100 or h>1000:



                #   continue

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)



            parrafo = pytesseract.image_to_string(img_gris[y:y + h, x:x + w], config=config)

            #print(parrafo)

            if 'CFE' not in parrafo:

                parrafos.append(parrafo)

            #print("#"*80)

            #print(parrafos)



        #### Quitar comentarios abajo si se desea que se muestre la imagen con los recuadros

        #cv2.imshow('Resultado', img)

        #cv2.imshow('gris', dilated)

        #cv2.imwrite('luz(procesada).jpg',img)

        #cv2.waitKey(0)

        #cv2.destroyAllWindows()

        return parrafos





    def OCR(self):

        texto = self.obtener_parrafos(self.file_name)

        return self.obtener_direccion(texto)





    def obtener_direccion(self,parrafos):



        for string in range(len(parrafos)):

            if 'Mary' in parrafos[string]:

                return (parrafos[string])





    # In[93]:





    import sys



# archivo = 'VGPO-IN-000002-01-19_SL1.jpg'

archivo = "/kaggle/input/hero-phone-single-v1.1.png"

o = reciboOCR(archivo)

print(o.OCR())



from IPython.display import Image

Image(archivo)