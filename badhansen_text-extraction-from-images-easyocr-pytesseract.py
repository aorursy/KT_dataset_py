# OCR using easyocr



!pip install easyocr
import torch

import easyocr

import os
# In case you do not have GPU or your GPU has low memory, 

# you can run it in CPU mode by adding gpu = False



# reader = easyocr.Reader(['en', 'en'], gpu=False)



reader = easyocr.Reader(['en', 'en'])
# Image to text using easyocr

# Output will be in list format, each item represents bounding box, text and confident level, respectively.



img_text = reader.readtext('../input/hackerearthimage/Test1161.jpg')

final_text = ""



for _, text, __ in img_text: # _ = bounding box, text = text and __ = confident level

    final_text += " "

    final_text += text

final_text
# Function to Traverse the folder



def traverse(directory):

    path, directory, files = next(os.walk(directory))

    return files
# Image directory and list of files



directory = '../input/hackerearthimage'

files_list = traverse(directory)
files_list[:4]
# Doing OCR using GPU

# save the images text to dict



images_text = {}

for files in files_list:

    img_text = reader.readtext(directory + '/' +  files)

    final_text = ""

    for _, text, __ in img_text:

        final_text += " "

        final_text += text

    images_text[files] = final_text
# For sorting the image file name



keys = list(images_text.keys())

new_keys = [int(k[4:-4]) for k in keys]

new_keys.sort()
# Saving the Text file with image name ascending order



import csv



with open('image_easy_ocr.csv', 'w') as file:

    writer = csv.writer(file)

    writer.writerow(["Filename", "Text"])

    

    for n in new_keys:

        writer.writerow(['Test' + str(n) + '.jpg', images_text['Test' + str(n) + '.jpg']])
# OCR using pytesseract



import cv2

import pytesseract

from pytesseract import Output

import pytesseract

from PIL import Image, ImageEnhance, ImageFilter
# Grayscale, Gaussian blur, Otsu's threshold

image = cv2.imread("../input/hackerearthimage/Test1161.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3), 0)

thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]





# Morph open to remove noise and invert image

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

invert = 255 - opening





# Perform text extraction

data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')

print(data)
def text_extraction(file_path):

    # Grayscale, Gaussian blur, Otsu's threshold

    image = cv2.imread(file_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3,3), 0)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]





    # Morph open to remove noise and invert image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    invert = 255 - opening





    # Perform text extraction

    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')

    return data
# Doing OCR using pytesseract

# save the images text to dict



images_text = {}

for files in files_list:

    img_text = text_extraction(directory + '/' +  files)

    final_text = ""

    for text in img_text:

        final_text += text

    images_text[files] = final_text
# For sorting the image file name



keys = list(images_text.keys())

new_keys = [int(k[4:-4]) for k in keys]

new_keys.sort()
# Saving the Text file with image name ascending order



import csv



with open('image_pytesseract_ocr.csv', 'w') as file:

    writer = csv.writer(file)

    writer.writerow(["Filename", "Text"])

    

    for n in new_keys:

        writer.writerow(['Test' + str(n) + '.jpg', images_text['Test' + str(n) + '.jpg']])