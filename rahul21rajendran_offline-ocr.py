import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install git+https://github.com/myhub/tr.git@master
from tr import *

from PIL import Image, ImageDraw, ImageFont
img_pil = Image.open("/kaggle/input/testfol/new.jpg")

MAX_SIZE = 2000

if img_pil.height > MAX_SIZE or img_pil.width > MAX_SIZE:

    scale = max(img_pil.height / MAX_SIZE, img_pil.width / MAX_SIZE)



    new_width = int(img_pil.width / scale + 0.5)

    new_height = int(img_pil.height / scale + 0.5)

    img_pil = img_pil.resize((new_width, new_height), Image.BICUBIC)



print(img_pil.width, img_pil.height)

img_pil
#combined text 

img_pil2 = Image.open("/kaggle/input/combinedtext/Arialaj.png")

MAX_SIZE = 2000

if img_pil2.height > MAX_SIZE or img_pil2.width > MAX_SIZE:

    scale = max(img_pil2.height / MAX_SIZE, img_pil2.width / MAX_SIZE)



    new_width = int(img_pil2.width / scale + 0.5)

    new_height = int(img_pil2.height / scale + 0.5)

    img_pil2 = img_pil2.resize((new_width, new_height), Image.BICUBIC)



print(img_pil2.width, img_pil2.height)

img_pil2
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
gray_pil2 = img_pil2.convert("L")



rect_arr = detect(img_pil2, FLAG_RECT)



img_draw = ImageDraw.Draw(img_pil2)

colors = ['red', 'green', 'blue', "yellow", "pink"]



for i, rect in enumerate(rect_arr):

    x, y, w, h = rect

    img_draw.rectangle(

        (x, y, x + w, y + h),

        outline=colors[i % len(colors)],

        width=4)



img_pil2
blank_pil = Image.new("L", img_pil.size, 255)

blank_draw = ImageDraw.Draw(blank_pil)



results = run(gray_pil)

for line in results:

    x, y, w, h = line[0]

    txt = line[1]

    font = ImageFont.truetype("/kaggle/input/ocr-test/msyh.ttf", max(int(h * 0.6), 14))

    blank_draw.text(xy=(x, y), text=txt, font=font)



blank_pil
blank_pil2 = Image.new("L", img_pil2.size, 255)

blank_draw2 = ImageDraw.Draw(blank_pil2)



results = run(gray_pil2)

for line in results:

    x, y, w, h = line[0]

    txt = line[1]

    font = ImageFont.truetype("/kaggle/input/ocr-test/msyh.ttf", max(int(h * 0.6), 14))

    blank_draw2.text(xy=(x, y), text=txt, font=font)



blank_pil2