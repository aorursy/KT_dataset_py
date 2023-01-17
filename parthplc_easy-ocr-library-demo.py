!pip install easyocr
import matplotlib.pyplot as plt

import cv2

import easyocr

from pylab import rcParams

from IPython.display import Image

rcParams['figure.figsize'] = 8, 16
reader = easyocr.Reader(['en'])
!ls ../input/
path = "../input/"
import PIL

from PIL import ImageDraw

img = PIL.Image.open(path+"1.jpg")

img
output = reader.readtext(path+'1.jpg')
output[0][-2],output[1][-2],output[2][-2]
def draw_boxes(image, bounds, color='yellow', width=2):

    draw = ImageDraw.Draw(image)

    for bound in bounds:

        p0, p1, p2, p3 = bound[0]

        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)

    return image



draw_boxes(img,output)
img = PIL.Image.open(path+"3.png")
img
output = reader.readtext(path+"3.png")

draw_boxes(img,output)
for i in range(len(output)):

    print(output[i][-2])
img = PIL.Image.open(path+"4.jpg")

img
output = reader.readtext(path+"4.jpg")

print(output)
for i in range(len(output)):

    print(output[i][-2])
draw_boxes(img,output)

img = PIL.Image.open(path+"1.png")

img
output = reader.readtext(path+"1.png")

print(output)
for i in range(len(output)):

    print(output[i][-2])
draw_boxes(img,output)