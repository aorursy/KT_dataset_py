import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import os
import io
from PIL import Image, ImageDraw
import datetime
import random
import cv2
from matplotlib.pyplot import imshow
# Original function, using PIL module
def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0
    
    return ''.join(hex_string)
# 4x faster function, using cv2 module and some numpy tricks
def dhash_cv2(im, hash_size = 16):
    # Convert to grayscale
    imz = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Resize
    px = cv2.resize(imz, (hash_size+1,hash_size), interpolation=cv2.INTER_AREA)

    # Calculate difference between adjacent pixels
    diff = (px[:,:-1]>px[:,1:]).ravel()
    # Create hex string from every 8 bits. Steps:
    # 'difference.reshape(-1, 8)' -> Group every 8 booleans
    # 'np.where(x)[0]'            -> Convert an array of booleans to index values.
    #                                [False, True, True, False, True] becames [1, 2, 4]
    # '(1 << x).sum()'            -> Faster version of sum(2**x)
    #                                [1, 2, 4] -> [1**2, 2**2, 4**2] -> [1, 4, 16] -> 21
    # '"%0.2x" % x'               -> Convert integer to hex string (21 -> '15')
    hex_string = map(lambda x: "%0.2x" % (1 << np.where(x)[0]).sum(), diff.reshape(-1, 8))
    

    # Join hex string array
    return ''.join(hex_string)
# Generate a fake image (.zip files are not available for Kernels)
def drawImage():
    testImage = Image.new("RGB", (600,600), (255,255,255))
    draw = ImageDraw.Draw(testImage)
    for r in range(50):
        x = random.randrange(0, 500)
        y = random.randrange(0, 500)
        h = random.randrange(20, 100)
        w = random.randrange(20, 100)
        r = random.randrange(0,255)
        g = random.randrange(0,255)
        b = random.randrange(0,255)
        
        draw.rectangle(((x,y),(x+w,y+h)), fill=(r, g, b))
        
    del draw
    return testImage

# PIL instance
img = drawImage()

# Numpy instance
img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

%matplotlib inline
imshow(np.asarray(img))
# Generate hashes

hash1 = dhash(img)
hash2 = dhash_cv2(img_np)

# NOTICE THAT BOTH ARE DIFFERENT! In my debug of what caused this difference, I noticed that is due to 
# different resize functions. However, I plotted the 'difference' bool array of both methods and can
# assure that they are very close
print(hash1, hash2)
# Performance of previous function
%timeit dhash(img)
# Performance of cv2 version (4x speedup!)
%timeit dhash_cv2(img_np)