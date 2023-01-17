# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
def image_with_mask(img, mask):
    # returns a copy of the image with edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    return img_color

def mask_not_blank(mask):
    return sum(mask.flatten()) > 0

def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img)) 

def fimg_to_fmask(img_path):
    # convert an image file path into a corresponding mask file path 
    dirname, basename = os.path.split(img_path)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)

print(mask_not_blank(plt.imread("../input/train/10_100_mask.tif")))
#from subprocess import check_output
#print(check_output(["ls", "../input/train"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
IMAGES_TO_SHOW = 5  # configure to taste :)


import numpy as np
import matplotlib.pyplot as plt
import glob, os, os.path
import cv2


def image_with_mask(img, mask):
    # returns a copy of the image with edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    return img_color

def fimg_to_fmask(img_path):
    # convert an image file path into a corresponding mask file path 
    dirname, basename = os.path.split(img_path)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)

def mask_not_blank(mask):
    return sum(mask.flatten()) > 0

def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img)) 

def plot_image(img, title=None):
    plt.figure(figsize=(15,20))
    plt.title(title)
    plt.imshow(img)
    plt.show()

def main():

    f_ultrasounds = [img for img in glob.glob("../input/train/*.tif") if 'mask' not in img]
    # f_ultrasounds.sort()  
    f_masks       = [fimg_to_fmask(fimg) for fimg in f_ultrasounds]
    
    images_shown = 0 

    for f_ultrasound, f_mask in zip(f_ultrasounds, f_masks):

        img  = plt.imread(f_ultrasound)
        mask = plt.imread(f_mask)

        if mask_not_blank(mask):

            # plot_image(grays_to_RGB(img),  title=f_ultrasound)
            # plot_image(grays_to_RGB(mask), title=f_mask)

            f_combined = f_ultrasound + " & " + f_mask 
            plot_image(image_with_mask(img, mask), title=f_combined)
            print('plotted:', f_combined)
            images_shown += 1

        if images_shown >= IMAGES_TO_SHOW:
            break

main()
from subprocess import check_output
#print(check_output(["ls", "../input/"]).decode("utf8"))
import pandas as pd
train_masks = pd.read_csv("../input/train_masks.csv")
train_masks
import pandas as pd
pd.read_csv("../input/sample_submission.csv")
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import random

# Definitions
IMAGES_TO_SHOW = 20

# Plot image
def plot_image(img, title=None):
    plt.figure(figsize=(15,20))
    plt.title(title)
    plt.imshow(img)
    plt.show()
    
# Draw elipsis on image
def draw_ellipse(mask):
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    has_ellipse = len(contours) > 0
    if has_ellipse:
        cnt = contours[0]
        ellipse = cv2.fitEllipse(cnt)
        cx, cy = np.array(ellipse[0], dtype=np.int)
        m3[cy-2:cy+2,cx-2:cx+2] = (255, 0, 0)
        cv2.ellipse(m3, ellipse, (0, 255, 0), 1)
        
    return has_ellipse, m3

# Read some files
mfiles = glob.glob("../input/train/*_mask.tif")
random.shuffle(mfiles) # Shuffle for random results

files_with_ellipse = 0
for mfile in mfiles:
    mask = cv2.imread(mfile, -1)  # imread(..., -1) returns grayscale images
    has_ellipse, mask_with_ellipse = draw_ellipse(mask)
    if has_ellipse:
        files_with_ellipse = files_with_ellipse+1
        plot_image(mask_with_ellipse, mfile)
        if files_with_ellipse > IMAGES_TO_SHOW:
            break

