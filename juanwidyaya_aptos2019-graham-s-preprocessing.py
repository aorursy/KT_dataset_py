%reload_ext autoreload
%autoreload 2
%matplotlib inline
# Preprocess training images.
# Scale 300 seems to be sufficient; 500 and 1000 may be overkill
import os
import cv2
import numpy
import fnmatch
import shutil
from matplotlib import pyplot as plt
plt.style.use("dark_background")
# create train_graham for output directory first
if os.path.isdir("train_graham") == False:
    os.mkdir("train_graham")
    
!cp "../input/aptos2019-blindness-detection/train.csv" "./"
source_dir = "../input/aptos2019-blindness-detection/train_images"
target_dir = "train_graham"
def scaleRadius(img, scale):
    # sum one row & columns over channels
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)
# for scale in [300, 500, 1000]:
scale = 300
counter = 0
for f in fnmatch.filter(os.listdir(source_dir), "*.png"):
    try:
        a = cv2.imread(os.path.join(source_dir, f))
        
        # scale img to a given radius
        a = scaleRadius(a, scale)
        
        # create masking to remove outer 10%
        b = numpy.zeros(a.shape)
        cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(scale * 0.9),
                   (1, 1, 1), -1, 8, 0)
        
        # subtract local mean color
        aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4,
                             128) * b + 128 * (1 - b)
        
        # save the image
        cv2.imwrite(os.path.join(target_dir, f), aa)

        if counter % 200 == 0:
            print("processed images: ", counter)
        counter += 1

    except:
        print(f)
output_img = fnmatch.filter(os.listdir("train_graham"), "*.png")
output_img[0]
img_output = cv2.imread(os.path.join(target_dir, output_img[0]))
img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
plt.imshow(img_output)
shutil.make_archive("train_graham", 'zip', "train_graham")
!rm -r "train_graham"
import glob
fnames = glob.glob("../input/aptos2019-blindness-detection/train_images/*.png")
fnames[:3]
img = cv2.imread(fnames[1])
scale = 500
img.shape
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.imshow(img)
# let's see, what is x, r, s
# x is the middle row, contain x columns, 1 channel
# r is the radius, like column width devided by 2
# s is scaling factor that will be applied to the image
x = img[img.shape[0] // 2, :, :].sum(1)
r = (x > x.mean() / 10).sum() / 2
s = scale * 1.0 / r
x, r, s,
x.shape
# resize the image according to scaling factor
img_rsz = cv2.resize(img, (0, 0), fx=s, fy=s)
img_rsz.shape
# apply gaussian / subtract local mean color
temp1 = cv2.GaussianBlur(img_rsz, (0, 0), scale / 30)
plt.imshow(temp1)
# add gaussian weight to the image
temp2 = cv2.addWeighted(img_rsz, 4, temp1, -4, 128)
plt.imshow(temp2)
mask = numpy.zeros(img_rsz.shape)
plt.imshow(mask)
# remove outer 10%
cv2.circle(mask, (img_rsz.shape[1] // 2, img_rsz.shape[0] // 2),
           int(scale * 0.9), (1, 1, 1), -1, 8, 0)
plt.imshow(mask)
# apply the mask
temp3 = temp2 * mask + 128 * (1 - mask)
cv2.imwrite("test.png", temp3)
load_saved = cv2.imread("test.png")
plt.imshow(load_saved)