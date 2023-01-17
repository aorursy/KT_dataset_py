import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt

data_dir = '/kaggle/input/image-grid'
adtn_dir = '/kaggle/input/bosung'

data_file_1 = os.listdir(data_dir)
data_file_2 = os.listdir(adtn_dir)

data_file_1.sort()
data_file_2.sort()

print(data_file_1)
print(data_file_2)
path_1 = os.path.join(data_dir, 'fig4.jpg')

img = cv2.imread(path_1)
img_rgb_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize = [10, 9])
plt.imshow(img_rgb_plt)
from IPython.core.display import Image, display
display(Image(os.path.join(adtn_dir, 'hsv_range.png'), width=960, unconfined=True))
import numpy as np

lower_green = np.array([35, 10, 0])             
upper_green = np.array([70, 220, 255])

def extract_color_range(img_path, lower_vl, upper_vl, crc = 0):
    """
    img_path (str) : path to image
    lower_vl, upper_vl : 1D_array of 3 values H(0-180), S(0-255), V(0-255)
    crc (integer in (0, 255)) : complement_range_color. For example 0 for black and 255 for white
    """
    img = cv2.imread(img_path)

    ## convert the img_scr to HSV mode
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## put the range of green colors to the mask in HSV's mode
    mask_green = cv2.inRange(hsv_img, lower_vl, upper_vl)

    ## get the image_area which have the green_colors
    extract_range = cv2.bitwise_and(img, img, mask = mask_green)

    ## convert to RGB mode
    extract_range = cv2.cvtColor(extract_range, cv2.COLOR_BGR2RGB)

    ## covert the complement_range to any color
    extract_range[extract_range == [0, 0, 0]] = crc

    ## display
    return extract_range


plt.figure(figsize = (25, 8))
plt.subplot(131); plt.imshow(img_rgb_plt), plt.title('original images')

plt.subplot(132); w_green_RBG = extract_color_range(path_1, lower_green, upper_green, 255)
plt.imshow(w_green_RBG); plt.title('Extract green color \n fixed complement_range = 255 (white)')

plt.subplot(133); b_green_RBG = extract_color_range(path_1, lower_green, upper_green)
plt.imshow(b_green_RBG); plt.title('Extract green color \n fixed complement_range = 0 (black)')

plt.show()
mask = w_green_RBG[w_green_RBG == 255]
mask.shape, img.shape
green_prop = 1 - mask.shape[0] / (960*960*3)
green_prop
path_2 = os.path.join(data_dir, "fig3.jpg")
img_2 = cv2.imread(path_2)
img2_rgb_plt = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

lower_blue = np.array([52, 15, 40])
upper_blue = np.array([120, 255, 255])

w_blue_RBG = extract_color_range(path_2, lower_blue, upper_blue, 255)
b_blue_RBG = extract_color_range(path_2, lower_blue, upper_blue)

fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(23, 8),
                                sharex=True, sharey=True)


ax1.set_title("original image"); ax1.imshow(img2_rgb_plt)
ax2.set_title("extract sky \n fixed complement_range = 255"); ax2.imshow(w_blue_RBG)
ax3.set_title("extract sky \n fixed complement_range = 0"); ax3.imshow(b_blue_RBG);
mask2 = b_blue_RBG[b_blue_RBG == 0]
w, h, k = img.shape
print("sky_prop = ", mask2.shape[0] / (w*h*k)) 
lower_pink = np.array([160, 120, 30])        
upper_pink = np.array([180, 255, 255])

w_pink_RBG = extract_color_range(path_2, lower_pink, upper_pink, 255)
b_pink_RBG = extract_color_range(path_2, lower_pink, upper_pink)

fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(23, 8),
                                sharex=True, sharey=True)


ax1.set_title("original image"); ax1.imshow(img2_rgb_plt)
ax2.set_title("extract flower \n fixed complement_range = 255"); ax2.imshow(w_pink_RBG)
ax3.set_title("extract flower \n fixed complement_range = 0"); ax3.imshow(b_pink_RBG);
mask3 = b_pink_RBG[b_pink_RBG == 0]
w, h, k = img.shape
print("flower_prop = ", 1 - mask3.shape[0] / (w*h*k)) 
lower_yellow = np.array([14, 70, 70])             
upper_yellow = np.array([30, 255, 255])

w_yellow_RBG = extract_color_range(path_2, lower_yellow, upper_yellow, 255)
b_yellow_RBG = extract_color_range(path_2, lower_yellow, upper_yellow)

fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(23, 8),
                                sharex=True, sharey=True)
ax1.set_title("original image"); ax1.imshow(img2_rgb_plt)
ax2.set_title("extract building \n fixed complement_range = 255"); ax2.imshow(w_yellow_RBG)
ax3.set_title("extract building \n fixed complement_range = 0"); ax3.imshow(b_yellow_RBG);

mask4 = b_yellow_RBG[b_yellow_RBG == 0]
w, h, k = img.shape
print("building_prop = ", mask4.shape[0] / (w*h*k))
path_3 = os.path.join(data_dir, 'girl_face_1.jpg')
img_3 = cv2.imread(path_3)
img3_rgb_plt = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)

lower_yellow = np.array([1, 2, 2])             
upper_yellow = np.array([22, 170, 255])

w_yellow_RBG = extract_color_range(path_3, lower_yellow, upper_yellow, 255)
b_yellow_RBG = extract_color_range(path_3, lower_yellow, upper_yellow)

fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(23, 6),
                                sharex=True, sharey=True)
ax1.set_title("original image"); ax1.imshow(img3_rgb_plt)
ax2.set_title("extract skin&hair \n fixed complement_range = 255"); ax2.imshow(w_yellow_RBG)
ax3.set_title("extract skin&hair \n fixed complement_range = 0"); ax3.imshow(b_yellow_RBG);
from skimage import io
from skimage import color

path_3 = os.path.join(adtn_dir, "monkey.jpg")

# Load picture, convert to grayscale and detect edges
monkey = io.imread(path_3)

## Extract the exactly region contains the monkey
image_rgb = monkey[0:500, 720: 1150]

## convert the image to gray_mode
image_gray = color.rgb2gray(image_rgb)

## Extract exactly the edges from the images
edges = cv2.Canny(monkey[0:500, 720: 1150], 225, 450)

## display
plt.figure(figsize = [20, 10])
plt.subplot(221); plt.imshow(monkey)
plt.subplot(222); plt.imshow(image_rgb)
plt.subplot(223); plt.imshow(image_gray)
plt.subplot(224); plt.imshow(edges)
plt.show()
ret_val, monkey_2 = cv2.threshold(edges, 100, 255, 0)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
im_contours = cv2.drawContours(edges, contours, -1, (255,0,0), -1)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(18, 8),
                                sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Filled color bounded by boundary')
ax2.imshow(im_contours)

plt.show()
mask5 = im_contours[im_contours == 0]
w,h,k = image_rgb.shape
print("prop_monkey_face = ", 1 - mask5.shape[0]/(w*h))