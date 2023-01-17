import numpy as np

import os

import cv2
dirname = "/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise\ Files/Ch02/02_01\ Begin"
fname = "opencv-logo.png"

f_path = os.path.join(dirname,fname)

print(f_path)
img = cv2.imread("/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch02/02_01 Begin/opencv-logo.png")
%matplotlib inline

from matplotlib import pyplot as plt

plt.imshow(img)

plt.show()
cv2.imwrite("/kaggle/working/output.jpg",img)
dirname = "/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch02/02_02 Begin/"
img = cv2.imread(os.path.join(dirname,"opencv-logo.png"),1)

plt.imshow(img)

plt.show()
len(img)
len(img[0])
len(img[0][0])
img.shape
img.dtype
img[10,5]
plt.imshow(img[:,:,1])

plt.show()
img.size
black = np.zeros([150,200,3],'uint8')

plt.imshow(black)

plt.show()
#almost black

ones = np.ones([150,200,3],'uint8')

plt.imshow(ones)

plt.show()
white = np.ones([150,200,3],'uint8')

white *= (2**8-1)

plt.imshow(white)

plt.show()
color = ones.copy()

color[:,:] = (0,0,255)

plt.imshow(color)

plt.show()
color_BGR = cv2.imread("/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch02/02_04 Begin/butterfly.jpg", 1)
plt.imshow(color)

plt.show()
color_RGB = color_BGR[:,:,[2,1,0]].copy()

plt.imshow(color_RGB)

plt.show()
print(color_BGR.shape)
height, width, channels = color_BGR.shape
b,g,r = cv2.split(color_BGR)
rgb_split = np.empty([height, width*3,3],'uint8')
rgb_split[:,0:width] = cv2.merge([b,b,b])

rgb_split[:,width:2*width] = cv2.merge([g,g,g])

rgb_split[:,width*2:width*3] = cv2.merge([r,r,r])
plt.imshow(rgb_split)

plt.show()
cv2.imwrite('/kaggle/working/butterfly_rgb.jpg',rgb_split)
hsv = cv2.cvtColor(color_BGR, cv2.COLOR_BGR2HSV)

h,s,v=cv2.split(hsv)

hsv_split = np.concatenate((h,s,v),axis=1)

plt.imshow(hsv_split)

plt.show()
maindir = "/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch02/"
color = cv2.imread(maindir+"02_05 Begin/butterfly.jpg", 1)
gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
b = color[:,:,0]

g = color[:,:,1]

r = color[:,:,2]
rgba = cv2.merge((r,g,b,g))

plt.imshow(rgba)

plt.show()
image_BGR = cv2.imread(os.path.join(maindir,"02_06 Begin/thresh.jpg"))
image_RGB = cv2.cvtColor(image_BGR,cv2.COLOR_BGR2RGB)
plt.imshow(image_RGB)

plt.show()
blur_RGB = cv2.GaussianBlur(image_RGB, (5,55),0)

plt.imshow(blur_RGB)

plt.show()
kernel = np.ones((5,5),'uint8')

dilate = cv2.dilate(image_RGB, kernel, iterations=1)

erode = cv2.erode(image_RGB,kernel, iterations=1)



f,ax = plt.subplots(1,2)

ax[0].imshow(dilate)

ax[0].set_title("dilate")

ax[1].imshow(erode)

ax[1].set_title("erode")

plt.show()
img = cv2.imread(os.path.join(maindir,"02_07 Begin/players.jpg"))
#scale

img_half = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

img_stretch = cv2.resize(img,(600,600))

img_stretch_near = cv2.resize(img,(600,600),interpolation=cv2.INTER_NEAREST)
f,ax=plt.subplots(1,2,figsize=(10,5))

ax[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

ax[1].imshow(cv2.cvtColor(img_half,cv2.COLOR_BGR2RGB))

plt.show()
f,ax=plt.subplots(1,2,figsize=(10,5))

ax[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

ax[1].imshow(cv2.cvtColor(img_stretch,cv2.COLOR_BGR2RGB))

plt.show()
f,ax=plt.subplots(1,2,figsize=(10,5))

ax[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

ax[1].imshow(cv2.cvtColor(img_stretch_near,cv2.COLOR_BGR2RGB))

plt.show()
#rotation

#(0,0) to rotate about top-left-hand corner

M = cv2.getRotationMatrix2D((0,0), -30, 1)

rotated = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))

plt.imshow(cv2.cvtColor(rotated,cv2.COLOR_BGR2RGB))

plt.show()
maindir = "/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch03/"
bw = cv2.imread(os.path.join(maindir,"03_02 Begin/detect_blob.png"),0)
height, width = bw.shape[0:2]
#plt.imshow(cv2.cvtColor(bw,cv2.COLOR_BGR2RGB))

plt.imshow(bw)

plt.show()
threshold = 85

idxs = bw>threshold
binary = np.zeros([height,width,1],'uint8')

binary[idxs] = 255
plt.imshow(binary[:,:,0])

plt.show()
ret, thresh = cv2.threshold(bw,threshold,255,cv2.THRESH_BINARY)

plt.imshow(thresh)

plt.show()
file_dir = '/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch03/03_03 Begin/'
img = cv2.imread(file_dir+'sudoku.png',0)

plt.imshow(img)

plt.show()
ret, thresh_basic = cv2.threshold(img,70,255,cv2.THRESH_BINARY)

plt.imshow(thresh_basic)

plt.show()
thresh_adapt = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115, 1)

plt.imshow(thresh_adapt)

plt.show()
file_dir = '/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch03/03_04 Begin/'
img = cv2.imread(file_dir + 'faces.jpeg',1)

img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img_RGB)

plt.show()
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

h = hsv[:,:,0]

s = hsv[:,:,1]

v = hsv[:,:,2]
hsv_split = np.concatenate([h,s,v],axis=1)

plt.imshow(hsv_split)

plt.show()
ret, min_sat = cv2.threshold(s,40,255,cv2.THRESH_BINARY)

ret, max_hue = cv2.threshold(h,15,255,cv2.THRESH_BINARY_INV)

final = cv2.bitwise_and(min_sat,max_hue)
f, ax = plt.subplots(1,3,figsize=(30,10))

ax[0].imshow(min_sat)

ax[1].imshow(max_hue)

ax[2].imshow(final)

plt.show()
file_dir = '/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch03/03_06 Begin/'
img = cv2.imread(file_dir+'detect_blob.png',1)#color image

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1)

plt.imshow(thresh)

plt.show()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img2 = img.copy()

index = -1 #draw all the contours

thickness = 4 #thickness of contours

color = (255,0,255)



cv2.drawContours(img2, contours, index, color, thickness)

plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))

plt.show()
objects = np.zeros([img.shape[0],img.shape[1],3],'uint8')

for c in contours:

    cv2.drawContours(objects,[c],-1, color, -1)# first -1 shows all contours in the list, here 1 contour. second -1 says we want to completely fill the contours

    

    area = cv2.contourArea(c)#pixle squared

    perimeter = cv2.arcLength(c,True)

    

    M = cv2.moments(c)#image moment

    cx = int( M['m10']/M['m00'])

    cy = int ( M['m01']/M['m00'])

    cv2.circle(objects, (cx,cy),4, (0,0,255), -1)

    plt.imshow(objects)

plt.show()
file_dir = '/kaggle/input/Ex_Files_OpenCV_Python_Dev/Exercise Files/Ch03/03_08 Begin/'
img = cv2.imread(file_dir+'tomatoes.jpg',1)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

plt.show()
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

res,thresh = cv2.threshold(hsv[:,:,0], 25, 255, cv2.THRESH_BINARY_INV)

plt.imshow(thresh)

plt.show()
edges = cv2.Canny(img,100, 70)

plt.imshow(edges)

plt.show()