# import cv2

# import matplotlib.pyplot as plt

import glob

import os

from tqdm import tqdm

# %matplotlib inline
print(os.listdir("../input"))
import cv2

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
full = cv2.imread('../input/CV Object Detecton Dataset/gorilla.jpg')#LOADING THE IMAGE of SOURCE IMAGE

full = cv2.cvtColor(full,cv2.COLOR_BGR2RGB)#CHANGING THE COLOR SPACE FROM BGR TO RGB



part = cv2.imread('../input/CV Object Detecton Dataset/gorilla_shorted.jpg')#LOADING THE IMAGE OF TEMPLATE IMAGE

part = cv2.cvtColor(part,cv2.COLOR_BGR2RGB)#CHANGING THE COLOR SPACE FROM BGR TO RGB
full.shape[::-1]
plt.imshow(full)#SOURCE IMAGE
plt.imshow(part)#TEMPLATE IMAGE
'''



DIFFERENT METHODS AVAILABLE IN OPENCV 1ST IS COEFFICIENT BASED 3RD IS CORRELATION BASED 

5TH IS DIFFERENCE BASED AND 2ND,4TH AND 6TH ARE NORMALISED VERSION OF MENTIONED METHODS.



AS, 3RD METHOD IS DIFFERENCE BASED SO IT WILL GIVE THE REGION WHERE IT 

WILL FIND THE DIFFERENCE BETWEEN THE SOURCE AND TEMPLATE IS LESS.



SO,IT IS DIFFERENT THAN OTHE METHODS  



'''



methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
for i in methods:

    func = eval(i)# eval() makes any string to a function if it is present in th the module

    full_img = full.copy()

    

    res = cv2.matchTemplate(full_img,part,func)# THIS IS THE FUNCTION THAT WE USE TO DO THAT

    '''

    cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) the first parameter is the mainimage, 

    second parameter is the template to be matched and third parameter is the method used for matching.

    

    '''

    

    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)#OUTPUT OF 'cv2.matchTemplate()' CONTAINS THE MIN,MAX VALUE AND THEIR LOCATION

    

    if i in ['cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']:

        '''

        AS IT IS A DIFFERENCE BASED METHOD

        '''

        top_left = min_loc

        height,width,channel = part.shape

        bot_right = (top_left[0]+width,top_left[1]+height)

    elif i in ['cv2.TM_CCORR']:

        '''

        IT WAS NOT SHOWING THE DESIRED OUTPUT THAT'S WHY I CUSTOMISED THAT IT IDENTIFY THE TEMPLATE IN SOURCE

        '''

        top_left = tuple(map(lambda i, j: i - j, min_loc, (-500,300)))

        height,width,channel = part.shape

        bot_right = (top_left[0]+width+200,top_left[1]+height+100)

    else:

        top_left = max_loc

        height,width,channel = part.shape

        bot_right = (top_left[0]+width,top_left[1]+height)

        

#     height,width,channel = part.shape

#     bot_right = (top_left[0]+width,top_left[1]+height)

    cv2.rectangle(full_img,top_left,bot_right,(255,0,0),25)#DRAWING RECTANGLE ON THE INDENTIFIED PORTION

    font = cv2.FONT_HERSHEY_SIMPLEX#PUTTING TEXT TO SHOW WHICH METHOD IS USED

    cv2.putText(full_img,text=i,org = (1400,700),fontFace=font,fontScale = 7,color = (255,0,0),thickness=25,lineType = cv2.LINE_AA)

    

    '''

    SHOWING THE OUTPUT👍

    

    '''

    plt.figure(figsize=(12,10))

    plt.subplot(121)

    plt.imshow(res)

    plt.title('heatmap')

    

    plt.subplot(122)

    plt.imshow(full_img)

    plt.title('detection')

    

#     plt.suptitle(i)

    plt.show()
full = cv2.imread('../input/CV Object Detecton Dataset/gorilla.jpg')

full = cv2.cvtColor(full,cv2.COLOR_BGR2RGB)



part = cv2.imread('../input/CV Object Detecton Dataset/gorilla_shorted.jpg')

part = cv2.cvtColor(part,cv2.COLOR_BGR2RGB)
full_img = full.copy()



res = cv2.matchTemplate(full_img,part,cv2.TM_CCORR)

min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

top_left = tuple(map(lambda i, j: i - j, min_loc, (-500,300)))

height,width,channel = part.shape

bot_right = (top_left[0]+width+200,top_left[1]+height+100)

cv2.rectangle(full_img,top_left,bot_right,(255,0,0),25)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(full_img,text='TM_CCORR',org = (3000,500),fontFace=font,fontScale = 10,color = (255,0,0),thickness=25,lineType = cv2.LINE_AA)



plt.figure(figsize=(12,10))

plt.subplot(121)

plt.imshow(res)

plt.title('heatmap')



# plt.figure(figsize=(8,8))   

plt.subplot(122)

plt.imshow(full_img)

plt.title('detection')

    

# plt.suptitle('TM_CCOEFF')

plt.show()
img_rgb = cv2.imread('../input/CV Object Detecton Dataset/source1.jfif')

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('../input/CV Object Detecton Dataset/template1.jfif',0)

w, h = template.shape[::-1]



res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

threshold = 0.42# THIS IS THE THRESHOLD FOR WHICH WE GOT EVERY COIN WHICH MATCHES THE TEMPLATE

loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):

    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 10)



plt.imshow(img_rgb)
template1 = cv2.imread('../input/CV Object Detecton Dataset/template1.jfif')

sourcee = cv2.imread('../input/CV Object Detecton Dataset/source1.jfif')

sourcee.shape
template1 = cv2.resize(template1,(1600,1200))

template1.shape


multiple_img = np.concatenate((img_rgb, template1), axis=1)

sourcee = cv2.resize(sourcee,(3200,1200))

multiple_img1 = np.concatenate((multiple_img,sourcee),axis=1)

# concat = np.concatenate((img_rgb,template),axis=1)

plt.figure(figsize=(12,14))

plt.imshow(multiple_img1)

plt.show()
import cv2

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
#STEP1: LOADING IMAGES

ches = cv2.imread('../input/CV Object Detecton Dataset/chess1.png')

ches = cv2.cvtColor(ches,cv2.COLOR_BGR2RGB)



wing = cv2.imread('../input/CV Object Detecton Dataset/wing_2.jfif')

wing = cv2.cvtColor(wing,cv2.COLOR_BGR2RGB)



plt.imshow(ches)
gray = cv2.cvtColor(ches,cv2.COLOR_RGB2GRAY)#STEP2: CONVERTING TO GRAY SCALE
plt.imshow(gray,cmap='gray')
gray.shape
gray_img = np.float32(gray)#STEP3: IT IS A IMPORTANT STEP WHERE WE ARE CONVERTING THE GRAY SCALED IMAGE IN FLOATING TYPE

dst = cv2.cornerHarris(gray_img,2,3,0.04)

'''

STEP4: USING CORNERHARRIS ALGORITHUM 

1ST ARGUMENT: Input Image,

2ND ARGUMENT: Neighborhood size,

3RD ARGUMENT: Aperture parameter for the Sobel() operator

4TH ARGUMENT: VALUE OF 'k'



YOU CAN PLAY AROUND THE 2ND TO 3RD ARGUMENTS TO GET YOUR DESIRED OUTPUT

'''
dst = cv2.dilate(dst,None)#YOU HAVE TO DILATE THE OUTPUT OF cv2.cornerHarris() TO VISUALIZE
# THRESHOLD IS APPLIED WHICH IS 1% OF THE MAX VALUE OF dst, THOSE POINTS

#WHICH ARE GREATER THAN THAT IS ASSIGNED THE VALUE RED, SO WE CAN VIEW THE POINTS

ches[dst>0.01*dst.max()] = [255,0,0]
plt.imshow(ches)
plt.imshow(wing)
wing_gray = cv2.cvtColor(wing,cv2.COLOR_RGB2GRAY)
wing_gray.shape
plt.imshow(wing_gray,cmap = 'gray')
wing_gray_img = np.float32(wing_gray)
dst1 = cv2.cornerHarris(src=wing_gray_img,blockSize = 2,ksize=3,k=0.04)
dst1 = cv2.dilate(dst1,None)
wing[dst1>0.01*dst1.max()] = [255,0,0]
plt.imshow(wing)
#LOADING THE IMAGES

ches = cv2.imread('../input/CV Object Detecton Dataset/chess1.png')

ches = cv2.cvtColor(ches,cv2.COLOR_BGR2RGB)



wing = cv2.imread('../input/CV Object Detecton Dataset/wing_2.jfif')

wing = cv2.cvtColor(wing,cv2.COLOR_BGR2RGB)



gray = cv2.cvtColor(ches,cv2.COLOR_RGB2GRAY)

wing_gray = cv2.cvtColor(wing,cv2.COLOR_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(gray,64,0.01,10)# APPLING THE ALGORITHM WICH IS NAMED BY THE NAME OF THE PAPER ITSELF
corners = np.int0(corners)
#EXTRACTING THE POINTS 

for i in corners:

    x,y = i.ravel()

    cv2.circle(ches,(x,y),3,(255,0,0),-1)# DRAWING CIRCLE USING THOSE POINTS
plt.imshow(ches)
corners1 = cv2.goodFeaturesToTrack(wing_gray,90,0.01,10)
corners1 = np.int0(corners1)
for m in corners1:

    x1,y1 = m.ravel()

    cv2.circle(wing,(x1,y1),3,(0,255,0),-1)
plt.imshow(wing)
#LOADING AND SHOWING IMAGE

edge_img = cv2.imread('../input/CV Object Detecton Dataset/chess1.png')

plt.imshow(edge_img)
edge = cv2.Canny(edge_img,0,255)#APPLYING THE ALGORITHM
plt.imshow(edge)#SHOWING THE IMAGE
med_val = np.median(edge_img)
upper = int(min(255,1.3*med_val))#ABOVE MENTIONED FORMULA FOR CLACULATING UPPER THRESHOLD

lower = int(min(0,0.7*med_val))#ABOVE MENTIONED FORMULA FOR CLACULATING LOWER THRESHOLD
blurred_img = cv2.blur(edge_img,ksize = (5,5))#BLURING THE IMAGE
edge = cv2.Canny(blurred_img,lower,upper+60)# APPLYING THE ALGO
plt.imshow(edge)#WE HAVE NOT GOT THE DESIRED OUTPUT BECAUSE THERE IS NO NOISE IN THE IMAGE.
edge = cv2.Canny(edge_img,lower,upper+450)
plt.imshow(edge)#BETTER OUTPUT
edge_img1 = cv2.imread('../input/CV Object Detecton Dataset/SharedScreenshot1.jpg')

plt.imshow(edge_img1)
med_val1 = np.median(edge_img1)
upper1 = int(min(255,1.3*med_val1))

lower1 = int(min(0,0.7*med_val1))
blurred_img1 = cv2.blur(edge_img1,ksize = (5,5))
edge1 = cv2.Canny(blurred_img1,lower1,upper1+60)
plt.imshow(edge1)# CAN BE TUNED
edge_not_blur = cv2.Canny(edge_img1,upper1+130,lower1+60)
plt.imshow(edge_not_blur)#ALMOST DESIRED OUTPUT
lower1
# grid1 = cv2.imread('left14.jpg')

# gray1 = cv2.cvtColor(grid1,cv2.COLOR_BGR2GRAY)



# objp = np.zeros((6*7,3), np.float32)

# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)



# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)



# objpoints1 = []

# imgpoints1 = []

# ret1,cor1 = cv2.findChessboardCorners(grid1,(7,7))

# if ret1 == True:

#         objpoints1.append(objp)



#         corners2 = cv2.cornerSubPix(gray1,cor1,(11,11),(-1,-1),criteria)

#         imgpoints1.append(corners2)



#         # Draw and display the corners

#         grid_copy1 = grid1.copy()

#         grid_copy1 = cv2.drawChessboardCorners(grid_copy1, (7,6), corners2,ret1)

# # grid_copy = grid.copy()

# # cv2.drawChessboardCorners(grid_copy,(7,7),cor,found)

#         plt.imshow(grid_copy1)

import cv2

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
grid = cv2.imread('../input/CV Object Detecton Dataset/flat_board.png')#LOADING IMAGE
type(grid)
plt.imshow(grid)
found,cor = cv2.findChessboardCorners(grid,(7,7))#APPLYING THE ALGO
found
grid_copy = grid.copy()

cv2.drawChessboardCorners(grid_copy,(7,7),cor,found)#DRAWING THE CORNERS ON THE COPY OF THE IMAGE
plt.imshow(grid_copy)#SHOWING THE IMAGE
dot = cv2.imread('../input/CV Object Detecton Dataset/dot_grid.png')#loading the image
type(dot)
plt.imshow(dot)#showing the image
found1,cor1 = cv2.findCirclesGrid(dot,(10,10),cv2.CALIB_CB_ASYMMETRIC_GRID)#APPLYING THE ALGO AS MENTIONED ABOVE
found1
cv2.drawChessboardCorners(dot,(10,10),cor1,found1)# DRAWING CORNERS
plt.imshow(dot)#SHOWING THE OUTPUT
import cv2

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
image = cv2.imread("../input/CV Object Detecton Dataset/leaf.jfif")

type(image)
# convert to RGB

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# convert to grayscale

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# create a binary thresholded image

_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# show it

plt.imshow(binary, cmap="gray")

plt.show()
# find the contours from the thresholded image

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw all contours

image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# show the image with the drawn contours

plt.imshow(image)

plt.show()
#LOADING IMAGE

furier = cv2.imread('../input/CV Object Detecton Dataset/a_0.png',cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(furier)#MAIN STEP1 PASSING THE IMAGE TO fast Fourier Transform function (FFT)

fshift = np.fft.fftshift(f)# MAIN STEP2 PASSING THE OUTPUT OF FFT IN np.fft.fftshift()

magnitude_spectrum = 20*np.log(np.abs(fshift))#MAIN STEP3, THESE THREE STEPS WILL BE SAME FOR EVERYTIME

magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

img_and_magnitude = np.concatenate((furier, magnitude_spectrum), axis=1)
#SHOWING THE OUTPUT

plt.figure(figsize=(12,8))

plt.imshow(img_and_magnitude,cmap='gray')

plt.show()
furier = cv2.imread('../input/CV Object Detecton Dataset/a_1.png',cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(furier)

fshift = np.fft.fftshift(f)

magnitude_spectrum = 20*np.log(np.abs(fshift))

magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

img_and_magnitude = np.concatenate((furier, magnitude_spectrum), axis=1)



plt.figure(figsize=(12,8))

plt.imshow(img_and_magnitude,cmap='gray')

plt.show()
furier = cv2.imread('../input/CV Object Detecton Dataset/c_0.png',cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(furier)

fshift = np.fft.fftshift(f)

magnitude_spectrum = 20*np.log(np.abs(fshift))

magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

img_and_magnitude = np.concatenate((furier, magnitude_spectrum), axis=1)



plt.figure(figsize=(12,8))

plt.imshow(img_and_magnitude,cmap='gray')

plt.show()
furier = cv2.imread('../input/CV Object Detecton Dataset/c_1.png',cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(furier)#MAIN STEP1 PASSING THE IMAGE TO fast Fourier Transform function (FFT)

fshift = np.fft.fftshift(f)# MAIN STEP2 PASSING THE OUTPUT OF FFT IN np.fft.fftshift()

magnitude_spectrum = 20*np.log(np.abs(fshift))#MAIN STEP3, THESE THREE STEPS WILL BE SAME FOR EVERYTIME

magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

img_and_magnitude = np.concatenate((furier, magnitude_spectrum), axis=1)



#SHOWING OUTPUT

plt.figure(figsize=(12,8))

plt.imshow(img_and_magnitude,cmap='gray')

plt.show()
from IPython.display import YouTubeVideo

YouTubeVideo('nVbaNcRldmw')

# https://youtu.be/nVbaNcRldmw
#LOADING THE IMAGE



nadia = cv2.imread('../input/CV Object Detecton Dataset/Nadia.jpg')

face_color1 = cv2.cvtColor(nadia,cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(nadia,cv2.COLOR_BGR2GRAY)

# blurrr = cv2.GaussianBlur(gray,(35,35),0)

plt.imshow(face_color1)

# Read in the cascade classifiers for face and eyes 



dog_face = cv2.CascadeClassifier('../input/CV Object Detecton Dataset/haarcascade_frontalface_default.xml')

eye = cv2.CascadeClassifier('../input/CV Object Detecton Dataset/haarcascade_eye.xml')
# create a function to detect face AND DRAWING RECTANGLE AROUND THE FACE



def detect_face(img):

    face_img = img.copy()

    face_gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)

    face_rects = dog_face.detectMultiScale(face_img)

    

    for(x,y,w,h) in face_rects:

        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)

        roi_gray = face_gray[y:y+h,x:x+w]

        roi_color = face_img[y:y+h,x:x+w]

        eye_rects = eye.detectMultiScale(roi_gray)

        for(x1,y1,w1,h1) in eye_rects:

            cv2.rectangle(roi_color,(x1,y1),(x1+w1,y1+h1),(255,255,255),10)

        

        return face_img
re = detect_face(face_color1)
#SHOWING THE OUTPUT

plt.figure(figsize=(12,8))

plt.imshow(re)

plt.show()
#LOADING THE IMAGE

face = cv2.imread('../input/CV Object Detecton Dataset/solvay.jpg')

fece_color = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

# blurrr = cv2.GaussianBlur(gray,(35,35),0)

# plt.imshow(gray,cmap='gray')

# Read in the cascade classifiers for face and eyes 



face_cascade = cv2.CascadeClassifier('../input/CV Object Detecton Dataset/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('../input/CV Object Detecton Dataset/haarcascade_eye.xml')
face1 = face_cascade.detectMultiScale(gray1,1.3,5)#FACE COORDINATES ARE CALCULATED USING "face_cascade.detectMultiScale()"

#DRAWING THE RECTANGLE FOR FACE AND EYES

for (x,y,w,h) in face1:

    cv2.rectangle(fece_color,(x,y),(x+w,y+h),(255,0,0),10)

    roi_gray1 = gray1[y:y+h,x:x+w]

    roi_color1 = fece_color[y:y+h,x:x+w]

    eyes1 = eye_cascade.detectMultiScale(roi_gray1)

    for (ex,ey,ew,eh) in eyes1:

        cv2.rectangle(roi_color1,(ex,ey),(ex+ew,ey+eh),(0,255,0),10)
#SHOWING THE OUTPUT

plt.figure(figsize=(12,8))

plt.imshow(fece_color)

plt.show()
print('Faces Found:{}'.format(len(face1)))# NUMBER OF FACES FOUND