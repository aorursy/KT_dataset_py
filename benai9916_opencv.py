import numpy as np
import matplotlib.pyplot as plt
import cv2

# fix the figure size for matplotlin
#plt.rcParams['figure.figsize'] = [10, 8]
# load wrong image wont throw error

img = cv2.imread('djbfjdbf', 1)

print(img)
# load correct image

img = cv2.imread('../input/chicky_512.png', 1)
plt.imshow(img)
plt.show()
# convert to RGB

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print('Image shape : ', img.shape)

plt.imshow(img)
# Load the image in gray scale

img_gray = cv2.imread('../input/chicky_512.png', 0)

print('Image shape: ', img_gray.shape)

# since by default it is not in grayscale we will covert it into gray scale
plt.imshow(img_gray, cmap='gray')
cv2.imwrite('dog.png', img)
resize_img = cv2.resize(img, (600, 400))

plt.imshow(resize_img)
ratio_resize_img = cv2.resize(img, (0,0), img, 0.5, 0.5)

plt.imshow(ratio_resize_img)

print('Image size: ', ratio_resize_img.shape)
# flip the image

flip_img = cv2.flip(img, 0)

plt.imshow(flip_img)
blank_img =  np.zeros(shape=(512, 512, 3))

blank_img.shape
plt.imshow(blank_img)
cv2.rectangle(blank_img, pt1=(400, 300), pt2=(200, 100), color=(0, 255, 0), thickness=(5))

plt.imshow(blank_img)
# print another rectange
cv2.rectangle(blank_img, pt1=(100, 480), pt2=(20, 420), color=(54, 120, 80), thickness=(5))

plt.imshow(blank_img)
# cricle

cv2.circle(blank_img, center=(120, 120), radius=60, color=(255, 0, 0), thickness=5)

plt.imshow(blank_img)
# Fill the circle or rectange by -1

cv2.circle(blank_img, center=(400, 400), radius=60, color=(255, 0, 0), thickness=-1)

plt.imshow(blank_img)
# line

cv2.line(blank_img, pt1=(0, 0), pt2=(440, 340), color=(100, 100, 0), thickness=10)

plt.imshow(blank_img)
# choose font 
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(blank_img, text='Keep learning', org=(0, 380),fontFace=font, 
            fontScale=1.2, color = (0,2,177), thickness=2, lineType=cv2.LINE_4)

plt.imshow(blank_img)
# new blank image
 
new_blank = np.ones(shape=(400, 600, 3), dtype=np.int32)

plt.imshow(new_blank)
verticies =  np.array([ [100, 200], [250, 300], [400, 250], [500, 100] ], dtype=np.int32)

verticies
# it is in two dimension

verticies.shape
# covert it into three dimension

pts = verticies.reshape(-1,1,2)

pts.shape, pts
cv2.polylines(new_blank,[pts], isClosed = True, color = (15, 129, 0), thickness = 3)

plt.imshow(new_blank)
new_vertices = np.array( [ [250,700], [425, 400], [600, 700] ], np.int32)
new_vertices
# reshape
pts_2 = new_vertices.reshape(-1,1,2)

# fillpoly
cv2.fillPoly(new_blank,[pts_2], (0, 0, 255))

plt.imshow(new_blank)
cv2.rectangle(img, (410, 50), (100, 350), (0, 255, 0), 3)

plt.imshow(img)
img2 = cv2.imread('../input/building.jpg')

plt.imshow(img2)
to_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.imshow(to_rgb)
to_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

plt.imshow(to_hsv)
to_hsL = cv2.cvtColor(img2, cv2.COLOR_RGB2HLS)

plt.imshow(to_hsv)
img_1 = cv2.imread('../input/dog_backpack.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

img_2 = cv2.imread('../input/watermark_no_copy.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
plt.imshow(img_1)
plt.imshow(img_2)
# shape of both the image

img_1.shape, img_2.shape
# resize both the image so that we can blend

img_1 = cv2.resize(img_1, (1000, 1000))
img_2 = cv2.resize(img_2, (1000, 1000))


img_1.shape, img_2.shape
blend = cv2.addWeighted(src1 = img_1, alpha = 0.8, src2 = img_2, beta = 0.1, gamma = 20)

plt.imshow(blend)
# load image again
img_1 = cv2.imread('../input/dog_backpack.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

img_2 = cv2.imread('../input/watermark_no_copy.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
# make img_2 very small

img_2 = cv2.resize(img_2, (400, 400))

plt.imshow(img_2)
# assing large image and small image to a variable
large_image = img_1
small_image = img_2
# overy lay small image on the big image

large_image[0: small_image.shape[1], 0: small_image.shape[0]] = small_image

plt.imshow(large_image)
# load image again
img_1 = cv2.imread('../input/dog_backpack.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

img_2 = cv2.imread('../input/watermark_no_copy.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)


# make img_2  small in size
img_2 = cv2.resize(img_2, (400, 400))
# img_1 shape y-axis = 1401, x = 934 

img_1.shape
# x_offset = x of img_1x -  x axis of img_2

x_offset =  934 - 600
y_offset =  1401 - 600
# create an ROI (Region of interest)

rows, cols, channel = img_2.shape
# check rows, columns, shape

rows, cols, channel
# grab the ROI

roi =  img_1[y_offset : 1401, x_offset : 943]

plt.imshow(roi)
# get the gray scale version of the image

img_2gray = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)

plt.imshow(img_2gray, cmap='gray')
# get pure white of mask

mask_inverse = cv2.bitwise_not(img_2gray)

plt.imshow(mask_inverse, cmap='gray')
# but the mask contain no channel

mask_inverse.shape
# covert it to 3 channel

white_bg = np.full(img_2.shape, 255, np.uint8)
bk = cv2.bitwise_or(white_bg, white_bg, mask = mask_inverse)
bk.shape
plt.imshow(bk)
fg = cv2.bitwise_or(img_2, img_2, mask = mask_inverse)


plt.imshow(fg)
# which part of the large image do we want to blend (it is call Region of interest (ROI))

np.asarray(roi)
np.asarray(fg)
 
final_roi = cv2.bitwise_not(fg, roi)
plt.imshow(final_roi)
th = cv2.imread('../input/leuvenA.jpg', 0)

plt.imshow(th, cmap='gray')
ret, thresh1 = cv2.threshold(th, 127, 255, cv2.THRESH_BINARY)


ret, thresh1
plt.imshow(thresh1, cmap='gray')
ret, thresh2 = cv2.threshold(th, 127, 255, cv2.THRESH_BINARY_INV)


plt.imshow(thresh2, cmap='gray')
ret, thresh3 = cv2.threshold(th, 127, 255, cv2.THRESH_TRUNC)

plt.imshow(thresh3, cmap='gray')
ret, thresh4 = cv2.threshold(th, 127, 255, cv2.THRESH_TOZERO)

plt.imshow(thresh4, cmap='gray')
ret, thresh5 = cv2.threshold(th, 127, 255, cv2.THRESH_TOZERO_INV)

plt.imshow(thresh5, cmap='gray')
# function to show big picture

def big_fig(img):
    # make the figure size big
    plt.figure(figsize=(14,14))
    plt.imshow(img, cmap='gray')
    plt.show()
cross_word =  cv2.imread('../input/crossword.jpg', 0)

big_fig(cross_word)
# use binary threshold to make pic either white or black

ret, cross_thresh = cv2.threshold(cross_word, 170, 255, cv2.THRESH_BINARY)

# call function to show image
big_fig(cross_thresh)
th2 =  cv2.adaptiveThreshold(cross_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 8)

big_fig(th2)
# blend cross_thresh and th2

blend_img =  cv2.addWeighted(alpha = 0.2, src1=th2, beta = 0.9, src2 = cross_thresh, gamma = 12)

big_fig(blend_img)
# load image function

def load_image():
    brick_img = cv2.imread('../input/bricks.jpg').astype(np.float32) / 255
    brick_img = cv2.cvtColor(brick_img, cv2.COLOR_BGR2RGB)

    return brick_img

# display image functio
def display_img(img):
    plt.figure(figsize=(12,10))
    plt.imshow(img, cmap='gray')
    plt.show()
# call the functions

loaded_img = load_image()

display_img(loaded_img)
# gamma value
gamma = 0.6

gamma_result1 = np.power(loaded_img, gamma)

display_img(gamma_result1)
# load image
text_img = load_image()

# write text on the image
cv2.putText(text_img, text= 'Strong', org = (20, 600),
            fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 11,
            color = (255, 0, 0), thickness = 4, lineType = cv2.LINE_AA)

display_img(text_img)
# define a kernel ( we are manually choosing value)

# we can play around with the kernel to see different type of blur in the image

kernel = np.ones(shape=(5,5), dtype=np.float32) / 30

kernel
# -1 representing the depth of the output image.

dst = cv2.filter2D(text_img, -1, kernel)

display_img(dst)
# load image again
text_img2 = load_image()

# write text on the image
cv2.putText(text_img2, text= 'Strong', org = (20, 600),
            fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 11,
            color = (255, 0, 0), thickness = 4, lineType = cv2.LINE_AA)

print('reset image')
text_img2_blur = cv2.blur(text_img2, ksize = (5,5))

display_img(text_img2_blur)
# when we increase the kernel size the blurring become more dense

text_img2_blur = cv2.blur(text_img2, ksize = (10,10))

display_img(text_img2_blur)
# load image again
text_img3 = load_image()

# write text on the image
cv2.putText(text_img3, text= 'Strong', org = (20, 600),
            fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 11,
            color = (255, 0, 0), thickness = 4, lineType = cv2.LINE_AA)

print('reset image')
# gaussianblur

text_img3_blur = cv2.GaussianBlur(text_img3, (5,5), 10)

display_img(text_img3_blur)
# load image again
text_img4 = load_image()

# write text on the image
cv2.putText(text_img4, text = 'Strong', org = (20, 600),
            fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 11,
            color = (255, 0, 0), thickness = 4, lineType = cv2.LINE_AA)

print('reset image')
text_img4_blur = cv2.medianBlur(text_img4, 5)

display_img(text_img4_blur)
# median blur in another image

dog_img = cv2.imread('../input/sammy.jpg')
dog_img = cv2.cvtColor(dog_img, cv2.COLOR_BGR2RGB)

display_img(dog_img)
dog_img_blur = cv2.imread('../input/sammy_noise.jpg')

plt.figure(figsize=(15,15))
plt.imshow(dog_img_blur)
# in above image we can see some noise, we will try to remove it by median blur

dog_img_gaussian = cv2.medianBlur(dog_img_blur, 5)

plt.figure(figsize=(15,15))
plt.imshow(dog_img_gaussian)
# load image again
text_img5 = load_image()

# write text on the image
cv2.putText(text_img5, text = 'Strong', org = (20, 600),
            fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 11,
            color = (255, 0, 0), thickness = 4, lineType = cv2.LINE_AA)

print('reset image')
text_img5_blur = cv2.bilateralFilter(text_img5, 9, 75, 75)

display_img(text_img5_blur)
# function to generate white background and generate text

def bg_img_text():
    bg = np.zeros((500, 600))
    
    cv2.putText(bg, text = 'Dope', org=(100, 280), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 5, color= (255, 0, 0), thickness = 30)
    
    return bg

# display image
def bg_load_img(img):
    plt.figure(figsize=(12,10))
    plt.imshow(img, cmap='gray')
    plt.show()
img = bg_img_text()
bg_load_img(img)
# create kernel

kernel = np.ones((5,5), dtype=np.uint8)

kernel
result1 = cv2.erode(img, kernel, iterations=1)

bg_load_img(result1)
# with iteration 4
result2 = cv2.erode(img, kernel, iterations=4)

bg_load_img(result2)
result3 = cv2.dilate(img, kernel, iterations = 2)

bg_load_img(result3)
# create a background noisy image

img = bg_img_text()
white_noise = np.random.randint(0, 2, (500, 600))

white_noise
bg_load_img(white_noise)
white_noise = white_noise * 255
# mix the text image and the noise image

noise_img = white_noise  + img

bg_load_img(noise_img)
# removing noise form the image

opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)

bg_load_img(opening)
# create a foreground noisy image

img = bg_img_text()

# foreground noise
fg_noise = np.random.randint(0, 2, size=(500, 600))

fg_noise
fg_noise = fg_noise * -255

fg_noise
fg_noise_img = img + fg_noise

bg_load_img(fg_noise_img)
fg_noise_img[fg_noise_img == -255] = 0
bg_load_img(fg_noise_img)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

bg_load_img(closing)
img = bg_img_text()

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

bg_load_img(gradient)
img = cv2.imread('../input/sudoku.jpg', 0)

display_img(img)
# verticle line are more visible

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)

display_img(sobelx)
# horizontal line are more visible
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

display_img(sobely)
# capture both horizontal and vetical line

laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
display_img(laplacian)
blend =  cv2.addWeighted(alpha = 0.5, src1=sobelx,gamma = 0.5, src2 = sobely, beta =5)

display_img(blend)
# threshold

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

display_img(th1)
# morpohological operator

kernel =  np.ones((4,4), dtype=np.uint8)

mo = cv2.morphologyEx(blend, cv2.MORPH_GRADIENT, kernel)

display_img(mo)
# load three images
horse = cv2.imread('../input/horse.jpg')
show_horse = cv2.cvtColor(horse, cv2.COLOR_BGR2RGB)

raimbow = cv2.imread('../input/rainbow.jpg')
show_raimbow = cv2.cvtColor(raimbow, cv2.COLOR_BGR2RGB)

bricks = cv2.imread('../input/bricks.jpg')
show_bricks = cv2.cvtColor(bricks, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(2,3, figsize=(16,10))
ax[0,0].imshow(horse)
ax[0,1].imshow(raimbow)
ax[0,2].imshow(bricks)
ax[1,0].imshow(show_horse)
ax[1,1].imshow(show_raimbow)
ax[1,2].imshow(show_bricks)
plt.show()
hist_values = cv2.calcHist(bricks, channels = [0], mask=None, histSize = [256], ranges = [0, 256])

hist_values.shape 
plt.plot(hist_values)
hist_horse = cv2.calcHist(horse, channels = [0], mask =None, histSize = [256], ranges = [0,256])

plt.plot(hist_horse)
img = horse

colors = ('b','g','r')

for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256],[0,256])
    plt.plot(hist, color=col)
# raimbow image
img = raimbow

# create mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

plt.imshow(mask, cmap='gray')
# in the above mask select some portion out of it

mask[200: 300, 100: 380] = 255

plt.imshow(mask, cmap='gray')
masked_img = cv2.bitwise_and(img, img, mask = mask)
show_masked_img = cv2.bitwise_and(show_raimbow, show_raimbow, mask= mask)
plt.imshow(show_masked_img)
hist_mask_value_red = cv2.calcHist([raimbow], channels=[2],mask = mask, histSize=[256], ranges=[0,256])

hist_value_red = cv2.calcHist([raimbow], channels=[2],mask = None, histSize=[256], ranges=[0,256])
plt.plot(hist_mask_value_red, label='with mask RED value')
plt.plot(hist_value_red, label ='without mask RED value')
plt.legend()
# new image 
gorilla = cv2.imread('../input/gorilla.jpg', 0)

display_img(gorilla)
hist_values = cv2.calcHist([gorilla], [0], None, [256], [0,256])

plt.plot(hist_values)
eq_hist = cv2.equalizeHist(gorilla)

display_img(eq_hist)
eq_hist_values = cv2.calcHist([eq_hist], [0], None, [256], [0,256])

plt.plot(eq_hist_values)
color_gorilla = cv2.imread('../input/gorilla.jpg')

show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)

plt.imshow(show_gorilla)
# to equalize the histogram of a color image/ to increase the contrast of an image 
# translate the image to HSV 

hsv =  cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)
# grab the value channel
hsv[:,:,2].max(), hsv[:,:,2].min()
hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
# covert HSV to  RGB

eq_color_gorilla = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

plt.imshow(eq_color_gorilla)
# load the full image
full_img = cv2.imread('../input/sammy.jpg')

full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

plt.imshow(full_img)
# load the subset of image to match
face_img = cv2.imread('../input/sammy_face.jpg')

face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

plt.imshow(face_img)
# shape of both the image

full_img.shape, face_img.shape
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for m in methods:
    
    # create a copy of image
    full_img_cpy = full_img.copy()
    
    method = eval(m)
    
    # TEMPLATE MATCHNG
    res = cv2.matchTemplate(full_img_cpy, face_img, method)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    
    else:
        top_left = max_loc
        
    height, width, channel = face_img.shape
    
    bottom_right = (top_left[0] + width, top_left[1] + height)
    
    # draw rectangle on the detected area
    cv2.rectangle(full_img_cpy, top_left, bottom_right, (0, 255, 0), 8)
    
    # plot and show the image
    plt.subplot(121)
    plt.imshow(res)
    plt.title('Matching Result')

    plt.subplot(122),
    plt.imshow(full_img_cpy)
    plt.title('Detected Point')
    plt.suptitle(m)
    plt.show()
# read image

flat_chess = cv2.imread('../input/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)

plt.imshow(flat_chess)
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_RGB2GRAY)


plt.imshow(gray_flat_chess, cmap='gray')
real_chess =  cv2.imread('../input/real_chessboard.jpg')

real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)

plt.imshow(real_chess)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_RGB2GRAY)


plt.imshow(gray_real_chess, cmap='gray')
# covert it to float
gray = np.float32(gray_flat_chess)

# harris corner
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
# just to show the corner we use dialat 

# dilate
dst = cv2.dilate(dst, None)

flat_chess[dst > 0.01 * dst.max()] = [255,0,0]

plt.imshow(flat_chess)
gray2 = np.float32(gray_real_chess)

dst2 = cv2.cornerHarris(gray2, 2, 3, 0.04)
dst2 = cv2.dilate(dst2, None)

real_chess[dst2 > 0.01  * dst2.max()] = [255, 0, 0]

plt.imshow(real_chess)
# Load image 
flat_chess = cv2.imread('../input/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_RGB2GRAY)

real_chess =  cv2.imread('../input/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(gray_flat_chess, 70, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x,y), 3, (255, 0, 0), -1)
    
    
# show image
plt.imshow(flat_chess)
# detect corner in the real chess image

corners2 = cv2.goodFeaturesToTrack(gray_real_chess, 70, 0.01, 10)

corners2 = np.int0(corners2)

for i in corners2:
    x, y = i.ravel()
    cv2.circle(real_chess, (x,y), 3, (255, 0, 0), -1)
    
    
# show image
plt.imshow(real_chess)
# load image

img = cv2.imread('../input/sammy_face.jpg')

plt.imshow(img)
# canny edge detector
edge = cv2.Canny(image = img, threshold1 = 127, threshold2 = 127)

plt.imshow(edge)
edge = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)

plt.imshow(edge)
# find the best threshold value

med_val = np.median(img)

display(med_val)

# choose lower threshold value to either 0 or 70% of median value, whichever is greater
lower = int(max(0, 0.7 * med_val))

# upper threshold  to either 130% of the median or 255, which ever is small
upper = int(min(255, 1.3 * med_val))
edge = cv2.Canny(image = img, threshold1 = lower, threshold2 = upper)

plt.imshow(edge)
# Blur and the  apply canny edge detection

blurred_img = cv2.blur(img, (4,4))

edge = cv2.Canny(image = blurred_img, threshold1 = 127, threshold2 = 157)

plt.imshow(edge)
flat_chess = cv2.imread('../input/flat_chessboard.png')

plt.imshow(flat_chess)
# findChessBoardCorners is specifically work with chess board type image

found, corners = cv2.findChessboardCorners(flat_chess, (7,7))

found
cv2.drawChessboardCorners(flat_chess, (7,7), corners, found)

plt.imshow(flat_chess)
dots = cv2.imread('../input/dot_grid.png')

plt.imshow(dots)
found, corners = cv2.findCirclesGrid(dots, (10,10), cv2.CALIB_CB_SYMMETRIC_GRID)

found
cv2.drawChessboardCorners(dots,(10, 10), corners, found)

plt.imshow(dots)
img = cv2.imread('../input/internal_external.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

plt.imshow(img, cmap='gray')
img.shape
# cv2.RETR_CCOMP extract both interal and external contour
contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
len(contours), type(hierarchy)
# external_contour = np.zeros(img.shape)

# external_contour.shape

# for i in range(len(contours)):
#     if hierarchy[0][i][3] == -1:
#         cv2.drawContours(external_contour, contours, i, 255, -1)

#  plt.imshow(external_contour)
cv2.drawContours(img, contours, -1, (255,0,0), 8)

plt.imshow(img, cmap='gray')
internel_contour = np.zeros(img.shape)

internel_contour.shape

for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(internel_contour, contours, i, 255, -1)

plt.imshow(internel_contour)
reeses = cv2.imread('../input/reeses_puffs.png', 0)

display_img(reeses)
cereals = cv2.imread('../input/many_cereals.jpg', 0)

display_img(cereals)
# create orb instance
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(reeses, None)
kp2, des2 = orb.detectAndCompute(cereals, None)
# Create a matching object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
reeses_match = cv2.drawMatches(reeses,kp1,cereals,kp2,matches[:20],None, flags=2)

display_img(reeses_match)
sift = cv2.SIFT_create()

kp1, des1 = orb.detectAndCompute(reeses, None)
kp2, des2 = orb.detectAndCompute(cereals, None)
bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

# ration test to check if the first match and the second match are close to each other or not
# less distance better match
good = []

# for match1, match2 in matches:
#     # if match 1 distance is less then 75% of match 2 distance
#     # if descriptor is a good match then apedn it to good 
#     if match1.distance < 0.75 * match2.distance:
#         good.append(match1)
len(good), len(matches)
sift_matchs = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, matches, None, flags=2)

display_img(sift_matchs)
sift = cv2.SIFT_create()

kp1, des1 = orb.detectAndCompute(reeses, None)
kp2, des2 = orb.detectAndCompute(cereals, None)
des1
# FLANN 
FLANN_INDEX_KDTREE = 0
index_parms = dict(algorithm=FLANN_INDEX_KDTREE, tree=5) 
search_parms = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_parms, search_parms)

matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
good = []

for match1, match2 in matches:
    if match1.distance < 0.7 * match2.distance:
        good.append(match1)
flann_matchs = cv2.drawMatches(reeses, kp1, cereals, kp2, good, None, flags=0)

display_img(flann_matchs)
sep_coins = cv2.imread('../input/pennies.jpg')

# call function to show image
display_img(sep_coins)
sep_blur = cv2.medianBlur(sep_coins, 25)

display_img(sep_blur)
gray_sep_coin = cv2.cvtColor(sep_coins, cv2.COLOR_BGR2GRAY)

display_img(gray_sep_coin)
ret, sep_thresh = cv2.threshold(gray_sep_coin, 180, 255, cv2.THRESH_BINARY_INV)

display_img(sep_thresh)
contours, hierarchy = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )

cv2.drawContours(sep_coins, contours, -1, (255, 0 , 0), 10)

display_img(sep_coins)
img = cv2.imread('../input/pennies.jpg')

img = cv2.medianBlur(img, 35)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply otsu method in threshold 
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

display_img(thresh)
# noise removal option

kernel = np.ones((3,3), dtype=np.uint8)

# good way to reduce noise
opening = cv2.morphologyEx(thresh , cv2.MORPH_OPEN, kernel, iterations=2)

display_img(opening)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# DISTANCE TRANSFORM to seperate the coins
# close to black will fade way, and closer to white will become brighter

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region

sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
display_img(markers)
markers = cv2.watershed(img,markers)

img[markers == -1] = [255,0,0]

display_img(markers)
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )

cv2.drawContours(img, contours, -1, (255, 0 , 0), 10)

display_img(img)
nadia = cv2.imread('../input/Nadia_Murad.jpg', 0)
denis = cv2.imread('../input/Denis_Mukwege.jpg', 0)
solvay = cv2.imread('../input/solvay_conference.jpg', 0)
plt.imshow(nadia, cmap='gray')
plt.imshow(denis, cmap='gray')
face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')
def adj_detect_face(cascade_type, img):
    
    face_img = img.copy()
    
    face_rects = cascade_type.detectMultiScale(face_img, scaleFactor =1.2, minNeighbors = 5)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255, 0, 0), 10)
        
    return face_img
result = adj_detect_face(face_cascade, nadia)

plt.imshow(result, cmap='gray')
eye_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_eye.xml')

result = adj_detect_face(eye_cascade, nadia)

plt.imshow(result, cmap='gray')
result = adj_detect_face(eye_cascade, denis)

plt.imshow(result, cmap='gray')

# In this case we are not able to dtect the eys because the color of the eyes is not learly visible
