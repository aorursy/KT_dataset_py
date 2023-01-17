import cv2

import numpy as np

import random

import matplotlib.pyplot as plt
image = cv2.imread('/kaggle/input/catdataaugmentation/cat.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #conver image to RGB in order to use in matplotlib.

image = cv2.resize(image, (640, 480))

rows, cols = image.shape[:2]

plt.figure(figsize=(15, 15))

plt.axis('off')

plt.imshow(image)

aug_img_H_Flip = cv2.flip(image, 0)  #vertical flipping

aug_img_V_Flip = cv2.flip(image, 1)  #vertical flipping

aug_img_HV_Flip = cv2.flip(image, -1)  #vertical and horizaontal flipping



fig, ax = plt.subplots(nrows=2, ncols=2,sharex=True, figsize=(25, 25))



ax[0][0].set_title("Original Image", fontsize=25); ax[0][0].imshow(image)

ax[0][1].set_title("Vertical and Horizaontal flip", fontsize=25); ax[0][1].imshow(aug_img_HV_Flip)

ax[1][0].set_title("Horizaontal flip", fontsize=25); ax[1][0].imshow(aug_img_H_Flip)

ax[1][1].set_title("Vertical flip", fontsize=25); ax[1][1].imshow(aug_img_V_Flip)

aug_img = cv2.flip(image,random.randint(-1, 1)) 

plt.figure(figsize=(10, 10))

plt.imshow(aug_img)
matrix = np.float32([[1, 0, 1],

                     [0,1, 2]])



tst_img = np.uint16([[10, 20, 30, 40, 50, 60],

                     [70, 80, 90, 100, 110, 120],

                     [130, 140, 150, 160, 170, 180],

                     [140, 150, 160, 170, 180, 190],

                     [200, 210, 220, 230, 240, 250],

                     [260, 270, 280, 290, 300, 310]])



cv2.warpAffine(tst_img,matrix, (6, 6))

tx = random.randint(-.25*cols, .25*cols)

ty = random.randint(-.25*rows, .25*rows)

M = np.float32([[1, 0, tx], [0, 1, ty]])

aug_img = cv2.warpAffine(image, M, (cols, rows))

plt.imshow(aug_img)
x, y = max(tx, 0), max(ty, 0)

w, h = cols - abs(tx), rows - abs(ty)

aug_img = aug_img[y:y+h, x:x+w] 

aug_img = cv2.resize(aug_img, (cols, rows))

plt.imshow(aug_img)
angle = random.randint(0, 180) #angle in degree.

angle_radian = angle*(np.pi)/180 # angle in radian.

Cx, Cy = random.randint(0, 50), random.randint(0, 50)



print(f'Angle is {angle} ||| Cx is {Cx} ||| Cy is {Cy}')



A = (1-np.cos(angle_radian)) * Cx - np.sin(angle_radian)*Cy 

B = np.sin(angle_radian) * Cx + (1-np.cos(angle_radian)) * Cy

mat_1 = np.float32([[np.cos(angle_radian), np.sin(angle_radian), A], 

                    [-np.sin(angle_radian), np.cos(angle_radian), B]])

print("\n our Calculations")

print(mat_1)

print('-'*50)

print("\n openCV Calculations")

mat_2 = cv2.getRotationMatrix2D((Cx, Cy),angle ,1)

print(mat_2)

Cx , Cy = rows, cols #center of rotation

rand_angle = random.randint(-180,180) #random angle range

M = cv2.getRotationMatrix2D((Cy//2, Cx//2),rand_angle ,1) #center angle scale

aug_imgR = cv2.warpAffine(image, M, (cols, rows))  #apply rotation matrix such as previously explained

plt.imshow(aug_imgR)
aug_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # transform to HSV color space .

h, s, v = cv2.split(aug_img) # split each channel in order to add seperate range of values to each channel.

h += np.random.randint(0, 100,size=(rows, cols), dtype=np.uint8 )

s += np.random.randint(0, 20,size=(rows, cols), dtype=np.uint8 )

v += np.random.randint(0, 10,size=(rows, cols) , dtype=np.uint8 )

aug_img = cv2.merge([h,s,v ])

aug_img = cv2.cvtColor(aug_img, cv2.COLOR_HSV2RGB)

plt.imshow(aug_img)
blur_val = random.randint(5,15) #blur value random

aug_img = cv2.blur(image,(blur_val, blur_val))

plt.imshow(aug_img)
#function returns augmented image

def augment_image(original_image):

    rows, cols = original_image.shape[:2]

    #Random flipping

    aug_img_final = cv2.flip(original_image,random.randint(-1, 1)) 

    

    #shifting

    tx = random.randint(-.35*cols, .35*cols)

    ty = random.randint(-.35*rows, .35*rows)

    M = np.float32([[1, 0, tx], [0, 1, ty]])

    aug_img_final = cv2.warpAffine(aug_img_final, M, (cols, rows))  

    

    #cropROI 

    x, y = max(tx, 0), max(ty, 0)

    w, h = cols - abs(tx), rows - abs(ty)

    aug_img_final = aug_img_final[y:y+h, x:x+w] 

    aug_img_final = cv2.resize(aug_img_final, (cols, rows))        

    

    aug_img_final = cv2.cvtColor(aug_img_final, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(aug_img_final)

    h += np.random.randint(0, 40,size=(rows, cols), dtype=np.uint8 )

    s += np.random.randint(0, 10,size=(rows, cols), dtype=np.uint8 )

    v += np.random.randint(0, 10,size=(rows, cols) , dtype=np.uint8 )

    aug_img_final = cv2.merge([h,s,v ])

    aug_img_final = cv2.cvtColor(aug_img_final, cv2.COLOR_HSV2RGB)

    

    blur_val = random.randint(2,7)

    aug_img = cv2.blur(aug_img_final,(blur_val, blur_val))

    

    #rotation

    Cx , Cy = rows, cols

    rand_angle = random.randint(-45,45)

    M = cv2.getRotationMatrix2D((Cy//2, Cx//2),rand_angle ,1)

    aug_img_final = cv2.warpAffine(aug_img_final, M, (cols, rows))

    

    return aug_img_final

    
aug_rows, aug_cols = 3, 3

fig, ax = plt.subplots(nrows=aug_rows, ncols=aug_cols, figsize=(50, 50))

for j in range(aug_rows):

    for k in range(aug_cols) :

        img_augf = augment_image(image)

        ax[j][k].imshow(img_augf)

# plt.imshow(augment_image(image))