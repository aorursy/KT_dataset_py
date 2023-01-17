import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import os
image = cv2.imread('../input/opencv-1/IMG_1747.jpg')
print(image.shape)
image
cv2.imshow('IMAGE',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('IMAGE',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image = cv2.resize(image,(500,500))
cv2.imshow('IMAGE',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('GRAY.png',image)
image = plt.imread('../input/opencv-1/IMG_1747.jpg')
print(image.shape)
image
plt.imshow(image)
image = np.resize(image,(500,500))
plt.imshow(image)
plt.imsave('MATPLOTLIB_OUTPUT.PNG',image)
plt.subplots(nrows = 1,ncols = 3,figsize = (15,5))
for i in [-1,0,1]:
    plt.subplot(1,3,i+2)
    image = cv2.imread('../input/opencv-1/IMG_1747.jpg',i)
    plt.title('IMAGE SHAPE: {}'.format(np.shape(image)))
    plt.imshow(image)
for i,j in zip(['A','B','C'],[-1,0,1]):
    image = plt.imread('../input/opencv-1/IMG_1747.jpg')
    image = cv2.resize(image,(420,420))
    cv2.imshow(i,image)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_1 = np.zeros((250,500,3),np.uint8)
img_1 = cv2.rectangle(img_1,(200,0),(300,100),(255,255,255),-1)
img_2 = cv2.imread('../input/opencv-1/IMG_1747.jpg')
img_2 = cv2.resize(img_2,(500,250))

cv2.imshow('RECT',img_1)
cv2.imshow('ROAD',img_2)

cv2.waitKey(0)
cv2.destroyAllWindows()
bitAnd = cv2.bitwise_and(img_2,img_1)

cv2.imshow('RECT',img_1)
cv2.imshow('ROAD',img_2)
cv2.imshow('bitAND',bitAnd)
cv2.waitKey(0)
cv2.destroyAllWindows()
bitOr = cv2.bitwise_or(img_2,img_1)

cv2.imshow('RECT',img_1)
cv2.imshow('ROAD',img_2)
cv2.imshow('bitOR',bitOr)
cv2.waitKey(0)
cv2.destroyAllWindows()
bitXor = cv2.bitwise_xor(img_2,img_1)

cv2.imshow('RECT',img_1)
cv2.imshow('ROAD',img_2)
cv2.imshow('bitXOR',bitXor)
cv2.waitKey(0)
cv2.destroyAllWindows()
bitNot_1 = cv2.bitwise_not(img_1)
bitNot_2 = cv2.bitwise_not(img_2)

cv2.imshow('RECT',img_1)
cv2.imshow('ROAD',img_2)
cv2.imshow('NOT_RECT',bitNot_1)
cv2.imshow('NOT_ROAD',bitNot_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()
    cv2.imshow('FRAME',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)

while(True):
    ret ,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('FRAME',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture('../input/opencv-1/box.mp4')

while(cap.isOpened()):
    ret ,frame = cap.read()
    if ret:
        cv2.imshow('FRAME',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture('../input/opencv-1/box.mp4')

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret:
        colour = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow('FRAME',colour)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()