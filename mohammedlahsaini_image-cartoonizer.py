import cv2

import matplotlib.pyplot as plt



originalmage = cv2.imread('/kaggle/input/image-cartoonizer-samples/sample5.jpg')

plt.imshow(cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB))
colorImage = cv2.bilateralFilter(originalmage, 7, 75, 300)

grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)

getEdge = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

plt.imshow(getEdge, cmap='gray')
Image = cv2.medianBlur(originalmage, 9)

cartoonImage = cv2.bitwise_and(Image, Image, mask=getEdge)

plt.imshow(cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2RGB))