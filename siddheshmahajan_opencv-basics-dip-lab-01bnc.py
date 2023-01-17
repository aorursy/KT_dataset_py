import cv2

from matplotlib import pyplot as plt
img = cv2.imread(r'../input/image-for-basic-digital-image-processing-operation/lena.jpg')



img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray)

plt.title('Gray Lena')

plt.imshow(gray, cmap="gray", vmin=0, vmax=255)