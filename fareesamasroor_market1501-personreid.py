!pip install tensorflow==1.8
!pip install Keras==2.1.3
!pip install git+https://github.com/jutanke/pak.git
!conda install -c menpo opencv3 --yes
!pip install git+https://github.com/jutanke/person_reid.git
import cv2

from reid import reid
im1 = cv2.cvtColor(cv2.imread('../input/images01/img001.png'), cv2.COLOR_BGR2RGB)

im2 = cv2.cvtColor(cv2.imread('../input/images01/img002.png'), cv2.COLOR_BGR2RGB)
model = reid.ReId()
import matplotlib.pyplot as plt

from imageio import imread
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(121); ax.axis('off')

ax.imshow(im1)

ax = fig.add_subplot(122); ax.axis('off')

ax.imshow(im2)

plt.show()
score = model.predict(im1, im2)
print(score)