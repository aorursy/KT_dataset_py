import cv2



import numpy as np

import matplotlib.pyplot as plt



import glob,os
path = '../input/'

print(os.listdir(path))
def load_data(path,ext='.png'):

  files = glob.glob(path+'*'+ext)

  data = []

  for file in files:

    img = cv2.imread(file)

    data.append(img)

  return data



data = load_data(path,ext='.png')

data = np.array(data)

plt.imshow(data[0])

plt.axis('off')

plt.show()
def adjust_gamma(image, gamma=1.0):

  table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)])

  return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
k = './augmented_data/'

flag=0

if(os.path.isdir(k)):

  print('path '+k+' is already existed.')

  aug_path = k

else:

  print('path '+k+' is created.')

  os.mkdir(k)

  aug_path = k

  flag=1

  

if flag:

  print('Data Augmentation is being processed...')

  for i in range(len(data)):

    r = np.round(np.random.uniform(low=0.01, high=5),2)

    cv2.imwrite(aug_path+str(i)+'.png',adjust_gamma(data[i],r))
aug_data = load_data(aug_path,ext='.png')  

aug_data = np.array(aug_data)

plt.imshow(aug_data[4])

plt.axis('off')

plt.show()
def brightness_score(img,method=1):

  if(method==1):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.round((np.mean(gray)/255)*10,2)

  else:    

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(hsv)

    return np.round((np.mean(v)/255)*10,2)



idx = 3

sample_img = aug_data[idx]

plt.subplot(121)

plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

plt.title('Method1 - score :'+str(brightness_score(sample_img,method=1)))

plt.axis('off')

plt.subplot(122)

plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

plt.title('Method2 - score :'+str(brightness_score(sample_img,method=2)))

plt.axis('off')

plt.show()
white = cv2.imread(path+'white.jpg')

white[white<255]=255

black = cv2.imread(path+'black.jpg')

black[black>0]=0



plt.subplot(221)

plt.imshow(cv2.cvtColor(white, cv2.COLOR_BGR2RGB))

plt.title('Method1 - score :'+str(brightness_score(white,method=1)))

plt.axis('off')

plt.subplot(222)

plt.imshow(cv2.cvtColor(black, cv2.COLOR_BGR2RGB))

plt.title('Method1 - score :'+str(brightness_score(black,method=1)))

plt.axis('off')

plt.subplot(223)

plt.imshow(cv2.cvtColor(white, cv2.COLOR_BGR2RGB))

plt.title('Method2 - score :'+str(brightness_score(white,method=2)))

plt.axis('off')

plt.subplot(224)

plt.imshow(cv2.cvtColor(black, cv2.COLOR_BGR2RGB))

plt.title('Method2 - score :'+str(brightness_score(black,method=2)))

plt.axis('off')

plt.show()
c=1

plt.figure(figsize=(10,10))

for i in range(3):

  for j in range(3):

    plt.subplot(3,3,c)

    idx = np.random.choice(range(50), replace=False)

    img = aug_data[idx]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title('score :'+str(brightness_score(img,method=1)))

    plt.axis('off')

    c+=1
c=1

plt.figure(figsize=(10,10))

for i in range(3):

  for j in range(3):

    plt.subplot(3,3,c)

    idx = np.random.choice(range(50), replace=False)

    img = aug_data[idx]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title('score :'+str(brightness_score(img,method=2)))

    plt.axis('off')

    c+=1