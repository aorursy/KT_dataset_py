# https://www.kaggle.com/mathematician/easy-submition-from-cian
from glob import glob

import pandas as pd

import numpy as np

from PIL import Image as img



import matplotlib.pyplot as plt

%matplotlib inline
train_images = glob('../input/train/train/*/*.jpg')
temp = []



for path in train_images:

    image = img.open(path) #открваем картинку

    np_image = np.array(image) #преобразуем в массив

    np_image = np_image[:int(np_image.shape[0]/9.5),:,:] # сохраняем небо из картинки

    image_class = path.split('/')[-2] #определяем класс

    

    target = 0 if image_class == 'indoor' else 1

    

    temp.append({'y': target, 'R':np_image[:,:,0].mean(), 'G':np_image[:,:,1].mean(), 'B':np_image[:,:,2].mean()})



df = pd.DataFrame(temp)
# red

plt.hist(df[df['y'] == 0]['R'], range=(0,255), alpha=0.3, color='green',bins=255)

plt.hist(df[df['y'] == 1]['R'], range=(0,255), alpha=0.3, color='red',bins=255)

plt.show()
# green

plt.hist(df[df['y'] == 0]['G'], range=(0,255), alpha=0.3, color='green',bins=255)

plt.hist(df[df['y'] == 1]['G'], range=(0,255), alpha=0.3, color='red',bins=255)

plt.show()
# blue (хорошее разделение)

plt.hist(df[df['y'] == 0]['B'], range=(0,255), alpha=0.3, color='green',bins=255)

plt.hist(df[df['y'] == 1]['B'], range=(0,255), alpha=0.3, color='red',bins=255)

plt.show()
test_images = glob('../input/test/test/*.jpg')
temp = []



for path in test_images:

    imageid = int(path.split('/')[-1].replace('.jpg',''))

    image = img.open(path) #открваем картинку

    np_image = np.array(image) #преобразуем в массив

    np_image = np_image[:int(np_image.shape[0]/9.5),:,:] # сохраняем небо из картинки

    

    temp.append({'imageid': imageid, 'R':np_image[:,:,0].mean(), 'G':np_image[:,:,1].mean(), 'B':np_image[:,:,2].mean()})



df = pd.DataFrame(temp)
df.loc[df['B'] <= 186.469, 'pred'] = 0

df['pred'].fillna(1, inplace = True)
submission = df[['imageid', 'pred']]

submission.columns = ['image_number','prob_outdoor']

submission.to_csv('submission.csv',index=False)