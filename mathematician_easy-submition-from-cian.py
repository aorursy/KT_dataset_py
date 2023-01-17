from glob import glob



train_images = glob ('../input/train/train/*/*.jpg')
train_images[:3], len(train_images)
from PIL import Image

import numpy as np



indoor=[]

outdoor=[]



for path in train_images:

    image = Image.open(path) #открваем картинку

    np_image = np.array(image) #преобразуем в массив

    image_class = path.split('/')[-2] #определяем класс

    

    if image_class == 'indoor':

        indoor.append(np_image[:,:,2].mean())

    else:

        outdoor.append(np_image[:,:,2].mean())
import matplotlib.pyplot as plt

plt.hist(indoor, range=(0,255), alpha=0.3, color='green',bins=255)

plt.hist(outdoor, range=(0,255), alpha=0.3, color='red',bins=255)

plt.show()
X=indoor+outdoor

y=[0 for i in range(len(indoor))]+[1 for i in range(len(outdoor))]

model = [1 if x>110 else 0 for x in X]
import sys

if 'sklearn' in sys.modules:

    from sklearn.metrics import roc_auc_score

    print (roc_auc_score(y, model))

else:

    print ('sklearn is not avaliable')
test_images = glob ('../input/test/test/*.jpg')
res=[]



for path in test_images:

    imageid = int(path.split('/')[-1].replace('.jpg',''))

    image = Image.open(path) 

    np_image = np.array(image) 

    prob = (np_image[:,:,2].mean() > 110)+0.0

    res.append([imageid, prob])
import pandas as pd

df = pd.DataFrame(res,columns=['image_number','prob_outdoor'])
df.to_csv('submission.csv',index=False)