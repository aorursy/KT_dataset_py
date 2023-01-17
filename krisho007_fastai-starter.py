from fastai.vision import *

import pandas as pd
# Load Labels

train_label = pd.read_csv("../input/banana-count-and-weight-in-a-bunch/Estu.csv", header=None)
#if flip_vert=true (default), errors are too huge and do not decrease as we train.

tfms = get_transforms(flip_vert=False, max_lighting=0.2, max_zoom=1.05, max_warp=0.1)  
data = (ImageList.from_csv(path='../input/banana-count-and-weight-in-a-bunch', csv_name='Estu.csv', folder='Banana_bunch_images')

        .split_by_rand_pct()

        .label_from_df(label_cls=FloatList) #label_cls=FloatList ensures that this is handled as 'Regression', not 'Classification'

        .transform(tfms, size=224)  #without the 'size', it results in cuda out of memory

        .databunch())
# data.normalize(imagenet_stats)  #To do. How the train works without normalization
data.show_batch(rows=3, figsize=(9,9))
# learn = cnn_learner(data, models.resnet34) #Todo. Check if other models can be used

learn = cnn_learner(data, models.resnet50) #Todo. Check if other models can be used

learn.loss = MSELossFlat # For Regression MSE loss is the Loss function
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(30, 0.05)  

learn.data.valid_ds[0][0].shape
learn.data.valid_ds[0][0]
learn.data.valid_ds[0][1]
learn.predict(learn.data.valid_ds[0][0])
# Predict for a New Image

# img = open_image('banana/test/IMG_20190401_084251.jpg')

img = open_image('banana/train/IMG_20190401_081240.jpg')

img

learn.predict(img) 
img.shape  #It is observed that whenever the size is bigger, there is a high accuracy. Need to verify. Also, can this work with resnet32 as it was trained with sz 224.
learn.export()
learn = load_learner('banana')
il = ImageList.from_folder("banana/test/63")
for image in il:

    image.show()

    print(learn.predict(image))