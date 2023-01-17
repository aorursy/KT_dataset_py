import os

print(os.listdir("../input"))
from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from PIL import Image

import cv2

from tqdm import tqdm_notebook
print('Пример картинки')

plt.figure(figsize=(15,10))

img = cv2.imread("../input/coco2017/train2017/train2017/000000281563.jpg")

label = cv2.imread("../input/coco2017/stuffthingmaps_trainval2017/train2017/000000281563.png")

# changing to the BGR format of OpenCV to RGB format for matplotlib

plt.subplot(1,3, 1)

plt.imshow(img[:,:,::-1])

plt.title("Image")

plt.subplot(1,3, 2)

plt.imshow(label)

plt.title("Label")

dst = cv2.addWeighted(img,0.3,label,0.8,0)

plt.subplot(1,3, 3)

plt.imshow(dst[:,:,::-1])

plt.title("Blending")

plt.show()
!pip install keras_maskrcnn
# Скачаем веса предобученой модели

!wget https://github.com/fizyr/keras-maskrcnn/releases/download/0.2.2/resnet50_coco_v0.2.0.h5
!pip freeze > requirements.txt
# show images inline

%matplotlib inline



# automatically reload modules when they have changed

%load_ext autoreload

%autoreload 2



# import keras

import keras



# import keras_retinanet

from keras_maskrcnn import models

from keras_maskrcnn.utils.visualization import draw_mask

from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras_retinanet.utils.colors import label_color



# import miscellaneous modules

import matplotlib.pyplot as plt

import cv2

import os

import numpy as np

import time



# set tf backend to allow memory to grow, instead of claiming everything

import tensorflow as tf



def get_session():

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    return tf.Session(config=config)



# use this environment flag to change which GPU to use

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# set the modified tf session as backend in keras

keras.backend.tensorflow_backend.set_session(get_session())
ls
# adjust this to point to your downloaded/trained model

model_path = os.path.join('./','resnet50_coco_v0.2.0.h5')



# load retinanet model

model = models.load_model(model_path, backbone_name='resnet50')

#print(model.summary())



# load label to names mapping for visualization purposes

labels_to_names = {0: 'person',}
# load image

image = read_image_bgr('../input/coco2017/train2017/train2017/000000281563.jpg')



# copy to draw on

draw = image.copy()

draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



# preprocess image for network

image = preprocess_image(image)

image, scale = resize_image(image)



# process image

start = time.time()

outputs = model.predict_on_batch(np.expand_dims(image, axis=0))

print("processing time: ", time.time() - start)



boxes  = outputs[-4][0]

scores = outputs[-3][0]

labels = outputs[-2][0]

masks  = outputs[-1][0]



# correct for image scale

boxes /= scale



# visualize detections

for box, score, label, mask in zip(boxes, scores, labels, masks):

    if score < 0.5:

        break



    color = label_color(label)

    

    b = box.astype(int)

    draw_box(draw, b, color=color)

    

    mask = mask[:, :, label]

    draw_mask(draw, b, mask, color=label_color(label))

    

    caption = "{} {:.3f}".format(labels_to_names[label], score)

    #draw_caption(draw, b, caption)

    

plt.figure(figsize=(15, 15))

plt.axis('off')

plt.imshow(draw)

plt.show()
def mask_get (image, model, THRESHOLD=0.5):

    

    #image = read_image_bgr(image)

    # copy to draw on

    draw = image.copy()

    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



    # preprocess image for network

    image = preprocess_image(image)

    image, scale = resize_image(image)



    # process image

    start = time.time()

    outputs = model.predict_on_batch(np.expand_dims(image, axis=0))

    #print("processing time: ", time.time() - start)

    draw = np.zeros((draw.shape[0], draw.shape[1], 3), np.uint8)



    boxes  = outputs[-4][0]

    scores = outputs[-3][0]

    labels = outputs[-2][0]

    masks  = outputs[-1][0]



    # correct for image scale

    boxes /= scale



    # visualize detections

    #draw = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    for box, score, label, mask in zip(boxes, scores, labels, masks):

        if score < THRESHOLD:

            break

        b = box.astype(int)

        #draw_box(draw, b, color=color)

        if label == 0:

            mask = mask[:, :, label]

            #draw = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

            draw_mask(draw, b, mask, color=label_color(label))



        #caption = "{} {:.3f}".format(labels_to_names[label], score)

        #draw_caption(draw, b, caption)



    mask_out = (draw[:, :, 0] > THRESHOLD).astype(np.uint8)

    return(mask_out)
print('Сразу сравним с эталоном')

image = read_image_bgr('../input/coco2017/train2017/train2017/000000281563.jpg')

mask_out = mask_get(image, model, THRESHOLD=0.5)

plt.figure(figsize=(15, 15))

label = cv2.imread("../input/coco2017/stuffthingmaps_trainval2017/train2017/000000281563.png")

plt.subplot(1,2, 1)

plt.imshow(mask_out)

plt.title("Predict")

plt.subplot(1,2, 2)

plt.imshow(label[:, :, 0] < 0.5)

plt.title("Label")

plt.show()
# Из sample-submission читаем по каким именно картинкам нам нужно сделать предсказание

sample_submission = pd.read_csv('../input/sf-dl-person-segmentation/sample-submission.csv')

sample_submission.info()

sample_submission.head()
# кодирование маски в EncodedPixels

def mask_to_rle(mask):

    mask_flat = mask.flatten('F')

    flag = 0

    rle_list = list()

    for i in range(mask_flat.shape[0]):

        if flag == 0:

            if mask_flat[i] == 1:

                flag = 1

                starts = i+1

                rle_list.append(starts)

        else:

            if mask_flat[i] == 0:

                flag = 0

                ends = i

                rle_list.append(ends-starts+1)

    if flag == 1:

        ends = mask_flat.shape[0]

        rle_list.append(ends-starts+1)

    #sanity check

    if len(rle_list) % 2 != 0:

        print('NG')

    if len(rle_list) == 0:

        rle = np.nan

    else:

        rle = ' '.join(map(str,rle_list))

    return rle
%%time

# Осталось дело за малым, 

# пройтись по списку картинок и сделать предикты с дальнейшим их кодированием в EncodedPixels

# Это займет значительное время (около 8 часов)...



THRESHOLD=0.5  # с уровнем от которого считаеться маска - можно поиграться



submit_rle_arr = []



for img_id in tqdm_notebook(sample_submission.ImageId.values):

    image = read_image_bgr(f'../input/coco2017/val2017/{img_id}')

    mask_out = mask_get(image, model, THRESHOLD=THRESHOLD)

    rle = mask_to_rle(mask_out)

    submit_rle_arr.append(rle)
sample_submission['EncodedPixels'] = submit_rle_arr

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()