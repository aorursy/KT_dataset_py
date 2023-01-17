import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

!pwd

!ls
#!cp -r ../input/kerasretinanet/keras-retinanet/* .

#!pip install keras-resnet

#!pip install . --user

#!python setup.py build_ext --inplace
from keras_retinanet import models

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras_retinanet.utils.visualization import draw_box, draw_caption

from keras_retinanet.utils.colors import label_color
!cp ../input/resnet50pretrained/* ./snapshots/

model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])



model = models.load_model(model_path, backbone_name='resnet50')

model = models.convert_model(model)
def predict(image):

    image = preprocess_image(image.copy())

    image, scale = resize_image(image)

    print(scale)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))



    boxes /= scale



    return boxes, scores, labels
def show_detected_objects(image_row):

    #img_path = image_row.image_name

    img_path = image_row

  

    image = read_image_bgr(img_path)



    boxes, scores, labels = predict(image)

  

    return boxes, scores, labels

  
"""

img_path = "/kaggle/input/globalwheatdetection/test/cc3532ff6.jpg"

boxes, scores, labels = show_detected_objects(img_path)

boxes, scores, labels = boxes[0], scores[0], labels[0]

print(boxes.shape)

print(scores.shape)



sc = len(scores[scores > 0.5])

sc = 5

b = boxes[:sc, :]

s = scores[:sc]

bs = []

for i, el in enumerate(b):

    el = el.astype(int)

    el[2] = el[2] - el[0] # Convert to x1, y1, w, h

    el[3] = el[3] - el[1]

    el = list(el)

    el.insert(0, s[i])

    print(el)

    bs.append(el)

rl = list(itertools.chain.from_iterable(bs))

print(rl)

rs = ' '.join(str(e) for e in rl)

print(rs)

dres = {

            'image_id': img_path.split('.')[0],

            'PredictionString': rs

            }

res = []

res.append(dres)

print(res)

test_df = pd.DataFrame(res, columns=['image_id', 'PredictionString'])

test_df.head()

"""
TEST_PATH = '/kaggle/input/global-wheat-detection/test/'

test_ids = os.listdir(TEST_PATH)

import itertools
res = []

for idx, row in enumerate(test_ids):

    img_path = TEST_PATH + row

    boxes, scores, labels = show_detected_objects(img_path)

    boxes, scores, labels = boxes[0], scores[0], labels[0]



    sc = len(scores[scores > 0.5])

    b = boxes[:sc,:]

    s = scores[:sc]

    

    bs = []

    for i, el in enumerate(b):

        el = el.astype(int)

        el[2] = el[2] - el[0] # Convert to x1, y1, w, h

        el[3] = el[3] - el[1]

        el = list(el)

        el.insert(0, s[i])

        bs.append(el)

    

    rl = list(itertools.chain.from_iterable(bs))

    rs = ' '.join(str(e) for e in rl)

    dres = {

            'image_id': row.split('.')[0],

            'PredictionString': rs

            }

    res.append(dres)

test_df = pd.DataFrame(res, columns=['image_id', 'PredictionString'])

test_df.head()
test_df.to_csv('submission.csv', index=False)