# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
df.head()
import json
train_df = pd.DataFrame()
train_df["path"] = df["image_id"].apply(lambda x: "/kaggle/input/global-wheat-detection/train/%s.jpg" % x)
train_df["x1"] = df["bbox"].apply(lambda x: json.loads(x)[0]).astype("int")
train_df["y1"] = df["bbox"].apply(lambda x: json.loads(x)[1]).astype("int")
train_df["x2"] = df["bbox"].apply(lambda x: json.loads(x)[2] + json.loads(x)[0]).astype("int")
train_df["y2"] = df["bbox"].apply(lambda x: json.loads(x)[3] + json.loads(x)[1]).astype("int")
train_df["class"] = df["source"]
train_df.to_csv("annotations.csv", header=False, index=False)
train_df.head()
classnames = [i for i in set(train_df["class"])]
idx = range(len(classnames))
train_classes = pd.DataFrame()
train_classes["class_name"] = classnames
train_classes["class_id"] = idx
train_classes.to_csv("classes.csv", header=False, index=False)
train_classes.head()
!git clone https://github.com/fizyr/keras-retinanet.git
!cd keras-retinanet && python setup.py build_ext --inplace
!cd keras-retinanet && pip install .
!cd keras-retinanet && python keras_retinanet/bin/train.py csv
!cd keras-retinanet && python keras_retinanet/bin/train.py --steps=2000  --backbone=resnet101 --epochs=5 csv ../annotations.csv ../classes.csv
!cd keras-retinanet && python keras_retinanet/bin/convert_model.py snapshots/resnet101_csv_05.h5  ./model.h5 
# show images inline
%matplotlib inline

# automatically reload modules when they have changed
%load_ext autoreload
%autoreload 2

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)
model_path = "./keras-retinanet/model.h5"

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet101')
class_df = pd.read_csv("classes.csv", header=None)
labels_to_names = {}
for i in range(len(class_df)):
    labels_to_names[i] =  class_df[0][i]
    
labels_to_names
base_dir="/kaggle/input/global-wheat-detection/test"
files = os.listdir(base_dir)
submission_df = pd.DataFrame()
image_ids = []
submission_strings = []
for f in files:
    image = read_image_bgr(base_dir + "/" + f)
    image_ids.append(f.replace(".jpg", ""))
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    subm_txt = ""
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        
        color = label_color(label)

        b = box.astype(int)
        subm_txt += "%s %s %s %s %s " % (score, b[0], b[1], b[2]-b[0], b[3]-b[1])
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    submission_strings.append(subm_txt)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


submission_df["image_id"] = image_ids
submission_df["PredictionString"] = submission_strings
submission_df.head()
submission_df.to_csv('submission.csv', index=False)
