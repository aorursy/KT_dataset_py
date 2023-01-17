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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import cv2
import json
lr = 1e-4
epochs = 20
BS = 16

anno_dir='/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations/'
images_dir='/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
images=[]
labels=[]
for filename in os.listdir(images_dir):
    num = filename.split('.')[ 0 ]
    if int(num) > 1800:
        class_name = None
        anno = filename + ".json"
        with open(os.path.join(anno_dir, anno)) as json_file:
            json_data = json.load(json_file)
            no_anno = json_data["NumOfAnno"]
            k = 0
            for i in range(0, no_anno):
                class_nam = json_data['Annotations'][i]['classname']
                if class_nam in ['face_with_mask',"gas_mask", "face_shield", "mask_surgical", "mask_colorful"]:
                    class_name = 'face_with_mask'
                    k = i
                    break
                elif class_nam in ['face_no_mask,"hijab_niqab', 'face_other_covering', "face_with_mask_incorrect", "scarf_bandana", "balaclava_ski_mask", "other" ]:
                    class_name = 'face_no_mask'
                    k = i
                    break
                else:
                    continue
                    
            box = json_data[ 'Annotations' ][k][ 'BoundingBox' ]
            (x1, x2, y1, y2) = box
        if class_name is not None:
            image = cv2.imread(os.path.join(images_dir, filename))
            img = image[x2:y2, x1:y1]
            img = cv2.resize(img, (224, 224))
            img = img[...,::-1].astype(np.float32)
            img = preprocess_input(img)
            images.append(img)
            labels.append(class_name)  
   
images = np.array(images, dtype="float32")
labels = np.array(labels)
print(len(images))
print(len(labels))
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(images, labels,test_size=0.20, stratify=labels, random_state=42)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
imagedata = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                      fill_mode="nearest")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.save('/kaggle/working/model.h5')
model.summary()
new = model.fit(imagedata.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX)//BS, validation_data=(testX,testY), 
               validation_steps=len(testX)//BS, epochs=epochs)
!pip install mtcnn
sub = pd.DataFrame(columns=['name', 'x1','x2','y1','y2','classname'])
detector = MTCNN()
for filename in os.listdir(images_dir):
    temp = []
    num = filename.split('.')[0]    
    if int(num) <= 1800:
        if int(num) % 100 == 0:
            print(int(num))
        image = cv2.imread(os.path.join(images_dir, filename))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]
        faces = detector.detect_faces(image)
        if len(faces)==0:
            class_curr = 'face_no_mask'
        else:
            face = max(faces, key=lambda x:x['confidence'])
            x,y,w,h = face['box']
            x, y = abs(x), abs(y)
            roi = image[y:y+h, x:x+w]            
            roi = cv2.resize(roi, (224, 224))
            roi = roi.astype(np.float32)
            roi = preprocess_input(roi)
            temp.append(roi)
            temp = np.asarray(temp)            
            [(a,b)] = model.predict(temp, batch_size=BS)
            if a > b:
                class_curr = 'face_no_mask'
            else:
                class_curr = 'face_with_mask'
        data = {'name': filename,'x1':x,'x2':y,'y1':x+w,'y2':y+h,'classname': class_curr}
        sub = sub.append(data, ignore_index=True)

print(len(sub))
sub.sort_values(by=['name'], inplace=True)
sub.to_csv('/kaggle/working/submission.csv')
from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model.load_weights('/kaggle/imput/model.h5')

detector = MTCNN()
temp = []
image = cv2.imread('/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/0004.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(h, w) = image.shape[:2]
faces = detector.detect_faces(image)
if len(faces)==0:
    class_curr = 'face_no_mask'
else:
    face = max(faces, key=lambda x:x['confidence'])
    x,y,w,h = face['box']
    x, y = abs(x), abs(y)
    roi = image[y:y+h, x:x+w]            
    roi = cv2.resize(roi, (224, 224))
    roi = roi.astype(np.float32)
    roi = preprocess_input(roi)
    temp.append(roi)
    temp = np.asarray(temp)            
    [(a,b)] = model.predict(temp, batch_size=16)
    if a > b:
        class_curr = 'face_no_mask'
    else:
        class_curr = 'face_with_mask'
cv2.putText(image,class_curr , (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)
cv2.imwrite('/kaggle/working/test.jpg',image)
