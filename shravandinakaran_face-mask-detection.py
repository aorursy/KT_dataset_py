
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
from mtcnn import MTCNN
import cv2
import json
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


lr = 1e-4
bs = 32
epochs = 20

images=[]
labels=[]
for filename in os.listdir(images_dir):
    num = filename.split('.')[ 0 ]
    if int(num) > 1800:
        class_name = None
        anno = filename + ".json"
        with open(os.path.join(annotations_dir, anno)) as json_file:
            json_data = json.load(json_file)
            no_anno = json_data["NumOfAnno"]
            k = 0
            for i in range(0, no_anno):
                class_nam = json_data['Annotations'][i]['classname']
                if class_nam == 'face_with_mask':
                    class_name = 'face_with_mask'
                    k = i
                    break
                elif class_nam == 'face_no_mask':
                    class_name = 'face_no_mask'
                    k = i
                    break
                else:
                    if class_nam in ['hijab_niqab', 'face_other_covering', "face_with_mask_incorrect", "scarf_bandana", "balaclava_ski_mask", "other" ]:
                        class_name = 'face_no_mask'
                    elif class_nam in ["gas_mask", "face_shield", "mask_surgical", "mask_colorful"]:
                        class_name = 'face_with_mask'
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

annotations_dir='../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations/'
images_dir='../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
(trainX, testX, trainY, testY) = train_test_split(images, labels,test_size=0.20, stratify=labels, random_state=42)
print(len(trainX))
print(len(trainY))
print(len(testX))
print(len(testY))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
imagedata = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                      fill_mode="nearest")
Model1 = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
Model2 = Model1.output
Model2 = AveragePooling2D(pool_size=(7,7))(Model2)
Model2 = Flatten(name="flatten")(Model2)
Model2 = Dense(128, activation="relu")(Model2)
Model2 = Dropout(0.5)(Model2)
Model2 = Dense(2, activation="softmax")(Model2)
model = Model(inputs=Model1.input, outputs=Model2)
for layer in Model1.layers:
    layer.trainable = False
optimizer = Adam(lr=lr,decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()
his = model.fit(imagedata.flow(trainX, trainY, batch_size=bs), steps_per_epoch=len(trainX)//bs, validation_data=(testX,testY), 
               validation_steps=len(testX)//bs, epochs=epochs)
pred = model.predict(testX, batch_size=bs)
pred = np.argmax(pred, axis=1)
print(classification_report(testY.argmax(axis=1), pred, target_names=lb.classes_))
detector = MTCNN()
image = cv2.imread(os.path.join(images_dir, '0004.jpg'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(image)
temp = []
maxx = max(faces, key=lambda x:x['confidence'])
x,y,w,h = maxx['box']
print(x)

# for i in range(len(faces)):
#     x,y,w,h = faces[i]['box']
#     x, y = abs(x), abs(y)
#     roi = image[y:y+h, x:x+w]            
#     roi = cv2.resize(roi, (224, 224))
#     roi = roi.astype(np.float32)
#     roi = preprocess_input(roi)
#     print(roi.shape)
#     seq = [x['confidence'] for x in faces]
#         print("pred")
#         temp.append(roi)
        

# temp = np.asarray(temp)
# print(type(testX))
# print(type(temp))
# [(a,b)] = model.predict(temp, batch_size=bs)
# print(a)
# print(b)
import pandas as pd
submissio = pd.DataFrame(columns=['name', 'x1','x2','y1','y2','classname'])
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
            x,y,w,h=50,50,50,50
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
            [(a,b)] = model.predict(temp, batch_size=bs)
            if a > b:
                class_curr = 'face_no_mask'
            else:
                class_curr = 'face_with_mask'
        data = {'name': filename,'x1':x,'x2':y,'y1':x+w,'y2':y+h,'classname': class_curr}
        submissio = submissio.append(data, ignore_index=True)

print(len(submissio))
                
submissio.sort_values(by=['name'], inplace=True)
print(len(submissio))
import base64
from IPython.display import HTML
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submissio)