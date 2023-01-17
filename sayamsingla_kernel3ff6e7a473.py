import numpy as np
import pandas as pd
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import json
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model, Model
from PIL import Image
from mtcnn.mtcnn import MTCNN

dataset=pd.read_csv("train.csv", header=None)
dataset = dataset.iloc[1:]
dataset[0] = 'images/' + dataset[0].astype(str)
dataset.head()
train = pd.read_csv('train.csv')
Y = train['classname']
train.shape
images=os.path.join("images")
annotations=os.path.join('annotations')
print(len(os.listdir(images)))
a=os.listdir(images)
b=os.listdir(annotations)
a.sort()
b.sort()
print(a[1698:1708])
print(b[:10])
test_images=a[:1698]
train_images=a[1698:]
train_ann=b
len(train_images)==len(train_ann)

df =train_csv=pd.read_csv(os.path.join('train.csv'))
train_csv.head()
submission=pd.read_csv(os.path.join("submission.csv"))
submission.head()
getbox=[]
for i in range(len(train_csv)):
    arr=[]
    for j in df.iloc[i][["x1",'x2','y1','y2']]:
        arr.append(j)
    getbox.append(arr)
df["getbox"]=getbox
df.head()
def get_boxes(id):
    boxes=[]
    for i in df[df["name"]==str(id)]["getbox"]:
        boxes.append(i)
    return boxes
print(get_boxes('1806.jpg'))
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
image=train_images[98]

img=plt.imread(os.path.join(images,image))

fig,ax = plt.subplots(1)
ax.imshow(img)
boxes=get_boxes(image)
for box in boxes:
    rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()

ann_path = "annotations/"
jdata = json.load(open(ann_path+train_ann[1860]))
anns = jdata["Annotations"]
#bb = anns[0]['BoundingBox']
bb = get_boxes('1861.jpg')
imgpath = "images/1861.jpg"
im = cv2.imread(imgpath)
fig,ax = plt.subplots(1)
ax.imshow(im)
print(bb)
for box in bb:
    print(box)
    rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()


path = "images/"
train_features = []
train_labels = []
img_size = 128

for image_name in range(3550):
    img = cv2.imread(path + train_images[image_name])
    boxes = get_boxes(train_images[image_name])
    for idx, bb in enumerate(boxes):
        x,y,w,h = bb
        label = list(df[df["name"]==train_images[image_name]]["classname"])
        #if label[idx] == "face_no_mask" or label[idx] == "face_with_mask":
        roi = img[y:h, x:w]
        try:
            roi = cv2.resize(roi, (img_size, img_size), cv2.INTER_AREA)
            train_features.append(roi)
            train_labels.append(label[idx])
        except Exception as e:
            print("[ERROR]")
X = np.array(train_features, dtype="float32")
X /= 255.0
y = np.array(train_labels)


X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3)
classes = ["hijab_niqab", "mask_colorful", "mask_surgical", "face_no_mask",
          "face_with_mask_incorrect", "face_with_mask", "face_other_covering",

           "scarf_bandana", "balaclava_ski_mask", "face_shield", "gas_mask",
          "turban", "helmet", "sunglasses", "eyeglasses", "hair_net", "hat",
          "goggles", "hood", "other"]
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
y = to_categorical(y, num_classes=len(classes))

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=2)
import pickle
with open("X.pickle","wb") as f1:
    pickle.dump(X, f1)
with open("y.pickle","wb") as f2:
    pickle.dump(y, f2)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42)
img_size = 128

vgg = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
for layer in vgg.layers:
    layer.trainable = False
top = vgg.output
top = GlobalAveragePooling2D()(top)
top = Dense(units=256, activation="relu")(top)
top = Dense(units=128, activation="relu")(top)
top = Dense(units=len(classes), activation="softmax")(top)

model = Model(inputs=vgg.input, outputs=top)
print(model.summary())
optimizer = Adam(lr=0.001)
checkpoint = ModelCheckpoint('face_mask.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)
reduceLR = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks = [checkpoint, reduceLR]
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(x_train, y_train, batch_size=64, epochs=45, 
                 validation_data=(x_test, y_test), verbose=1,
                callbacks=callbacks)
model = load_model('face_mask.h5')
score = model.evaluate(x_test, y_test)
score


sub = "submission.csv"
df = pd.read_csv(sub)
submission_images = list(df["name"])
path = "images/"

predicted_classes = []
coordinates = []
image_names = []

detector = MTCNN()


for img_name in submission_images:
    first = img_name.split(".")[0]
    last = img_name.split(".")[1]
    if last == "jpe":
        img_name = first+"."+"jpeg"
    im = cv2.imread(path+img_name)
    color = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.asarray(color)
    faces = detector.detect_faces(im)
    for i in range(len(faces)):
        x,y,w,h = faces[i]['box']
        x, y = abs(x), abs(y)
        roi = color[y:y+h, x:x+w]
        roi = cv2.resize(roi, (128,128), cv2.INTER_AREA)
        roi = np.array(roi).astype('float32')
        roi = roi.reshape(1, 128, 128, 3)
        preds = model.predict(roi)
        pred = np.argmax(preds, axis=1)
        predicted_classes.append(classes[int(pred)])
        coordinates.append([x,y,w,h])
        image_names.append(img_name)
df_names = pd.DataFrame(image_names, columns=["name"])
df_coord = pd.DataFrame(coordinates, columns=['x1','x2','y1','y2'], dtype=float)
df_class = pd.DataFrame(predicted_classes, columns=["classname"])
dataframes = [df_names, df_coord, df_class]
result = pd.concat(dataframes, axis=1)
result.to_csv(r'final_submission_face_mask.csv')
