import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
# from google.colab.patches import cv2_imshow
#*******************************************************************************
#*******************************************************************************
tf.keras.backend.clear_session()
def read_images(data):
  lst_images = []
  for i in range(len(data)):

    img = cv2.imread(data[i]) 
    
    lst_images.append(img)
  return lst_images
#*******************************************************************************
#*******************************************************************************
tf.keras.backend.clear_session()
annotations_data = pd.read_csv("/kaggle/input/brain-tumor-detection-mri/brain_tumor_detection_mri/train.csv")

print(annotations_data.head())

annotations_info = annotations_data[["region_shape_attributes"]]
print(annotations_info)
info_xy = annotations_info.values

# Coordinates
coordinates = np.zeros( (len(info_xy), 4) )
indx = 0
for info in info_xy:
  elem = info[0].split(":")
  x_min = int(elem[2].split(",")[0])
  y_min = int(elem[3].split(",")[0])

  w = int(elem[4].split(",")[0]) 
  h = int(elem[5].split("}")[0]) 
  w,h = math.sqrt(w), math.sqrt(h)
  coordinates[indx,:] = [x_min,y_min, w, h]
  indx += 1
  
# Images 
filename = annotations_data[["filename"]] 
filename = filename.values
imgs_filename = []
for fn in filename :
  imgs_filename.append("/kaggle/input/brain-tumor-detection-mri/brain_tumor_detection_mri/images/"+fn[0])


lst_imgs = read_images(imgs_filename)
print("Number of images : ", len(lst_imgs))

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout() 
  plt.imshow(lst_imgs[i], cmap='gray', interpolation='none')
  plt.title("Image")
  plt.xticks([])
  plt.yticks([])
fig.show()
print("------------------------------------------------------------------------")
#*******************************************************************************
# Scale the pixel intensities down to the [0,1] range by dividing them by 255.0 
# (this also converts them to floats).
X_train1, X_test1 = lst_imgs[0:135],lst_imgs[135:155] 
y_train1, y_test1 = coordinates[0:135], coordinates[135:155]
  
X_train, X_test = np.asarray(X_train1), np.asarray(X_test1)
y_train, y_test = np.asarray(y_train1), np.asarray(y_test1)
 

X_train = X_train.astype("float32")  
X_train = X_train / 255.0
X_test  = X_test.astype("float32")  
X_test  = X_test / 255.0
print("X_train : ",X_train.shape,"  X_test : ",X_test.shape)
print("------------------------------------------------------------------------")
#*******************************************************************************
"""
# Creating the model using the Sequential API
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="same", 
                              activation="relu", input_shape= (416,416,3)))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", 
                              activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", 
                              activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(4))

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mse"])
history = model.fit(X_train, y_train, epochs=1000, batch_size=16)
"""
# *********************************************************************************

# Creating the model 

base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(weights=
                                                  "imagenet",include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
layer_h1 = keras.layers.Dense(64, activation="relu")(avg)
layer_h2 = keras.layers.Dense(32, activation="relu")(layer_h1)
output = keras.layers.Dense(4)(layer_h2)
model = keras.Model(inputs=base_model.input, outputs=output)


"""
base_model = keras.applications.xception.Xception(weights="imagenet",include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
layer_h1 = keras.layers.Dense(64, activation="relu")(avg)
layer_h2 = keras.layers.Dense(32, activation="relu")(layer_h1)
output = keras.layers.Dense(4)(layer_h2)
model = keras.Model(inputs=base_model.input,outputs=output)
"""
#*******************************************************************************

# Freeze the weights of the pretrained layers
for layer in base_model.layers:
  layer.trainable = False

# Compiling the model 0
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mse"])
# Training the model 0
history = model.fit(X_train, y_train, epochs=10, batch_size=8)

# Unfreeze all the layers and continue training
for layer in base_model.layers:
  layer.trainable = True

# Compiling the model 1
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mse"])
# Training the model 1
history = model.fit(X_train, y_train, epochs=1000, batch_size=8)

#*******************************************************************************
# Evaluate the model
model_evaluate = model.evaluate(X_test, y_test)
print("Loss                   : ",model_evaluate) 
#*******************************************************************************
#*******************************************************************************
pred = model.predict(X_test)
pred = pred.tolist()
print("pred : \n",pred)

def intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


# from google.colab.patches import cv2_imshow

for ind in range(20):
  print("Img : ",ind)
  print("pred   : ",pred[ind])
  print("y_test : ",y_test[ind])
  image = X_test1[ind]
  yy =  y_test[ind]
  prd = pred[ind]

  iou = intersection_over_union(yy,prd)
  print("IOU : ",round(iou, 2))
  label = "IOU = "+str(round(iou, 2))
  xt,yt = int(prd[0]),int(prd[1])
  cv2.putText(image, label, (xt, yt-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1,cv2.LINE_AA)	
  
  cv2.rectangle(image, (int(yy[0]), int(yy[1])), (int(yy[2]), int(yy[3])), (0,255,0), 2)
  cv2.rectangle(image, (int(prd[0]), int(prd[1])), (int(prd[2]), int(prd[3])), (0,0,255), 2)
  #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
  # cv2_imshow(image)
  