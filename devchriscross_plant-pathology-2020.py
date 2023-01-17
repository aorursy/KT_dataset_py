import cv2
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

import os
import gc
import tensorflow as tf
X_train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
X_train.head()
X_train.shape
X_train.healthy.value_counts()
X_train.multiple_diseases.value_counts()
X_train.rust.value_counts()
X_train.scab.value_counts()
targets = ["healthy", "multiple_diseases", "rust", "scab"]
target_sum = 0
for target in targets:
    target_sum = target_sum + X_train[target].value_counts()[1] 
target_sum
def add_class(X_train):
    X_train["class"] = "healthy"
    X_train.loc[X_train.multiple_diseases==1, "class"] = "multiple_diseases"
    X_train.loc[X_train.rust==1, "class"] = "rust"
    X_train.loc[X_train.scab==1, "class"] = "scab"
    return X_train

X_train = add_class(X_train)
sns.countplot(x="class", data=X_train)
from sklearn.model_selection import train_test_split

random_state=10
X_train, X_valid, y_train, y_valid = train_test_split(X_train, X_train[targets], train_size=0.75, random_state=random_state, stratify=X_train["class"])
X_train["image_id"] = X_train["image_id"].map(lambda x: "".join([x, ".jpg"]))
X_valid["image_id"] = X_valid["image_id"].map(lambda x: "".join([x, ".jpg"]))

base_path = "../input/plant-pathology-2020-fgvc7/images/"
def crop_leaf(full_img, border_percentage=0.1, min_percentage_width=0.7, min_percentage_height=0.5,
             base_kernel=61, kernel_step=6, base_threshold1=50, threshold2_params=[150,100,50],
             return_crop_img=False):

    edges = np.zeros((1,1,3))
    boundary_box_shape = (0, 0) 
    threshold_iter = 0
    boundary_img = None

    while (base_kernel > kernel_step+1 or threshold_iter != len(threshold2_params)):

        if (boundary_box_shape[0] > full_img.shape[0]*min_percentage_height and 
               boundary_box_shape[1] > full_img.shape[1]*min_percentage_width):
            break
        
        if threshold_iter == len(threshold2_params):
            threshold_iter = 0
            base_kernel = base_kernel - kernel_step
#         print("Finding object with kernel:", base_kernel, "and thresholds:", base_threshold1, threshold2_params[threshold_iter])

        img_blur = cv2.medianBlur(full_img, ksize=base_kernel)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, base_threshold1, threshold2_params[threshold_iter], L2gradient=True)

        height, width = edges.shape

        border_height_top = int(height*border_percentage)
        border_height_bottom = height-border_height_top
        border_width_left = int(width*border_percentage)
        border_width_right = width-border_width_left

        edges_along_y = np.nonzero(np.argmax(edges[border_height_top:border_height_bottom, 
                                                 border_width_left:border_width_right], axis=1))[0]
        edges_along_x = np.nonzero(np.argmax(edges[border_height_top:border_height_bottom, 
                                                 border_width_left:border_width_right], axis=0))[0]

        y_start = border_height_top
        y_end = border_height_top

        x_start = border_width_left
        x_end = border_width_left
        
        if len(edges_along_y) > 0:
            y_start = y_start + edges_along_y[0] 
            y_end = y_end + edges_along_y[-1]
            
        if len(edges_along_x) > 0:
            x_start = x_start + edges_along_x[0] 
            x_end = x_end + edges_along_x[-1]

        boundary_box_shape = full_img[y_start:y_end, x_start:x_end].shape
        threshold_iter = threshold_iter + 1
        
    if return_crop_img:
        return full_img[y_start:y_end, x_start:x_end]
    else:
        return x_start,y_start, x_end,y_end,edges
X_train.image_id.unique()
X_valid.image_id.unique()
def crop_img(img):
    print("Preprocessing")
    x_start,y_start, x_end,y_end,edges = crop_leaf(img.numpy(),threshold2_params=[100],kernel_step=8)
    img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2GRAY)
    return tf.convert_to_tensor(img[y_start:y_end, x_start:x_end])
# from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 64
target_size = 512
transform_range=0.10
image_gen = ImageDataGenerator(
#                       featurewise_center=True, featurewise_std_normalization=True,
                      rotation_range=30, 
                      samplewise_center=True, samplewise_std_normalization=True, 
                      height_shift_range=transform_range,
                      zoom_range=[0.8, 0.9], horizontal_flip=True, vertical_flip=True, rescale=1./255,
                      preprocessing_function=preprocess_input)

def load_img(id_):
    img = cv2.imread("".join([base_path, id_]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, dsize=(target_size,target_size))

# sample_set = X_train.groupby(["class"]).apply(lambda x: x.sample(50))
# image_set = np.asarray(list(map(load_img, X_train.image_id.tolist())))
# image_gen.fit(image_set)


image_gen_train = image_gen.flow_from_dataframe(X_train, directory=base_path, x_col="image_id", y_col="class",
                             class_mode="categorical", seed=random_state, target_size=(target_size,target_size), shuffle=True,
                                               batch_size=BATCH_SIZE)

valid_gen = ImageDataGenerator(rescale=1./255)
image_gen_valid = valid_gen.flow_from_dataframe(X_valid, directory=base_path, x_col="image_id", y_col="class",
                             class_mode="categorical", seed=random_state, target_size=(target_size,target_size), shuffle=False,
                                               batch_size=BATCH_SIZE)
def display_image(class_, count, channel=None, display=True):
    plt.figure(figsize=(10,10))
    id_ = X_train[X_train["class"]==class_].image_id.reset_index(drop=True).iloc[count]
    img = cv2.imread("".join([base_path, id_]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channel is not None:
        img = img[:,:,channel]
    
    if display:
        if channel is not None:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        
    return img
img = display_image("scab", 10, display=False)
plt.imshow(image_gen.random_transform(img, seed=1))
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def create_model(last_layers_to_unfreeze=None):
    model = DenseNet121(include_top=False, input_shape=(target_size,target_size,3), weights="imagenet", pooling="max")

    print("Number of layers:", len(model.layers))
    if last_layers_to_unfreeze is not None:
        for layer in model.layers[:-last_layers_to_unfreeze]:
            layer.trainable = False
        print("Number of frozen layers:", len(model.layers[:-last_layers_to_unfreeze]))

    model_output = Flatten()(model.output)
    model_output = Dense(512, activation="relu")(model_output)
    model_output = Dropout(0.5)(model_output)
    model_output = Dense(256, activation="relu")(model_output)
    model_output = Dropout(0.5)(model_output)
    model_output = Dense(4, activation="softmax")(model_output)
    model = Model(inputs=model.input, outputs=model_output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    
    return model
model = create_model(114)
model.summary()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(X_train["class"])
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

encoded_class = encoder.transform(X_train["class"])
c_weight = compute_class_weight("balanced", np.unique(encoded_class), encoded_class)
c_weight = dict(zip(range(4), c_weight))

early_stop = EarlyStopping(monitor="val_loss", patience=3)
# history = model.fit_generator(image_gen_train, epochs=20, callbacks=[early_stop], validation_data=image_gen_valid)
history = model.fit_generator(image_gen_train, epochs=20, callbacks=[early_stop], validation_data=image_gen_valid, class_weight=c_weight)
model.save("model_with_validation.h5")
history_df = pd.DataFrame(list(zip(history.history["loss"], history.history["val_loss"])))\
    .rename(columns={0: "loss", 1: "val_loss", 2: "categorical_accuracy", 3: "val_categorical_accuracy", "index": "epochs"})
    
plt.figure(figsize=(10,10))
history_df.plot()
history_df = pd.DataFrame(list(zip(history.history["categorical_accuracy"],  history.history["val_categorical_accuracy"])))\
    .rename(columns={0: "categorical_accuracy", 1: "val_categorical_accuracy", "index": "epochs"})
plt.figure(figsize=(10,10))
history_df.plot()
# sns.lineplot(x="index", y="categorical_accuracy", data=history_df).set_label("sasa")
# sns.lineplot(x="index", y="val_categorical_accuracy", data=history_df)
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

def eval_model(model, image_gen_valid, y_valid):
    y_valid = add_class(y_valid)
    y_true_pred = encoder.transform(y_valid["class"])
    
    valid_probs = model.predict_generator(image_gen_valid)
    valid_preds = np.argmax(valid_probs, axis=1)
    print(classification_report(y_true_pred, valid_preds))
eval_model(model, image_gen_valid, y_valid) # with class_weight
# evaluated model macro f1 score is around 0.69 with higher precision and recall scores for each class compared to last version
X_train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
X_train = add_class(X_train)
X_train["image_id"] = X_train["image_id"].map(lambda x: "".join([x, ".jpg"]))

# reinitialize for the whole dataset
image_gen = ImageDataGenerator(
#                              featurewise_center=True, featurewise_std_normalization=True,
                             rotation_range=30, samplewise_center=True, samplewise_std_normalization=True, 
                             height_shift_range=transform_range,
                             zoom_range=[0.8, 0.9], horizontal_flip=True, vertical_flip=True, rescale=1./255,
                             preprocessing_function=preprocess_input)
# image_set = np.asarray(list(map(load_img, X_train.image_id.tolist())))
# image_gen.fit(image_set)

image_gen_final = image_gen.flow_from_dataframe(X_train, directory=base_path, x_col="image_id", y_col="class",
                             class_mode="categorical", seed=random_state, target_size=(target_size,target_size), batch_size=BATCH_SIZE)

model = create_model(114)
model.fit_generator(image_gen_final, epochs=3)

# predict and submit test data
X_test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
X_test["image_id"] = X_test["image_id"].map(lambda x: "".join([x, ".jpg"]))
test_gen = ImageDataGenerator(rescale=1./255)
image_test_gen = test_gen.flow_from_dataframe(X_test, directory=base_path, x_col="image_id",
                             class_mode=None, target_size=(target_size,target_size), shuffle=False)
probs = model.predict_generator(image_test_gen)

submission = X_test.join(pd.DataFrame(probs))
submission["image_id"] = submission["image_id"].map(lambda x: x.replace(".jpg",""))
submission = submission.rename(columns=dict(zip(range(4), targets)))
submission.to_csv("submission.csv", index=False)
model.history.history
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 100))
full_img = display_image("scab", 10)
x_start,y_start, x_end,y_end,edges = crop_leaf(full_img)
cv2.rectangle(full_img, (x_start,y_start), (x_end,y_end), color=(255,0,0), thickness=10)
# plt.figure(figsize=(10,10))
ax[0].imshow(cv2.bitwise_not(edges), cmap="gray")
ax[1].imshow(full_img)
# full_img = display_image("scab", 10)
hist = cv2.calcHist([full_img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist)
help(cv2.distTransform)
src = cv2.cvtColor(display_image("scab", 15), cv2.COLOR_RGB2GRAY)
# clahe = cv2.createCLAHE()

kernel = np.array([[-1, -1, -1], 
                   [-1, 9, -1], 
                   [-1, -1, -1]], dtype=np.float32)
print(np.sum(kernel))
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian
# # convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)

img = cv2.threshold(imgLaplacian, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
plt.imshow(dist, cmap="gray")
# plt.imshow(clahe.apply)
plt.imshow(cv2.normalize(full_img, None, 0, 255, cv2.NORM_MINMAX))
help(cv2.normalize)
fig, ax = plt.subplots(nrows=20, ncols=2, figsize=(30, 100))
for i in range(10,20):
    print("item", i)
    full_img = display_image("rust", i)
    x_start,y_start, x_end,y_end,edges = crop_leaf(full_img)
    cv2.rectangle(full_img, (x_start,y_start), (x_end,y_end), color=(255,0,0), thickness=10)
    # plt.figure(figsize=(10,10))
    ax[i,0].imshow(cv2.bitwise_not(edges), cmap="gray")
    ax[i,1].imshow(full_img)
plt.show()
len(train)