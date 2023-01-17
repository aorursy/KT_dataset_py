import os



import numpy as np

import pandas as pd



import seaborn as sea



import tensorflow as tf

from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.regularizers import l2



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder

from sklearn.utils import shuffle

from sklearn import svm



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
sea.set_style("darkgrid")
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

path = "/kaggle/input/flowers-recognition/flowers"



file_path = [os.path.join(path, "daisy/100080576_f52e8ee070_n.jpg"),

             os.path.join(path, "dandelion/10043234166_e6dd915111_n.jpg"),

             os.path.join(path, "rose/10090824183_d02c613f10_m.jpg"),

             os.path.join(path, "sunflower/1008566138_6927679c8a.jpg"),

             os.path.join(path, "tulip/100930342_92e8746431_n.jpg")]



fig = plt.figure(figsize=(10, 12))

gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)



for i in range(5):

    y, x = i//2, i%2 

    ax = fig.add_subplot(gs[y,x])

    ax.imshow(image.load_img(file_path[i]))

    ax.axis("off")

    ax.title.set_text(classes[i])
# load pretrained MobileNet

model = MobileNet(input_shape=(224,224,3), include_top=True)



model.summary()
vector = model.get_layer("reshape_2").output

feature_extractor = tf.keras.Model(model.input, vector)
# create empty feature and label lists

X_list = []

Y_list = []



for f in range(5):    

    folder_path = os.path.join(path, classes[f])

    for file in os.listdir(folder_path):    

        file_path = os.path.join(folder_path, file)

        

        # check file extension, skip file if not jpg

        if not(file.endswith(".jpg")):

            continue

        

        # load image

        img = image.load_img(file_path, target_size=(224,224))

        # convert image to numpy array

        img_arr = image.img_to_array(img)

        # add 1 more dimension

        img_arr_b = np.expand_dims(img_arr, axis=0)

        # preprocess image

        input_img = preprocess_input(img_arr_b)

        # extract feature

        feature_vec = feature_extractor.predict(input_img)

    

        X_list.append(feature_vec.ravel())

        Y_list.append(f)
X = np.asarray(X_list, dtype=np.float32)

Y = np.asarray(Y_list, dtype=np.float32)



for s in range(100):

    X, Y = shuffle(X, Y)

    

print("Shape of feature matrix X")

print(X.shape)

print("\nShape of label matrix Y")

print(Y.shape)



class_types, counts = np.unique(Y, return_counts=True)



print("\nClass labels")

print(class_types)

print("\nClass counts")

print(counts)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2,

                                                    stratify=Y,

                                                    random_state=0)



print("Shape of train_X")

print(train_X.shape)

print("\nShape of test_X")

print(test_X.shape)
svm_lin = svm.SVC(C=1.0, kernel="linear")

svm_lin.fit(train_X, train_Y)

y_pred = svm_lin.predict(test_X)

print(classification_report(test_Y, y_pred,

                            target_names=classes))
svm_nlin = svm.SVC(C=1.0, kernel="rbf")

svm_nlin.fit(train_X, train_Y)

y_pred = svm_nlin.predict(test_X)

print(classification_report(test_Y, y_pred,

                            target_names=classes))
n_encoder = OneHotEncoder(sparse=False)



# fit encoder to train_Y

n_encoder.fit(train_Y.reshape(-1,1))

# transform train_Y

e_train_Y = n_encoder.transform(train_Y.reshape(-1,1))

# transform test_Y

e_test_Y = n_encoder.transform(test_Y.reshape(-1,1))
def create_model():

    model = Sequential()

    model.add(Dense(256, input_dim=1000, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(5, kernel_regularizer=l2(0.1), activation="linear"))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),

                 loss="categorical_hinge")

    return model



epoch = 100

model = create_model()

history = model.fit(train_X, e_train_Y,

                    validation_split = 0.15,

                    epochs=epoch, batch_size=64, verbose=1)
e = np.linspace(1, epoch, epoch)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

sea.lineplot(x = e, y = history.history["loss"],

             ax=axes, label="train");

sea.lineplot(x = e, y = history.history["val_loss"],

             ax=axes, label="val");

axes.set_ylabel("Categorical Hinge Loss")

axes.set_xlabel("epoch");
y_pred = np.argmax(model.predict(test_X), axis=-1);

print(classification_report(test_Y, y_pred,

                            target_names=classes))