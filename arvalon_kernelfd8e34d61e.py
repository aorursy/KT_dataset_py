import sys
sys.path.insert(0, '../input/eve-optimizer')
import glob
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from collections import Counter
from functools import reduce, partial, wraps
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Flatten, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Activation
from keras.models import Model, Sequential
from keras.applications.inception_resnet_v2 import preprocess_input as inc_res_preprocess
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input as dense_preprocess
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from sklearn.metrics import fbeta_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from Eve import Eve
path_to_dataset = "../input/img-minor-2018/img_kagl_train"
data = []
print("Began")
image_handlers = os.listdir(path_to_dataset)
for i, handler in enumerate(image_handlers):
    print(handler)
    path = path_to_dataset + r'/' + handler + r'/*.jpg'
    images = [(cv2.imread(file), i) for file in glob.glob(path)]
    images = np.array(images)
    data.append(images)

print("over")
data = reduce(lambda x, y: np.r_[x, y], data)
np.random.shuffle(data)
X = data[:, 0].tolist()
X = list(map(lambda x: cv2.resize(x, (224, 224)), X))
X = np.asarray(X)

y = data[:, 1]
y_counts = Counter(y)
class_weights = compute_class_weight('balanced', np.unique(y), y)

print(y_counts)
path_to_weights_inception = "../input/weight/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
path_to_weights_densenet = "../input/weight/DenseNet-BC-121-32-no-top.h5"
path_to_weights_resnet = "../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.hp5"


class Models:
    
    @staticmethod
    def inception_resnet(n_classes, input_shape):
        base_model = InceptionResNetV2(weights=path_to_weights_inception, include_top=False, input_shape=input_shape)
        base_output = base_model.output
        flatten = Flatten()(base_output)
        dense = Dense(1000, activation='relu')(flatten)
        dropout = Dropout(0.5)(dense)
        dense2 = Dense(50, activation='relu')(dropout)
        prediction = Dense(n_classes, activation='softmax')(dense2)

        for layer in base_model.layers[:5]:
            layer.trainable = False

        model = Model(input=base_model.input, output=prediction)

        return model
    
    @staticmethod
    def densenet(n_classes, input_shape):
        base_model = DenseNet121(weights=path_to_weights_densenet, include_top=False, input_shape=input_shape)
        base_output = base_model.output
        flatten = Flatten()(base_output)
        prediction = Dense(n_classes, activation='softmax')(flatten)
        
        model = Model(input=base_model.input, output=prediction)
        return model
    
    @staticmethod
    def resnet50(n_classes, input_shape):
        base_model = ResNet50(weights=path_to_weights_resnet, include_top=False, input_shape=input_shape)
        base_output = base_model.output
        flatten = Flatten()(base_output)
        dense = Dense(1000, activation='relu')
        prediction = Dence(n_classes, activation='softmax')(dense)
        
        model = Model(input=base_model.input, output=prediction)
        return model
def fbeta(y_true, y_pred, threshold_shift=1):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, min_delta=0.1),
    ReduceLROnPlateau(patience=1, min_delta=0.5)
]
ir_model = Models.inception_resnet(4, (224, 224, 3))
ir_model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-5), metrics=[fbeta, 'accuracy'])
X = inc_res_preprocess(X)
y = to_categorical(y)
history = ir_model.fit(X, y,
             epochs=15,
             batch_size=32,
             validation_split=0.3,
             shuffle=True,
             class_weight=class_weights, 
             callbacks=callbacks
             )
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['fbeta'])
plt.plot(history.history['val_fbeta'])
plt.title('model fbeta')
plt.ylabel('fbeta')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
def to_class_label(data):
    return list(map(lambda x: np.argmax(x) + 1, data))
confusion_matrix(to_class_label(y), to_class_label(ir_model.predict(X, verbose=1)))
model_for_tsne = Model(inputs=ir_model.input, outputs=ir_model.layers[-2].output)
dense_data = model_for_tsne.predict(X, verbose=1)
tsne = TSNE(n_components=2, random_state=0)
intermediates_tsne = tsne.fit_transform(dense_data)
plt.figure(figsize=(8, 8))
plt.scatter(
            x=intermediates_tsne[:,0],
            y=intermediates_tsne[:,1],
            c=to_class_label(y)
           )
plt.show()
path_to_test = "../input/img-minor-2018/img_kagl_test/test/*.jpg"
X_test = [cv2.imread(file) for file in glob.glob(path_to_test)]
X_test = np.array(X_test)
X_test = list(map(lambda x: cv2.resize(x, (224, 224)), X_test))
X_test = np.asarray(X_test)
X_test = inc_res_preprocess(X_test)
prediction =  to_class_label(ir_model.predict(X_test, verbose=1))
test_image_names = os.listdir("../input/img-minor-2018/img_kagl_test/test/")

submittion = pd.DataFrame({"Image": test_image_names, "Type": prediction})
submittion.to_csv("submittion.csv", index=False)
submittion.head()
prediction_alt = list(map(lambda x: x - 1, prediction))

submittion_alt = pd.DataFrame({"Image": test_image_names, "Type": prediction_alt})
submittion_alt.to_csv("submittion_alt.csv", index=False)
Counter(prediction)