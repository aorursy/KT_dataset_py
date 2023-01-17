import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from keras.applications import ResNet50
from keras.layers import Dense,Dropout,Flatten,BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import top_k_categorical_accuracy


from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

print(os.listdir("../input"))
print(os.listdir("../input/resnet50/"))

from glob import glob
from skimage.io import imread

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
train_dir = "../input/humpback-whale-identification/train/"
test_dir = "../input/humpback-whale-identification/test/"

sample_submission = pd.read_csv("../input/humpback-whale-identification/sample_submission.csv")

# train.csv
train_df = pd.read_csv("../input/humpback-whale-identification/train.csv")

# psudo test.csv
test_df = pd.DataFrame(sample_submission["Image"])
test_df['Id'] = ''

print("train.csv shape = "+str(train_df.shape))
print("test.csv shape = "+str(test_df.shape))
# unique ids - also includes "new values" 
ids = train_df["Id"]
ids.value_counts().shape[0]
num_classes = 5005
#image_size = 224
train_data_gen = ImageDataGenerator(preprocess_input)
test_data_gen = ImageDataGenerator(preprocess_input)
train_generator = train_data_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='Image',
    y_col='Id',
    has_ext=True,
    shuffle=True
    )

test_generator = test_data_gen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='Image',
    y_col='Id',
    has_ext=True,
)

test_samples = test_generator.filenames
# Build the model

base_model = ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)

set_trainable = False

for layer in base_model.layers:
    if layer.name == 'res5b_branch2a':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
# ref - https://stats.stackexchange.com/questions/156471/imagenet-what-is-top-1-and-top-5-error-rate
def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
def build_model(base_model):
    model = Sequential()
    model.add(base_model)
    #model.add(Flatten())
    model.add(BatchNormalization(momentum=0.1, epsilon=1e-6))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=4096,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=num_classes,activation='softmax'))
    
    return model

model = build_model(base_model)

model.summary()
#adam = Adam()
model.compile(optimizer='sgd',loss=['categorical_crossentropy'],metrics=[top_5_accuracy])
history = model.fit_generator(train_generator,steps_per_epoch=50,epochs=25)
test_generator.reset() #?

predictions = model.predict_generator(generator=test_generator,verbose=1)
unique_labels = np.unique(ids)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])
best_th = 0.38

preds_t = np.concatenate([np.zeros((predictions.shape[0],1))+best_th, predictions],axis=1)
np.save("preds.npy",preds_t)
sample_df = pd.read_csv("../input/humpback-whale-identification/sample_submission.csv")
sample_list = list(sample_df.Image)
labels_list = ["new_whale"]+labels_list
pred_list = [[labels_list[i] for i in p.argsort()[-5:][::-1]] for p in preds_t]
pred_dic = dict((key, value) for (key, value) in zip(test_samples,pred_list))
pred_list_cor = [' '.join(pred_dic[id]) for id in sample_list]
df = pd.DataFrame({'Image':sample_list,'Id': pred_list_cor})
df.to_csv('submission.csv', header=True, index=False)
df.head()
