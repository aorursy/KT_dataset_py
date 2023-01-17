import numpy as np
np.random.seed(42)
import os
import cv2
import pickle
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras

from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
OK_Pickle_Train_list = []
NG_Pickle_Train_list = []
#OK_Pickle_Test_list = []
#NG_Pickle_Test_list = []
OK_Pickle_Train_Path = '/kaggle/input/traintrain/OKImage.pickle'
NG_Pickle_Train_Path = '/kaggle/input/traintrain/NGImage.pickle'
#OK_Pickle_Test_Path = '/kaggle/input/testtest/OKTestImage.pickle'
#NG_Pickle_Test_Path = '/kaggle/input/testtest/NGTestImage.pickle'
OK_Train_Cycle = 0
NG_Train_Cycle = 0
#OK_Test_Cycle = 0
#NG_Test_Cycle = 0
OK_Train = 30000 #64765
NG_Train = 6500
#OK_Test = 148958
#NG_Test = 6298
#Train NG list 불러오기
while NG_Train > NG_Train_Cycle:
    with open(NG_Pickle_Train_Path, 'rb') as g:
        NG_Pickle = pickle.load(g) #한줄씩 읽어옴
        NG_Pickle_Train_list.append(NG_Pickle)
    NG_Train_Cycle+=1
NG_Train_Cycle = 0
#Train OK list 불러오기
while OK_Train > OK_Train_Cycle:
    with open(OK_Pickle_Train_Path, 'rb') as h:
        OK_Pickle = pickle.load(h) #한줄씩 읽어옴
        OK_Pickle_Train_list.append(OK_Pickle)
    OK_Train_Cycle+=1
OK_Train_Cycle = 0
#Test NG list 불러오기
#while NG_Test > NG_Test_Cycle:
#    with open(NG_Pickle_Test_Path, 'rb') as j:
#        NG_Pickle_ = pickle.load(j) #한줄씩 읽어옴
#        NG_Pickle_Test_list.append(NG_Pickle_)
#    NG_Test_Cycle+=1
#NG_Test_Cycle = 0
#Test OK list 불러오기
#while OK_Test > OK_Test_Cycle:
#    with open(OK_Pickle_Test_Path, 'rb') as k:
#        OK_Pickle_ = pickle.load(k) #한줄씩 읽어옴
#        OK_Pickle_Test_list.append(OK_Pickle_)
#    OK_Test_Cycle+=1
#OK_Test_Cycle = 0
Train_image = []
Train_target = []
#Test_image = []
#Test_target = []
for c in range(0, len(OK_Pickle_Train_list)):
    Train_image.append(OK_Pickle_Train_list[c])
    Train_target.append(0)


for v in range(0, len(NG_Pickle_Train_list)):
    Train_image.append(NG_Pickle_Train_list[v])
    Train_target.append(1)


#for b in range(0, len(OK_Pickle_Test_list)):
#    Test_image.append(OK_Pickle_Test_list[b])
#    Test_target.append(0)


#for n in range(0, len(NG_Pickle_Test_list)):
#    Test_image.append(NG_Pickle_Test_list[n])
#    Test_target.append(1)


del(OK_Pickle_Train_list)
del(NG_Pickle_Train_list)
#del(OK_Pickle_Test_list)
#del(NG_Pickle_Test_list)

Train_image_array = np.array(Train_image)
Train_target_array = np.array(Train_target)
#Test_image_array = np.array(Test_image)
#Test_target_array = np.array(Test_target)

train_data = (Train_image_array, Train_target_array)
#test_data = (Test_image_array, Test_target_array)

x_train, y_train = train_data
#x_test, y_test = test_data

del(Train_image)
del(Train_target)
#del(Test_image)
#del(Test_target)
del(Train_image_array)
del(Train_target_array)
#del(Test_image_array)
#del(Test_target_array)
del(train_data)
#del(test_data)
#1D Array로 변환
x_train = x_train.reshape(-1, 154 * 154)
#x_test = x_test.reshape(-1, 154 * 154)
from sklearn.model_selection import train_test_split
#Train/Val 분리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42)

# 크기 출력
print("학습 데이터 input: ", x_train.shape)
print("검증 데이터 input: ", x_val.shape)
print("평가 데이터 input: ", x_test.shape)
print("학습 데이터 output: ", y_train.shape)
print("검증 데이터 output: ", y_val.shape)
print("평가 데이터 output: ", y_test.shape)
# 필요 라이브러리 load
from sklearn.preprocessing import StandardScaler
# scaler 선언
scaler = StandardScaler()
# 학습 데이터에 대해 fitting (학습 데이터의 평균 및 표준편차 계산)
scaler.fit(x_train)
# 학습 데이터의 통계량을 이용하여 모든 데이터를 scaling
x_train_scaled = scaler.transform(x_train)
del(x_train)
x_val_scaled = scaler.transform(x_val)
del(x_val)
x_test_scaled = scaler.transform(x_test)
del(x_test)
# Image Reshape
x_train_scaled = x_train_scaled.reshape(-1, 154, 154, 1)
x_val_scaled = x_val_scaled.reshape(-1, 154, 154, 1)
x_test_scaled = x_test_scaled.reshape(-1, 154, 154, 1)

print("학습 데이터 input: ", x_train_scaled.shape)
print("검증 데이터 input: ", x_val_scaled.shape)
print("평가 데이터 input: ", x_test_scaled.shape)
# 필요 라이브러리 load
from sklearn.preprocessing import OneHotEncoder
# encoder 선언
encoder = OneHotEncoder()

# encoder fitting (라벨 정보 저장)
encoder.fit(y_train.reshape(-1, 1))

# 학습-검증-평가 y값 변환
y_train_onehot = encoder.transform(y_train.reshape(-1, 1)).toarray()
y_val_onehot = encoder.transform(y_val.reshape(-1, 1)).toarray()
y_test_onehot = encoder.transform(y_test.reshape(-1, 1)).toarray()
# 필요 라이브러리 load
from keras import Sequential
from keras import layers
from keras.optimizers import Adam
# sequential model 선언
model = Sequential([
    layers.InputLayer(input_shape=(154, 154, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])
# optimizer 선언
optimizer = Adam(lr=0.001, decay=0.001)
# model compile
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model summary
model.summary()
# 전체 가중치를 weights라는 변수에 저장 (list type)
weights = model.weights

# 각 weight의 이름을 출력
for weight in weights:
    print(weight.name, weight.shape)
# 학습 전 첫번째 convolutional layer의 첫번째 필터(kernel)
np.asarray(model.weights[0][:, :, :, 0]).reshape(3, 3)
#모델 학습
# 필요 라이브러리 load
from keras.callbacks import ModelCheckpoint
# 모델이 저장될 경로 선언
filepath = './output/model/'

# 저장 경로 생성
os.makedirs(filepath, exist_ok=True)

# 모델 이름 선언 (문자열 포맷팅을 사용하면 에폭 및 학습 결과를 모델 이름으로 저장 가능)
filename = 'weights-{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5'

# checkpoint 선언
checkpoint = ModelCheckpoint(filepath+filename, monitor='val_loss',verbose=1, save_best_only=True)
results = model.fit(x=x_train_scaled,
                    y=y_train_onehot,
                    batch_size=64,
                    epochs=10,
                    validation_data=(x_val_scaled, y_val_onehot),
                    callbacks=[checkpoint])
from keras.models import load_model

model = load_model('./output/model/weights-10-0.4700-0.8214.hdf5')
model.summary()
model_builder = keras.applications.xception.Xception
img_size = (154, 154)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = [
    "avg_pool",
    "predictions",
]

# The local path to our target image
img_path = "../input/heatmapimage/20200322_171711566_00006_T_LOW_C_NONE_crop_CD_75641.png"
#keras.utils.get_file("african_elephant.jpg", " https://i.imgur.com/Bvro0YD.png")

display(Image(img_path))
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap
# Prepare image
img_array = preprocess_input(get_img_array(img_path, size=img_size))

# Make model
model = model_builder(weights="imagenet")

# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)

# Display heatmap
plt.matshow(heatmap)
plt.show()
# We load the original image
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# 모델이 저장될 경로 선언
filepath_ = './output/heatmapimage/'

# 저장 경로 생성
os.makedirs(filepath_, exist_ok=True)

# Save the superimposed image
save_path = "grad_cam.jpg"
superimposed_img.save(save_path)

# Display Grad CAM
display(Image(save_path))
# 필요 라이브러리 load
from keras.models import load_model
# 최고 성능 모델 불러오기 (validation accuracy 기준)
models = os.listdir('./output/model')
models.sort(key=lambda x: float(x.split('-')[3].replace('.hdf5', '')), reverse=True)
best_model = load_model('./output/model/'+models[0])
# 평가 데이터에 대한 정확도 산출
test_results = best_model.evaluate(x_test_scaled, y_test_onehot)

print("Test Loss: {:.4f}".format(test_results[0]))
print("Test Accuracy: {:.4f}".format(test_results[1]))
true = x_test_scaled[1][:]
result = y_test_onehot[1][:]
target_names = ['NG', 'OK']
import pandas as pd

y_true = pd.x_test_scaled
y_pred = pd.y_test_onehot

pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(true, result)
