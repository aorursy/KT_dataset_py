def plot_history(history, desc = ''):
    
    fig = plt.figure(figsize = (18 , 6))
    
    if desc:
        plt.title('{}'.format(desc), fontsize = 16, y = -0.1)

    subplot = (1, 2, 1)
    fig.add_subplot(*subplot)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train loss', 'valid loss'])
    plt.grid(True)
    plt.plot()
    
    subplot = (1, 2, 2)
    fig.add_subplot(*subplot)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train acc', 'valid acc'])
    plt.grid(True)
    plt.plot()

def crop_brain_contour(image, plot=False):
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    # extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]  
    
    return new_image


import os 
import re
import cv2
import numpy as np
import seaborn as sns
!pip install imutils
import imutils
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from IPython.display import Image
import matplotlib.cm as cm
from tqdm import tqdm
import matplotlib.pyplot as plt
barsize = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
IMGS = 256
main_path = '../input/mri-augmented/brain_tumor'
classes = ['no','yes']
images = []
labels = []
for i in classes:
    sub_path = os.path.join(main_path, i)
    temp = os.listdir(sub_path)
    for x in temp:
        addr = os.path.join(sub_path, x)
        img_arr = cv2.imread(addr)
#         img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr = crop_brain_contour(img_arr, False)
        img_arr = cv2.resize(img_arr, (IMGS, IMGS))
        images.append(img_arr)
        if i == 'yes':
            l = 1
        else:
            l = 0
        labels.append(l)

images = np.array(images)
labels = np.array(labels)
print(images.shape, labels.shape)
main_path = '../input/mri-augmented/brain_tumor'
classes = ['no','yes']
print(classes)
count = {}
path = main_path
for z in tqdm(classes, bar_format = barsize):
    count[z] = len(os.listdir(os.path.join(path, z)))
print('Classes : ', count)
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(images, labels, random_state = 42, test_size = .30)
x_valid, x_test, y_valid, y_test = tts(x_test, y_test, random_state = 42, test_size = .50)
x_train.shape
fig = plt.figure(figsize = (12, 4))
plt.grid(True)
plt.axis(False)

fig.add_subplot(1, 4, 1)
sns.countplot(labels, palette = 'autumn')
plt.xlabel('All')

fig.add_subplot(1, 4, 2)
sns.countplot(y_train, palette = 'autumn')
plt.xlabel('Train')

fig.add_subplot(1, 4, 3)
sns.countplot(y_test, palette = 'autumn')
plt.xlabel('Test')

fig.add_subplot(1, 4, 4)
sns.countplot(y_valid, palette = 'autumn')
plt.xlabel('Valid')

plt.show()
x_train = x_train.reshape(-1, IMGS, IMGS, 3)
x_valid = x_valid.reshape(-1, IMGS, IMGS, 3)
x_test = x_test.reshape(-1, IMGS, IMGS, 3)
fig = plt.figure(figsize = (16,7))
z = np.random.randint(1, 250, 11)
rows = 2
columns = 5
for i in range(1, rows*columns+1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[z[i]])
    plt.title(classes[labels[z[i]]])
    plt.axis(False)
plt.show()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_valid.shape, y_valid.shape)
class my_callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if (logs.get('val_accuracy') > .99):
            print("Accuracy is High Enough so Stopping Training")
            self.model.stop_training= True
base_model = tf.keras.applications.VGG16(weights = 'imagenet',
                                            include_top = False,
                                            input_shape = (IMGS, IMGS, 3))

last = base_model.get_layer('block3_pool').output
x = layers.GlobalAveragePooling2D()(last)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation = 'relu')(x)
pred = layers.Dense(1, activation = 'sigmoid')(x)
import keras
model = keras.engine.Model(base_model.input, pred)
model.compile(loss='binary_crossentropy',optimizer = tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
callback = my_callbacks()
history = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_data = (x_valid, y_valid), callbacks = [callback])
plot_history(history)
model.evaluate(x_test, y_test)
img_arr = cv2.imread('../input/mri-data/1.jpg')
# img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
img_arr = crop_brain_contour(img_arr, False)
img_arr = cv2.resize(img_arr, (IMGS, IMGS))
plt.imshow(img_arr)
plt.show()
img_arr = img_arr.reshape(1, IMGS, IMGS, 3)
print(classes[int(np.round(model.predict(img_arr)[0][0]))])
lay = []
for layer in model.layers:
    lay.append(layer.name)
    print(layer.name)
model_builder = model
img_size = (IMGS, IMGS)
def preprocess_input(img):
    img_arr = cv2.imread(img)
#     img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_arr = crop_brain_contour(img_arr, False)
    img_arr = cv2.resize(img_arr, img_size)
    img_arr = img_arr.reshape(IMGS, IMGS, 3)
    return img_arr

def decode_prediction(prediction):
    return classes[int(prediction[0])]

last_conv_layer_name = lay[9]

classifier_layer_names = lay[10:]

img_path = '../input/mri-data/1.jpg'

plt.imshow(preprocess_input(img_path))
plt.show()
def get_img_array(img_path, size):
    img_arr = cv2.imread(img_path)
#     img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_arr = crop_brain_contour(img_arr, False)
    img_arr = cv2.resize(img_arr, (IMGS, IMGS))
    img_arr = img_arr.reshape(1, IMGS, IMGS, 3)
    return img_arr
#     return img_arr
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.Input(shape = last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

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

img_array = get_img_array(img_path, size = img_size)
print(img_array.shape)

# # Make model
# model = model_builder(weights="imagenet")
# # Print what the top predicted class is
preds = model.predict(img_array)
# print(preds)
print("Predicted:", classes[int(np.round(model.predict(img_arr)[0][0]))])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)
# Display heatmap
plt.matshow(heatmap)
plt.show()
import tensorflow
img = tf.keras.preprocessing.image.load_img(img_path)
img = tf.keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = tensorflow.keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = tensorflow.keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = tensorflow.keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = "elephant_cam.jpg"
superimposed_img.save(save_path)

# Display Grad CAM
display(Image(save_path))
y_pred = model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(np.round(y_pred),y_test))
y_pred = np.round(y_pred)
y_pred = y_pred.reshape(-1,)
y_test = y_test.astype('float')
y_test.shape
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
print('''[‘True Neg’,’False Pos’
,’False Neg’,’True Pos’]''')
plt.show()
from keras.models import load_model
model.save('best_model.h5')
new_model = load_model('best_model.h5')
new_model = load_model('best_model.h5')
new_model.evaluate(x_test, y_test)
new_model.summary()