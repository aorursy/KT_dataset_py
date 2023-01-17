import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import Callback
from IPython.display import SVG
from sklearn.model_selection import train_test_split
%matplotlib inline
n_images = 50000
n_objects = 1
img_size = 16
min_obj_size = 1
max_obj_size = 8
def generate_training_set(n_images, n_objects, img_size, min_obj_size, max_obj_size):
    images = np.zeros((n_images, img_size, img_size))
    bounding_boxes = np.zeros((n_images, n_objects, 4))
    for i in range(n_images):
        for j in range(n_objects):
            width, height = np.random.randint(min_obj_size, max_obj_size, size = 2)
            x = np.random.randint(0, img_size - width)
            y = np.random.randint(0, img_size - height)
            images[i, x : x + width, y : y + height] = 1.0
            bounding_boxes[i, j] = [x, y, width, height]
    return (images, bounding_boxes)
images, bounding_boxes = generate_training_set(n_images, n_objects, img_size, min_obj_size, max_obj_size)
print("Images shape:", images.shape)
print("Bounding Boxes shape:", bounding_boxes.shape)
def display_image(index):
    plt.imshow(images[index].T, cmap = "binary", origin='lower', extent = [0, img_size, 0, img_size])
    for box in bounding_boxes[index]:
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2], box[3], ec = 'r', fc = 'none'))
    plt.xticks([])
    plt.yticks([])
    plt.show()
display_image(np.random.randint(0, n_images))
display_image(np.random.randint(0, n_images))
display_image(np.random.randint(0, n_images))
display_image(np.random.randint(0, n_images))
x = (images.reshape(n_images, -1) - np.mean(images)) / np.std(images)
x.shape
y = bounding_boxes.reshape(n_images, -1) / img_size
y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", x_test.shape)
def classifier():
    model = Sequential()
    model.add(Dense(256, input_dim = 256))
    model.add(Activation('relu'))
    model.add(Dense(4))
    return model
model = classifier()
model.summary()
model.compile(optimizer = "adadelta", loss = 'mean_squared_error', metrics = ['accuracy'])
learning_rate_history = []
class Learning_Rate_History(Callback):
    def on_epoch_begin(self, epoch, logs = {}):
        learning_rate_history.append(K.get_value(model.optimizer.lr))
        print('Learning Rate:', learning_rate_history[-1])
model.fit(x_train, y_train, epochs = 30, validation_split = 0.1, callbacks = [Learning_Rate_History()])
y_pred = model.predict(x_test)
box_pred = y_pred * img_size
box_pred.shape
def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I # Union = Total Area - I
    return I / U
iou = [IOU(y_test[i], y_pred[i]) for i in range(len(x_test))]
def display(x, box, box_pred):
    index = np.random.randint(0, len(x))
    plt.imshow(x[index].reshape(16, 16).T, cmap = 'binary', origin = 'lower', extent = [0, img_size, 0, img_size])
    plt.gca().add_patch(Rectangle((box_pred[index][0], box_pred[index][1]),
                                      box_pred[index][2], box_pred[index][3],
                                      ec = 'r', fc = 'none'))
    plt.title("IOU: " + str(iou[index]))
    plt.xticks([])
    plt.yticks([])
    plt.show()
display(x_test, y_test, box_pred)
display(x_test, y_test, box_pred)
display(x_test, y_test, box_pred)
display(x_test, y_test, box_pred)