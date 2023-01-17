from itertools import permutations
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from skimage.draw import circle
from sklearn.model_selection import train_test_split
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import tensorflow as tf
class SampleCreator:
    def __init__(self, img_size=(8, 8), min_obj_size=3, max_obj_size=5):
        if any([max_obj_size > dim for dim in img_size]):
            raise ValueError("Maximum object size should be less than image size")
        self._img_size = img_size
        self._min_obj_size = min_obj_size
        self._max_obj_size = max_obj_size
        self.available_figures = {
            "square": 0,
            "triangle": 1,
            "circle": 2,
            "dot": None
        }
    
    def get_square_img(self):
        img = np.zeros(self._img_size)

        w = np.random.randint(self._min_obj_size, self._max_obj_size)
        h = np.random.randint(self._min_obj_size, self._max_obj_size)
        x = random.randint(0, self._img_size[0] - w)
        y = random.randint(0, self._img_size[1] - h)

        img[x:x+w, y:y+h] += np.ones((w, h))
        return (img, (x, y, w, h, self.available_figures["square"]))
    
    def get_triangle_img(self):
        img = np.zeros(self._img_size)

        s = np.random.randint(self._min_obj_size, self._max_obj_size)
        x = random.randint(0, self._img_size[0] - s)
        y = random.randint(0, self._img_size[1] - s)

        triangle = np.ones((s, s))
        triangle = np.tril(triangle) if random.randint(0, 1) else np.triu(triangle)
        img[x:x+s, y:y+s] += triangle
        return (img, (x, y, s, s, self.available_figures["triangle"]))
    
    def get_dot_img(self):
        img = np.zeros(self._img_size)
        
        x = random.randint(0, self._img_size[0]-1)
        y = random.randint(0, self._img_size[1]-1)
        
        img[x, y] = 1
        return img
    
    def get_circle_img(self):
        img = np.zeros(self._img_size)
        
        s = np.random.randint(self._min_obj_size, self._max_obj_size)
        radius = int(math.floor(s / 2))
        r = random.randint(radius, self._img_size[0] - radius)
        c = random.randint(radius, self._img_size[1] - radius)
        
        rr, cc = circle(r, c, radius, shape=(self._img_size))
        img[rr, cc] = 1
        return (img, (r-radius, c-radius, radius * 2, radius * 2, self.available_figures["circle"]))
    
    def create_sample(self, objs=["square"], fake_objs=["triangle"], max_interception=1, fake_prob=0.5):
        """Create sample image and bounding boxes"""
        if set(objs + fake_objs) - set(self.available_figures.keys()):
            raise ValueError("There is unregistered figure in object list")
        while True:
            sample = np.zeros(self._img_size)
            bboxes = np.zeros((len(objs), 5))
            for i in range(len(objs)):
                if objs[i] == "square":
                    square = self.get_square_img()
                    sample += square[0]
                    bboxes[i] = square[1]
                elif objs[i] == "triangle":
                    triangle = self.get_triangle_img()
                    sample += triangle[0]
                    bboxes[i] = triangle[1]
                elif objs[i] == "circle":
                    circle = self.get_circle_img()
                    sample += circle[0]
                    bboxes[i] = circle[1]
            for i in range(len(fake_objs)):
                if random.random() > fake_prob:
                    break
                if fake_objs[i] == "square":
                    sample += self.get_square_img()[0]
                elif fake_objs[i] == "triangle":
                    sample += self.get_triangle_img()[0]
                elif fake_objs[i] == "circle":
                    sample += self.get_circle_img()[0]
                elif fake_objs[i] == "dot":
                    sample += self.get_dot_img()
            if not len([i for i in sample.flatten() if i > 1]) > max_interception:
                break
        return (sample.astype(bool).astype(int), bboxes)


class RGBSampleCreator(SampleCreator):
    def random_rgbcolor(self):
        return list(np.random.choice(range(256), size=3))
    
    def get_square_img(self):
        result = super(RGBSampleCreator, self).get_square_img()
        square = result[0]
        color = self.random_rgbcolor()
        square = [color if px != 0 else [0, 0, 0] for row in square for px in row]
        square = np.array(square).reshape(self._img_size + (3,))
        return (square,) + result[1:]
    
    def get_triangle_img(self):
        result = super(RGBSampleCreator, self).get_triangle_img()
        triangle = result[0]
        color = self.random_rgbcolor()
        triangle = [color if px != 0 else [0, 0, 0] for row in triangle for px in row]
        triangle = np.array(triangle).reshape(self._img_size + (3,))
        return (triangle,) + result[1:]
    
    def get_circle_img(self):
        result = super(RGBSampleCreator, self).get_circle_img()
        circle = result[0]
        color = self.random_rgbcolor()
        circle = [color if px != 0 else [0, 0, 0] for row in circle for px in row]
        circle = np.array(circle).reshape(self._img_size + (3,))
        return (circle,) + result[1:]
    
    def get_dot_img(self):
        img = np.zeros(self._img_size + (3,))
        
        x = random.randint(0, self._img_size[0]-1)
        y = random.randint(0, self._img_size[1]-1)
        color = self.random_rgbcolor()
        
        img[x, y] = color
        return img
    
    def interception(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Images must have equal shape")
        shape = self._img_size + (3,)
        img1 = img1.reshape(shape[0] * shape[1], shape[2])
        img2 = img2.reshape(shape[0] * shape[1], shape[2])
        img1 = [bool(sum(px)) for px in img1]
        img2 = [bool(sum(px)) for px in img2]
        return sum([px1 & px2 for px1, px2 in zip(img1, img2)])
    
    def concatim(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Images must have equal shape")
        shape = self._img_size + (3,)
        img1 = img1.reshape(shape[0] * shape[1], shape[2])
        img2 = img2.reshape(shape[0] * shape[1], shape[2])
        img_res = [px1 if sum(px2) == 0 else px2 for px1, px2 in zip(img1, img2)]
        return np.array(img_res).reshape(shape)
        
    def create_sample(self, objs=["square"], fake_objs=["triangle"], max_interception=1, fake_prob=0.5):
        """Create sample image and bounding boxes"""
        if set(objs + fake_objs) - set(self.available_figures.keys()):
            raise ValueError("There is unregistered figure in object list")
        sample = np.zeros(self._img_size + (3,))
        bboxes = np.zeros((len(objs), 4))
        classes = np.zeros((len(objs)))
        random.shuffle(objs)
        for i in range(len(objs)):
            while True:
                if objs[i] == "square":
                    img = self.get_square_img()
                elif objs[i] == "triangle":
                    img = self.get_triangle_img()
                elif objs[i] == "circle":
                    img = self.get_circle_img()
                if not self.interception(sample, img[0]) > max_interception:
                    sample = self.concatim(sample, img[0])
                    bboxes[i] += img[1][:-1]
                    classes[i] += img[1][-1]
                    break
        for i in range(len(fake_objs)):
            while True:
                if fake_objs[i] == "square":
                    img = self.get_square_img()
                elif fake_objs[i] == "triangle":
                    img = self.get_triangle_img()
                elif fake_objs[i] == "circle":
                    img = self.get_circle_img()
                elif fake_objs[i] == "dot":
                    img = self.get_dot_img()
                if not self.interception(sample, img[0]) > max_interception:
                    sample = self.concatim(sample, img[0])
                    break
        return (sample, bboxes, classes)
def draw_bbox(img, bboxes):
    colors = {
        0: "red",
        1: "green",
        2: "blue"
    }
    plt.imshow(img.T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img.shape[0], 0, img.shape[1]])
    for bbox in bboxes:
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec=colors[bbox[4]], fc='none'))

def draw_rgbbbox(img, bboxes, classes):
    colors = {
        0: "red",
        1: "green",
        2: "blue"
    }
    sns.set(style="dark")
    plt.imshow(np.transpose(img, (1, 0, 2)), interpolation='none', origin='lower', extent=[0, img.shape[0], 0, img.shape[1]])
    for bb, cl in zip(bboxes, classes):
        if not cl in colors:
            raise ValueError("Unknown class")
        plt.gca().add_patch(matplotlib.patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], ec=colors[cl], fc='none'))

def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U
img_size = (20, 20)
min_obj_size = 5
max_obj_size = 10
objs_count = 2
objs_variants = [
    ["square", "triangle"],
    ["square", "square"],
    ["triangle", "triangle"]
]
partial_count = int(40e3)
imgs_count = partial_count * len(objs_variants)
fake_objs = []
fake_prob = 0.90
max_interception = 2

creator = RGBSampleCreator(img_size, min_obj_size, max_obj_size)
sample_params = {
    "fake_objs": fake_objs,
    "max_interception": max_interception,
    "fake_prob": fake_prob
}
img = creator.create_sample(["square", "triangle"], **sample_params)
draw_rgbbbox(img[0] / 255, img[1], img[2])
"""Dataset generation"""
X = np.zeros((imgs_count, img_size[0], img_size[1], 3)) # X[i] - image with objects
y1 = np.zeros((imgs_count, objs_count, 4)) # y1[i] - list of bounding boxes
y2 = np.zeros((imgs_count, objs_count)) # y2[i] - list of class labels

for o, objs in enumerate(objs_variants):
    for i in range(partial_count * o, partial_count * (o+1)):
        sample = creator.create_sample(objs, **sample_params)
        X[i] = sample[0] / 255
        y1[i] = sample[1]
        y2[i] = sample[2]

# shuffle
p = np.random.permutation(imgs_count)
X = X[p]
y1 = y1[p]
y2 = y2[p]
print("Image data shape:", X.shape)
print("Bboxes data shape:", y1.shape)
print("Classes data shape:", y2.shape)
i = random.randint(0, imgs_count)
draw_rgbbbox(X[i], y1[i], y2[i])
# X = X.reshape((imgs_count, img_size[0] * img_size[1]))
y1 = y1.reshape((imgs_count, objs_count * 4))
train_size = int(imgs_count * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y1_train, y1_test = y1[:train_size], y1[train_size:]
y2_train, y2_test = y2[:train_size], y2[train_size:]
print("Train data shape:", X_train.shape, y1_train.shape, y2_train.shape)
print("Test data shape:", X_test.shape, y1_test.shape, y2_test.shape)
inp = Input((img_size[0], img_size[1], 3))

x = Conv2D(32, kernel_size=(3, 3), padding="same")(inp)
x = LeakyReLU()(x)
x = Conv2D(32, kernel_size=(3, 3))(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)
x = Dropout(0.25)(x)

x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
x = LeakyReLU()(x)
x = Conv2D(64, kernel_size=(3, 3))(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)
x = Dropout(0.25)(x)

x = Flatten()(x)

z1 = Dense(512, activation="relu")(x)
z1 = Dropout(0.25)(z1)
z1 = Dense(256, activation="relu")(z1)
z1 = Dropout(0.5)(z1)
z1 = Dense(objs_count * 4)(z1)
out1 = Activation("linear", name="regressor")(z1)

z2 = Dense(64, activation="relu")(x)
z2 = Dropout(0.25)(z2)
z2 = Dense(32, activation="relu")(z2)
z2 = Dropout(0.25)(z2)
z2 = Dense(objs_count)(z2)
out2 = Activation("sigmoid", name="classifier")(z2)

model = Model(inp, [out1, out2])
model.compile(optimizer="adadelta", loss="mse")
model.summary()
num_epochs = 50
batch_size = 32
def custom_mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

df_history = pd.DataFrame(columns=("loss", "val_loss", "regressor_loss",
                                   "val_regressor_loss", "classifier_loss",
                                   "val_classifier_loss"))
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} of {num_epochs}")
    hist = model.fit(X_train, [y1_train, y2_train], batch_size=batch_size, epochs=1, validation_data=(X_test, [y1_test, y2_test]), verbose=2)
    hist_row = list()
    for c in df_history.columns:
        hist_row.append(hist.history[c][0])
    df_history.loc[len(df_history)] = hist_row
    
    # TODO: generalize
    y_pred = model.predict(X_train)
    flipped_y1_train = np.concatenate([y1_train[:, 4:], y1_train[:, :4]], axis=1)
    flipped_y2_train = np.concatenate([y2_train[:, 1:], y2_train[:, :1]], axis=1)
    mse = custom_mse(y_pred[0], y1_train)
    flipped_mse = custom_mse(y_pred[0], flipped_y1_train)
    if flipped_mse < mse:
        y1_train = flipped_y1_train
        y2_train = flipped_y2_train
sns.set(style="darkgrid")
f = plt.figure(figsize=(13, 8))
plt.plot(df_history["loss"], c="blue", label="train")
plt.plot(df_history["val_loss"], c="orange", label="val")
plt.title("General loss", fontsize=15)
plt.xlabel("epoch", fontsize=13)
plt.ylabel("mean squared error", fontsize=13)
plt.show()
sns.set(style="darkgrid")
f = plt.figure(figsize=(13, 8))
plt.plot(df_history["regressor_loss"], c="blue", label="train")
plt.plot(df_history["val_regressor_loss"], c="orange", label="val")
plt.title("Regressor loss", fontsize=15)
plt.xlabel("epoch", fontsize=13)
plt.ylabel("mean squared error", fontsize=13)
plt.show()
sns.set(style="darkgrid")
f = plt.figure(figsize=(13, 8))
plt.plot(df_history["classifier_loss"], c="blue", label="train")
plt.plot(df_history["val_classifier_loss"], c="orange", label="val")
plt.title("Classifier loss", fontsize=15)
plt.xlabel("epoch", fontsize=13)
plt.ylabel("mean squared error", fontsize=13)
plt.show()
# Evaluate on a training sample
sample = X_train[random.randint(0, X_train.shape[0])]
pred = model.predict(sample.reshape((1, img_size[0], img_size[1], 3)))
y1_pred = pred[0].reshape((int(pred[0].shape[1] / 4), 4))
y2_pred = list(map(round, pred[1].flatten()))
draw_rgbbbox(sample, y1_pred, y2_pred)
# Evaluate on a new sample
sample = creator.create_sample(["triangle", "square"], **sample_params)[0] / 255
pred = model.predict(sample.reshape((1, img_size[0], img_size[1], 3)))
y1_pred = pred[0].reshape((int(pred[0].shape[1] / 4), 4))
y2_pred = list(map(round, pred[1].flatten()))
draw_rgbbbox(sample, y1_pred, y2_pred)
