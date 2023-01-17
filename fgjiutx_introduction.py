import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import glob
import tqdm
from PIL import Image
from keras.layers import Dense, Conv2D, Dropout, Activation, BatchNormalization
from keras.layers import InputLayer, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
print("Training Data: ",len(os.listdir("../input/train/train/")))
print("Test     Data: ",len(os.listdir("../input/test/test/")))
train_df = pd.read_csv("../input/train.csv")
train_df.head(5) # 先頭のデータを表示します
train_name2cat = {
    data["File Name"]: 1
    if data["Category"] == "cat"
    else 0
    for i, data 
    in train_df.iterrows()
}

train_name2cat["5MkHtIqumeK8jM9S.png"]
fig, axs = plt.subplots(1, 5, figsize=(13, 3))
for i, data in train_df.iterrows():
    if i == 5:
        break
    img = plt.imread("../input/train/train/%s" % data["File Name"], 0)
    axs[i].set_title(data["Category"])
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].imshow(img)
plt.show()
class ImageLoader(object):
    def __init__(self, dirpath, imsize=256):
        self.imsize = imsize
        self.files = glob.glob(dirpath)
        print("[i] %d files loaded" % len(self.files))
    
    # バッチごとに行う初期化処理
    def init_batchs(self):
        self.images = []
        self.labels = []

    # エポックごとに行う初期化処理
    def init_epochs(self):
        random.shuffle(self.files)
    
    # pythonのジェネレータによってバッチ処理を行います
    def flow(self, batchsize=64):
        while True:
            self.init_epochs()
            self.init_batchs()
            for file in self.files:
                if os.path.isfile(file):
                    img = Image.open(file).convert("RGB")
                    img = img.resize((self.imsize, self.imsize))
                else:
                    continue
                self.images.append(np.asarray(img))
                fname = os.path.basename(file)
                try:
                    label = train_name2cat[fname]
                except KeyError:
                    print('[E] dictionary do not have key:', fname)
                    break
                self.labels.append(label)
                if len(self.images) == batchsize:
                    imgs = np.array(self.images) / 255.
                    lbls = np.eye(2)[np.array(self.labels)]
                    self.init_batchs()
                    yield (imgs, lbls)
loader = ImageLoader("../input/train/train/*")
model = Sequential([
    InputLayer((256, 256, 3)),
    Conv2D(16, 3, padding="same"),
    Dropout(.5),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(32, 3, padding="same", strides=2),
    Dropout(.5),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(64, 3, padding="same", strides=2),
    Dropout(.5),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(64, 3, padding="same", strides=2),
    Dropout(.5),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(128, 3, padding="same", strides=2),
    Dropout(.5),
    BatchNormalization(),
    Activation("relu"),
    Conv2D(256, 3, padding="same", strides=2),
    Dropout(.5),
    BatchNormalization(),
    Activation("relu"),
    GlobalAveragePooling2D(),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax")
])

model.summary()
model.compile(Adam(lr=1e-3, beta_1=0.5, beta_2=.999),
              "categorical_crossentropy",
              metrics=["accuracy"])
LIMIT = 100
i = 0

for X, y in loader.flow(batchsize=128):
    loss = model.train_on_batch(X, y)
    if i % 10 == 0:
        print(
            "\r[%05d] Loss: %.3f, Accuracy: %.2f%%" % (
                i,
                loss[0],
                loss[1] * 100
            ), 
            end="")
    if i == LIMIT:
        print("\n[i] Training Completed!")
        break
    i += 1
sample_df = pd.read_csv("../input/sample.csv")
nob = len(sample_df) // 100
print(nob, "Batchs")
THRES = .05

for i in tqdm.tnrange(nob):
    X = sample_df.iloc[i * 100:(i + 1) * 100, 0]
    imgs = []
    for x in X:
        img = Image.open("../input/test/test/%s" % x).convert("RGB")
        imgs.append(np.asarray(img.resize((256, 256))))
    imgs = np.array(imgs) / 255.
    pred = np.vectorize(lambda x: "cat" if x == 1 else "dog")(model.predict(imgs).argmax(axis=1))
    sample_df.iloc[i * 100:(i + 1) * 100, 1] = pred.reshape(-1)
sample_df.to_csv("submission.csv", index=None)
