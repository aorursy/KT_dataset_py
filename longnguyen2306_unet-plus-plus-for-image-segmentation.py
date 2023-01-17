import numpy as np
import cv2
import os
import tensorflow as tf
from tqdm import tqdm

tf.__version__
import gc

x = []
y = []

gc.collect()

names = os.listdir("/kaggle/input/bacteria-detection-with-darkfield-microscopy/images")
for name in tqdm(names):
    image_path = f"/kaggle/input/bacteria-detection-with-darkfield-microscopy/images/{name}"
    mask_path = f"/kaggle/input/bacteria-detection-with-darkfield-microscopy/masks/{name}"
    assert os.path.exists(image_path)
    assert os.path.exists(mask_path)
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert len(image.shape) == 3
    assert len(mask.shape) == 2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    x.append(image.astype(np.float32))
    y.append(mask.astype(np.float32))
assert len(x) == len(y)
len(x), len(y)
import typing

class SlidingWindow:
    """
    Slide over and images with a stride of 256 and cut images into smaller 256 x 256 images.
    """
    def __init__(self, width=256, height=256, stride=256):
        self.width = width
        self.height = height
        self.stride = stride

    def do(self, image, mask) -> typing.List[typing.Tuple[np.ndarray, np.ndarray]]:
        assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]

        if image.shape[1] < self.width and image.shape[0] < self.height:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_AREA)
            assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]
        elif image.shape[1] > self.width and image.shape[0] < self.height:
            w = int(round(self.height * image.shape[1] / image.shape[0]))
            image = cv2.resize(image, (w, self.height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (w, self.height), interpolation=cv2.INTER_AREA)
            assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]
        elif image.shape[1] < self.width and image.shape[0] > self.height:
            h = int(round(self.width * image.shape[0] / image.shape[1]))
            image = cv2.resize(image, (self.width, h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.width, h), interpolation=cv2.INTER_AREA)
            assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]

        if image.shape[1] == self.width and image.shape[0] == self.height:
            return [(image, mask)]

        ret = []
        i_h, i_w, _ = image.shape
        s_h, s_w = self.width, self.height

        if image.shape[1] < self.width + self.stride or image.shape[0] < self.height + self.stride:  # Image too small
            left_top = (image[0:s_h, 0:s_w, :], mask[0:s_h, 0:s_w])
            right_bottom = (image[i_h - s_h:i_h, i_w - s_w:i_w, :], mask[i_h - s_h:i_h, i_w - s_w:i_w])
            left_bottom = (image[0:s_h, i_w - s_w: i_w, :], mask[0:s_h, i_w - s_w:i_w])
            right_top = (image[i_h - s_h:i_h, 0:s_w, :], mask[i_h - s_h:i_h, 0:s_w])

            assert left_top[0].shape[0] == self.height and left_top[0].shape[1] == self.width
            assert left_top[1].shape[0] == self.height and left_top[1].shape[1] == self.width
            assert right_bottom[0].shape[0] == self.height and right_bottom[0].shape[1] == self.width
            assert right_bottom[1].shape[0] == self.height and right_bottom[1].shape[1] == self.width
            assert left_bottom[0].shape[0] == self.height and left_bottom[0].shape[1] == self.width
            assert left_bottom[1].shape[0] == self.height and left_bottom[1].shape[1] == self.width
            assert right_top[0].shape[0] == self.height and right_top[0].shape[1] == self.width
            assert right_top[1].shape[0] == self.height and right_top[1].shape[1] == self.width

            ret.append(left_top)
            ret.append(right_bottom)
            ret.append(left_bottom)
            ret.append(right_top)
        elif image.shape[1] < self.width + self.stride and image.shape[0] > self.height + self.stride:  # Width small
            for x in range(0, image.shape[1] - self.stride, self.stride):
                left = (image[0:s_h, x:x + s_w, :], mask[0:s_h, x: x + s_w])
                right = (image[i_h - s_h:i_h, x: x + s_w, :], mask[i_h - s_h: i_h, x: x + s_w])

                assert left[0].shape[0] == self.height and left[0].shape[1] == self.width
                assert left[1].shape[0] == self.height and left[1].shape[1] == self.width
                assert right[0].shape[0] == self.height and right[0].shape[1] == self.width
                assert right[1].shape[0] == self.height and right[1].shape[1] == self.width

                ret.append(left)
                ret.append(right)
        elif image.shape[1] > self.width + self.stride and image.shape[0] < self.height + self.stride:  # Height small
            for y in range(0, image.shape[0] - self.stride, self.stride):
                top = (image[y:y + s_h, 0:s_w, :], mask[y:y + s_h, 0:s_w])
                bottom = (image[y:y + s_h, i_w - s_w: i_w, :], mask[y:y + s_h, i_w - s_w: i_w])

                assert bottom[1].shape[0] == self.height and bottom[1].shape[1] == self.width
                assert bottom[0].shape[0] == self.height and bottom[0].shape[1] == self.width
                assert top[1].shape[0] == self.height and top[1].shape[1] == self.width
                assert top[0].shape[0] == self.height and top[0].shape[1] == self.width

                ret.append(top)
                ret.append(bottom)
        else:  
            self.last_image = []
            self.last_mask = []
            for x in range(0, image.shape[1] - self.stride, self.stride):
                for y in range(0, image.shape[0] - self.stride, self.stride):
                    x0 = x
                    x1 = x + self.width
                    y0 = y
                    y1 = y + self.height

                    image_slide = image[y0:y1, x0:x1, :]
                    mask_slide = mask[y0:y1, x0:x1]

                    assert image_slide.shape[0] == mask_slide.shape[0] and image_slide.shape[1] == mask_slide.shape[1]

                    if image_slide.shape[1] == self.width and image_slide.shape[0] == self.height:
                        pass
                    elif image_slide.shape[1] < self.width and image_slide.shape[0] == self.height:
                        x1 = image.shape[1]
                        x0 = x1 - self.width
                    elif image_slide.shape[1] == self.width and image_slide.shape[0] < self.height:
                        y1 = image.shape[0]
                        y0 = y1 - self.height
                    else:
                        x1 = image.shape[1]
                        x0 = x1 - self.width
                        y1 = image.shape[0]
                        y0 = y1 - self.height

                    image_slide = image[y0:y1, x0:x1, :]
                    mask_slide = mask[y0:y1, x0:x1]

                    assert image_slide.shape[0] == mask_slide.shape[0] and image_slide.shape[1] == mask_slide.shape[1]
                    assert image_slide.shape[0] == self.height and image_slide.shape[1] == self.width

                    ret.append((image_slide, mask_slide))
        return ret
images = []
masks = []
sliding = SlidingWindow()
for _ in tqdm(range(len(x))):
    image = x.pop(0)
    mask = y.pop(0)

    assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]
    slides = sliding.do(image, mask)
    for slide in slides:
        slided_image = slide[0]
        slide_mask = slide[1]
        images.append(slided_image)
        masks.append(slide_mask)
images = np.array(images)
masks = np.array(masks)
masks = masks.reshape((masks.shape[0], masks.shape[1], masks.shape[2], 1))
images.shape, masks.shape
images.dtype, masks.dtype
import matplotlib.pyplot as plt

def visualize(index):
    image = images[index].copy()
    mask = masks[index].copy().reshape((256, 256))
    
    mask_applied = image.copy()
    mask_applied[mask == 1] = [255, 0, 0]
    mask_applied[mask == 2] = [255, 255, 0]

    out = image.copy()
    mask_applied = cv2.addWeighted(mask_applied, 0.5, out, 0.5, 0, out)
    
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(image)
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(mask, cmap="gray")
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(mask_applied)
visualize(1)
visualize(255)
visualize(520)
import tensorflow as tf
import tensorflow.keras.backend as K
import typing


def weighted_loss(original_loss_function: typing.Callable, weights_list: dict) -> typing.Callable:
    def loss_function(true, pred):
        class_selectors = tf.cast(K.argmax(true, axis=-1), tf.int32)
        class_selectors = [K.equal(i, class_selectors) for i in range(len(weights_list))]
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        loss = original_loss_function(true, pred)
        loss = loss * weight_multiplier
        return loss
    return loss_function


@tf.function
def loss(y_true, y_pred, smooth=1, cat_weight=1, iou_weight=1, dice_weight=1):
    return cat_weight * K.categorical_crossentropy(y_true, y_pred) \
           + iou_weight * log_iou(y_true, y_pred, smooth) \
           + dice_weight * log_dice(y_true, y_pred, smooth)

@tf.function
def log_iou(y_true, y_pred, smooth=1):
    return - K.log(iou(y_true, y_pred, smooth))


@tf.function
def log_dice(y_true, y_pred, smooth=1):
    return -K.log(dice(y_true, y_pred, smooth))


@tf.function
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


@tf.function
def dice(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam

batch_size = 2
epochs = 25
number_of_filters = 2
num_classes = 3


def conv2d(filters: int):
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  padding='same')


def conv2dtranspose(filters: int):
    return Conv2DTranspose(filters=filters,
                           kernel_size=(2, 2),
                           strides=(2, 2),
                           padding='same')


class UNetPP:
    def __init__(self):
        model_input = Input((256, 256, 3))
        x00 = conv2d(filters=int(16 * number_of_filters))(model_input)
        x00 = BatchNormalization()(x00)
        x00 = LeakyReLU(0.01)(x00)
        x00 = Dropout(0.2)(x00)
        x00 = conv2d(filters=int(16 * number_of_filters))(x00)
        x00 = BatchNormalization()(x00)
        x00 = LeakyReLU(0.01)(x00)
        x00 = Dropout(0.2)(x00)
        p0 = MaxPooling2D(pool_size=(2, 2))(x00)

        x10 = conv2d(filters=int(32 * number_of_filters))(p0)
        x10 = BatchNormalization()(x10)
        x10 = LeakyReLU(0.01)(x10)
        x10 = Dropout(0.2)(x10)
        x10 = conv2d(filters=int(32 * number_of_filters))(x10)
        x10 = BatchNormalization()(x10)
        x10 = LeakyReLU(0.01)(x10)
        x10 = Dropout(0.2)(x10)
        p1 = MaxPooling2D(pool_size=(2, 2))(x10)

        x01 = conv2dtranspose(int(16 * number_of_filters))(x10)
        x01 = concatenate([x00, x01])
        x01 = conv2d(filters=int(16 * number_of_filters))(x01)
        x01 = BatchNormalization()(x01)
        x01 = LeakyReLU(0.01)(x01)
        x01 = conv2d(filters=int(16 * number_of_filters))(x01)
        x01 = BatchNormalization()(x01)
        x01 = LeakyReLU(0.01)(x01)
        x01 = Dropout(0.2)(x01)

        x20 = conv2d(filters=int(64 * number_of_filters))(p1)
        x20 = BatchNormalization()(x20)
        x20 = LeakyReLU(0.01)(x20)
        x20 = Dropout(0.2)(x20)
        x20 = conv2d(filters=int(64 * number_of_filters))(x20)
        x20 = BatchNormalization()(x20)
        x20 = LeakyReLU(0.01)(x20)
        x20 = Dropout(0.2)(x20)
        p2 = MaxPooling2D(pool_size=(2, 2))(x20)

        x11 = conv2dtranspose(int(16 * number_of_filters))(x20)
        x11 = concatenate([x10, x11])
        x11 = conv2d(filters=int(16 * number_of_filters))(x11)
        x11 = BatchNormalization()(x11)
        x11 = LeakyReLU(0.01)(x11)
        x11 = conv2d(filters=int(16 * number_of_filters))(x11)
        x11 = BatchNormalization()(x11)
        x11 = LeakyReLU(0.01)(x11)
        x11 = Dropout(0.2)(x11)

        x02 = conv2dtranspose(int(16 * number_of_filters))(x11)
        x02 = concatenate([x00, x01, x02])
        x02 = conv2d(filters=int(16 * number_of_filters))(x02)
        x02 = BatchNormalization()(x02)
        x02 = LeakyReLU(0.01)(x02)
        x02 = conv2d(filters=int(16 * number_of_filters))(x02)
        x02 = BatchNormalization()(x02)
        x02 = LeakyReLU(0.01)(x02)
        x02 = Dropout(0.2)(x02)

        x30 = conv2d(filters=int(128 * number_of_filters))(p2)
        x30 = BatchNormalization()(x30)
        x30 = LeakyReLU(0.01)(x30)
        x30 = Dropout(0.2)(x30)
        x30 = conv2d(filters=int(128 * number_of_filters))(x30)
        x30 = BatchNormalization()(x30)
        x30 = LeakyReLU(0.01)(x30)
        x30 = Dropout(0.2)(x30)
        p3 = MaxPooling2D(pool_size=(2, 2))(x30)

        x21 = conv2dtranspose(int(16 * number_of_filters))(x30)
        x21 = concatenate([x20, x21])
        x21 = conv2d(filters=int(16 * number_of_filters))(x21)
        x21 = BatchNormalization()(x21)
        x21 = LeakyReLU(0.01)(x21)
        x21 = conv2d(filters=int(16 * number_of_filters))(x21)
        x21 = BatchNormalization()(x21)
        x21 = LeakyReLU(0.01)(x21)
        x21 = Dropout(0.2)(x21)

        x12 = conv2dtranspose(int(16 * number_of_filters))(x21)
        x12 = concatenate([x10, x11, x12])
        x12 = conv2d(filters=int(16 * number_of_filters))(x12)
        x12 = BatchNormalization()(x12)
        x12 = LeakyReLU(0.01)(x12)
        x12 = conv2d(filters=int(16 * number_of_filters))(x12)
        x12 = BatchNormalization()(x12)
        x12 = LeakyReLU(0.01)(x12)
        x12 = Dropout(0.2)(x12)

        x03 = conv2dtranspose(int(16 * number_of_filters))(x12)
        x03 = concatenate([x00, x01, x02, x03])
        x03 = conv2d(filters=int(16 * number_of_filters))(x03)
        x03 = BatchNormalization()(x03)
        x03 = LeakyReLU(0.01)(x03)
        x03 = conv2d(filters=int(16 * number_of_filters))(x03)
        x03 = BatchNormalization()(x03)
        x03 = LeakyReLU(0.01)(x03)
        x03 = Dropout(0.2)(x03)

        m = conv2d(filters=int(256 * number_of_filters))(p3)
        m = BatchNormalization()(m)
        m = LeakyReLU(0.01)(m)
        m = conv2d(filters=int(256 * number_of_filters))(m)
        m = BatchNormalization()(m)
        m = LeakyReLU(0.01)(m)
        m = Dropout(0.2)(m)

        x31 = conv2dtranspose(int(128 * number_of_filters))(m)
        x31 = concatenate([x31, x30])
        x31 = conv2d(filters=int(128 * number_of_filters))(x31)
        x31 = BatchNormalization()(x31)
        x31 = LeakyReLU(0.01)(x31)
        x31 = conv2d(filters=int(128 * number_of_filters))(x31)
        x31 = BatchNormalization()(x31)
        x31 = LeakyReLU(0.01)(x31)
        x31 = Dropout(0.2)(x31)

        x22 = conv2dtranspose(int(64 * number_of_filters))(x31)
        x22 = concatenate([x22, x20, x21])
        x22 = conv2d(filters=int(64 * number_of_filters))(x22)
        x22 = BatchNormalization()(x22)
        x22 = LeakyReLU(0.01)(x22)
        x22 = conv2d(filters=int(64 * number_of_filters))(x22)
        x22 = BatchNormalization()(x22)
        x22 = LeakyReLU(0.01)(x22)
        x22 = Dropout(0.2)(x22)

        x13 = conv2dtranspose(int(32 * number_of_filters))(x22)
        x13 = concatenate([x13, x10, x11, x12])
        x13 = conv2d(filters=int(32 * number_of_filters))(x13)
        x13 = BatchNormalization()(x13)
        x13 = LeakyReLU(0.01)(x13)
        x13 = conv2d(filters=int(32 * number_of_filters))(x13)
        x13 = BatchNormalization()(x13)
        x13 = LeakyReLU(0.01)(x13)
        x13 = Dropout(0.2)(x13)

        x04 = conv2dtranspose(int(16 * number_of_filters))(x13)
        x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
        x04 = conv2d(filters=int(16 * number_of_filters))(x04)
        x04 = BatchNormalization()(x04)
        x04 = LeakyReLU(0.01)(x04)
        x04 = conv2d(filters=int(16 * number_of_filters))(x04)
        x04 = BatchNormalization()(x04)
        x04 = LeakyReLU(0.01)(x04)
        x04 = Dropout(0.2)(x04)

        output = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax')(x04)

        self.model = tf.keras.Model(inputs=[model_input], outputs=[output])
        self.optimizer = Adam(lr=0.0005)

    def compile(self, loss_function, metrics=[iou, dice]):
        self.model.compile(optimizer=self.optimizer, loss=loss_function, metrics=metrics)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, class_weights=None):
        self.compile(loss_function=weighted_loss(loss, class_weights))
        return self.model.fit(x_train, y_train,
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              validation_data=[x_val, y_val],
                              validation_steps=x_val.shape[0] // batch_size,
                              batch_size=batch_size,
                              epochs=epochs,
                              shuffle=True)
    
unet = UNetPP()
unet.model.summary()
values, counts = np.unique(masks.astype(np.int32), return_counts=True)
weights = dict()
max_count = max(counts)
for i in range(len(counts)):
    weights[float(i)] = max_count / counts[i]
weights
validation_split = 0.15

# I have to use only 2000 images because the kaggle's cloud instance does not have enough RAM
images = images[0:2000]
masks = masks[0:2000]

gc.collect()

indices = np.random.permutation(images.shape[0])
boundary = images.shape[0] - int(images.shape[0] * validation_split)
training_idx, test_idx = indices[:boundary], indices[boundary:]
x_train, x_test = images[training_idx, :], images[test_idx, :]
y_train, y_test = masks[training_idx, :], masks[test_idx, :]
x_train.shape, y_train.shape, x_test.shape, y_test.shape
y_train = tf.keras.utils.to_categorical(y_train).astype(np.float32)
y_test = tf.keras.utils.to_categorical(y_test).astype(np.float32)
y_train.shape, y_test.shape
for i in range(0, x_train.shape[0], 300):
    unet.fit(x_train[i:i+300, :, :, :], y_train[i:i+300, :, :, :], x_test, y_test, weights)