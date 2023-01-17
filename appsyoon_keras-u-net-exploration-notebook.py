import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread 
from skimage.transform import resize
from skimage.morphology import label
# from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

IMG_WIDTH = IMG_HEIGHT = 256
IMG_CHANNELS = 3

BATCH_SIZE = 1
NUM_EPOCHS = 30
STEPS_PER_EPOCH = 600  # there are 670 training images
train_ids = next(os.walk(TRAIN_PATH))

mask_count = 0
for train_id in train_ids[1]:
    masks = next(os.walk(TRAIN_PATH + train_id + '/masks/'))[2]
    mask_count += len(masks)

print('There are {} images.'.format(len(train_ids[1])))
print('There are {} masks.'.format(mask_count))
print('Approximately {} masks per image.'.format(mask_count // len(train_ids[1])))
# image_id = train_ids[1][0]
# path = TRAIN_PATH + image_id + '/images/' + image_id + '.png'
# image = imread(path)[:,:,:3]
# print(image.shape)
# plt.imshow(image)
# plt.show()

# mask_ids = next(os.walk(TRAIN_PATH + image_id + '/masks/'))
# mask = imread(TRAIN_PATH + image_id + '/masks/' + mask_ids[2][0])
# print(mask.shape)
# plt.imshow(mask)
# plt.show()

# mask2 = imread(TRAIN_PATH + image_id + '/masks/' + mask_ids[2][1])
# masks = mask + mask2
# plt.imshow(masks)
# plt.show()


# marked = mark_boundaries(masks, mask)
# plt.imshow(marked)
# plt.show()
def preprocess_train(input_size, val_split=0.1):
    # data/
    #    stage1_train/
    #         image_id/
    #             images/
    #             masks/
    #         image_id/
    # ...
    print('Creating training generator...')

    print('Processing training data...')
    # getting the name of the images
    train_ids = next(os.walk(TRAIN_PATH))[1]

    np.random.shuffle(train_ids)

    # width, height, channels
    train_X = np.zeros((len(train_ids), input_size[0], input_size[1], input_size[2]), dtype=np.uint8)
    train_Y = np.zeros((len(train_ids), input_size[0], input_size[1], 1), dtype=np.bool)

    for index_, image_id in tqdm(enumerate(train_ids), total=len(train_ids)):
        # base path
        path = TRAIN_PATH + image_id
        # getting the images
        image_path = path + '/images/' + image_id + '.png'
        image = imread(image_path)[:,:,:input_size[2]]
        resized = resize(image, (input_size[0], input_size[1]), mode='constant', preserve_range=True)
        train_X[index_] = resized
        
        # getting the masks
        complete_mask = np.zeros((input_size[0], input_size[1], 1), dtype=np.bool)
        for mask_id in next(os.walk(path + '/masks/'))[2]:
            mask_path = path + '/masks/' + mask_id
            single_mask = imread(mask_path)
            single_mask = resize(single_mask, (input_size[0], input_size[1]), mode='constant', preserve_range=True)
            single_mask = np.expand_dims(single_mask, axis=-1)
            # creating one mask for all the masks for this image
#             iter_ = np.nditer(single_mask, flags=['multi_index'])
#             while not iter_.finished:
#                 mask_index = iter_.multi_index
#                 element = complete_mask[mask_index]
#                 if element == True and iter_[0] is not 0:
#                     complete_mask[mask_index] = False
#                 else:
#                     complete_mask[mask_index] = max(element, iter_[0])
#                 iter_.iternext()
            complete_mask = np.maximum(complete_mask, single_mask)
        train_Y[index_] = complete_mask

    val_size = int(len(train_X) * val_split)

    return (train_X[val_size:], train_Y[val_size:], train_X[:val_size], train_Y[:val_size])

def create_train_generator():
    return ImageDataGenerator(rotation_range=45, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True, vertical_flip=True)


def create_val_generator():
    return ImageDataGenerator()
train_X, train_Y, val_X, val_Y = preprocess_train(input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

train_generator = create_train_generator()
val_generator = create_val_generator()
random_i = random.randint(0, len(train_ids)) 

plt.imshow(train_X[random_i])
plt.show()

plt.imshow(np.squeeze(train_Y[random_i]))
plt.show()
def unet(input_size):
    inputs = Input((input_size[0], input_size[1], input_size[2])) # height, width, channels
    norm = Lambda(lambda x: x / 255)(inputs)

    conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(norm)
    conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_1)
    pool_1 = MaxPooling2D((2, 2))(conv_1)

    conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
    conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_2)
    pool_2 = MaxPooling2D((2, 2))(conv_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
    conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_3)
    pool_3 = MaxPooling2D((2, 2))(conv_3)

    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_3)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_4)
    pool_4 = MaxPooling2D((2, 2))(conv_4)

    conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_4)
    dropout = Dropout(0.2)(conv_5)
    conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same')(dropout)

    up_6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_5)
    up_6 = concatenate([up_6, conv_4], axis=3)
    conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up_6)
    conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_6)

    up_7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_6)
    up_7 = concatenate([up_7, conv_3], axis=3)
    conv_7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up_7)
    conv_7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_7)

    up_8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_7)
    up_8 = concatenate([up_8, conv_2], axis=3)
    conv_8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up_8)
    conv_8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_8)

    up_9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_8)
    up_9 = concatenate([up_9, conv_1], axis=3)
    conv_9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up_9)
    conv_9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_9)

    conv_10 = Conv2D(1, (1, 1), activation='sigmoid')(conv_9)

    return Model(inputs=[inputs], outputs=[conv_10])
model = unet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
def create_callbacks():
    print('Creating callbacks...')
    early = EarlyStopping(patience=5, verbose=1)
    checkpoint = ModelCheckpoint('keras_unet.h5', verbose=1, save_best_only=True)

    return [early, checkpoint]
callbacks = create_callbacks()
print('Starting training...')
history = model.fit_generator(train_generator.flow(train_X, train_Y, batch_size=BATCH_SIZE),
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              callbacks=callbacks,
                              verbose=1,
                              validation_data=val_generator.flow(
                                  val_X, val_Y, batch_size=BATCH_SIZE),
                              validation_steps=STEPS_PER_EPOCH * 0.1)

print('Finished training...')
def plot_loss_history(history):
    # validation losses
    val_loss = history.history['val_loss']
    loss = history.history['loss']

    plt.title('Loss')
    plt.plot(val_loss, 'r', loss, 'b')
    plt.show()
plot_loss_history(history)
def preprocess_test(input_size):
    test_ids = next(os.walk(TEST_PATH))[1]

    test_X = np.zeros(
        (len(test_ids), input_size[0], input_size[1], input_size[2]), dtype=np.uint8)
    # we are going to resize the predicted test images back to original size
    test_image_sizes = []

    for index_, test_id in tqdm(enumerate(test_ids), total=len(test_ids)):
        image_path = TEST_PATH + test_id + '/images/' + test_id + '.png'
        image = imread(image_path)[:,:,:input_size[2]]
        test_image_sizes.append((image.shape[0], image.shape[1]))
        resized = resize(image, (input_size[0], input_size[1]), mode='constant', preserve_range=True)
        test_X[index_] = resized

    return (test_X, test_ids, test_image_sizes)
test_X, test_ids, test_image_sizes = preprocess_test(input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
model = load_model('keras_unet.h5')
predictions = model.predict(test_X, verbose=1)
print(predictions.shape)
preds = np.squeeze(predictions)
print(preds.shape)
np.amax(preds)
index_ = random.randint(0, len(test_X))
test_image = test_X[index_]

plt.imshow(test_image)
plt.show()

prediction = preds[index_]

plt.imshow((prediction > 0.3).astype(np.uint8))
plt.show()
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
        
    return run_lengths

def prob_to_rles(x, cutoff=0.3):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
# resizing the predictions to original sizea
preds_resized = []
for index_, pred in enumerate(preds):
    image = resize(pred, test_image_sizes[index_], mode='constant', preserve_range=True)
    preds_resized.append(image)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_resized[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
submission = pd.DataFrame()
submission['ImageId'] = new_test_ids
submission['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
submission.to_csv('keras_unet_01.csv', index=False)
unique_test = set(new_test_ids)
print('Number of prediction test ids: {}'.format(len(unique_test)))
print('Number of given test ids: {}'.format(len(test_ids)))
