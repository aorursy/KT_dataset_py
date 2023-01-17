import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
def bn_act(x, act=True):
    'batch normalization layer with an optinal activation layer'
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation('relu')(x)
    return x
def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    'convolutional layer which always uses the batch normalization layer'
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv
def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = tf.keras.layers.Add()([conv, shortcut])
    return output
def residual_block(x, filters, kernel_size=3, padding='same',strides=1):
    res = conv_block(x, filters, 3, padding, strides)
    res = conv_block(res, filters, 3, padding, 1)
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = tf.keras.layers.Add()([shortcut, res])
    return output
def upsample_concat_block(x, xskip):
    u = tf.keras.layers.UpSampling2D((2,2))(x)
    c = tf.keras.layers.Concatenate()([u, xskip])
    return c
def ResUNet(img_h, img_w):
    f = [16, 32, 64, 128, 256]
    inputs = tf.keras.layers.Input((img_h, img_w, 3))
    
    ## Encoder
    e0 = inputs#448×448×3
    e1 = stem(e0, f[0])#448×448×3->448×448×16
    e2 = residual_block(e1, f[1], strides=2)#448×448×16->224×224×32
    e3 = residual_block(e2, f[2], strides=2)#224×224×32->112×112×64
    e4 = residual_block(e3, f[3], strides=2)#112×112×64->56×56×128
    e5 = residual_block(e4, f[4], strides=2)#56×56×128->28×28×256
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)#28×28×256->28×28×256
    b1 = conv_block(b0, f[4], strides=1)#28×28×256->28×28×256
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)#(b1:28×28×256->56×56×256 concat e4:56×56×128 )->56×56×384
    d1 = residual_block(u1, f[4]) #56×56×384->56×56×256
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = tf.keras.layers.Conv2D(3, (1, 1), padding="same", activation="sigmoid")(d4)
    model = tf.keras.models.Model(inputs, outputs)
    return model
resunet_model = ResUNet(448, 448)
tf.keras.utils.plot_model(resunet_model
                          , show_shapes=True)
label = glob.glob('../input/coc-map-data/*/label_cost.png')
img = glob.glob('../input/coc-map-data/*/img.png')
np.random.seed(465)
index = np.random.permutation(len(img))
img = np.array(img)[index]
label = np.array(label)[index]
dataset = tf.data.Dataset.from_tensor_slices((img, label))
dataset
len(glob.glob('../input/coc-map-data/val/*'))
test_count = int(len(img)*0.2)
train_count = len(img) - test_count
dataset_train = dataset.skip(test_count)
dataset_test = dataset.take(test_count)
#训练数据集，测试数据
train_count,test_count,dataset_train,dataset_test
def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return img
def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32)/127.5 - 1
    input_mask = tf.cast(input_mask, tf.float32)
    return input_image, input_mask
@tf.function
def load_image(input_image_path, input_mask_path):
    input_image = read_jpg(input_image_path)
    input_mask = read_png(input_mask_path)
    input_image = tf.image.resize(input_image,(448, 448))
    input_mask = tf.image.resize(input_mask,(448, 448))
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask
BATCH_SIZE = 16
BUFFER_SIZE = 64
STEPS_PER_EPOCH = train_count // BATCH_SIZE
VALIDATION_STEPS = test_count // BATCH_SIZE

train = dataset_train.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset_test.map(load_image)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)
train_dataset

resunet_model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 
EPOCHS = 30
history = resunet_model.fit(train_dataset, 
                            epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset)
def loss_acc_history():
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(EPOCHS)

    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value and Accuracy Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
loss_acc_history()
import random
#test_img = glob.glob('../input/coc-val/val/*')
test_img = glob.glob('../input/coc-map-data/val/*')
aaa = random.randint(0,19)
def load_test(path,y,y1,x,x1):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = img[y:y1,x:x1]
    img = tf.image.resize(img,(448, 448))
    img = tf.cast(img, tf.float32)/127.5 - 1
    img = tf.expand_dims(img, 0)
    return img
def load_test_label(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img,(224, 224))
    return img
def back_resize(img):
    img = tf.image.resize(img,(500, 800))
    return img
def prediction_to_label(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask
prediction1 = resunet_model.predict(load_test(test_img[aaa],0,500,0,800))
prediction1 = prediction_to_label(prediction1)
prediction2 = resunet_model.predict(load_test(test_img[aaa],0,500,800,1600))
prediction2 = prediction_to_label(prediction2)
prediction3 = resunet_model.predict(load_test(test_img[aaa],500,1000,0,800))
prediction3 = prediction_to_label(prediction3)
prediction4 = resunet_model.predict(load_test(test_img[aaa],500,1000,800,1600))
prediction4 = prediction_to_label(prediction4)
plt.subplot(2,2,1)
plt.imshow(tf.keras.preprocessing.image.array_to_img(tf.image.resize(prediction1[0],(500, 800))))
plt.subplot(2,2,2)
plt.imshow(tf.keras.preprocessing.image.array_to_img(tf.image.resize(prediction2[0],(500, 800))))

plt.subplot(2,2,3)
plt.imshow(tf.keras.preprocessing.image.array_to_img(tf.image.resize(prediction3[0],(500, 800))))
plt.subplot(2,2,4)
plt.imshow(tf.keras.preprocessing.image.array_to_img(tf.image.resize(prediction4[0],(500, 800))))
plt.axis('off')
#plt.subplots_adjust(wspace=0.9, hspace=0.00000000000000000000000001)
plt.show()
test_imgs = tf.io.read_file(test_img[aaa])
test_imgs = tf.image.decode_png(test_imgs)
test_imgs1 = test_imgs[0:500,0:800]
test_imgs2 = test_imgs[0:500,800:1600]
test_imgs3 = test_imgs[500:1000,0:800]
test_imgs4 = test_imgs[500:1000,800:1600]
test_imgs = tf.cast(test_imgs, tf.float32)/127.5 - 1

plt.subplot(2,2,1)
plt.imshow(test_imgs1.numpy())
plt.subplot(2,2,2)
plt.imshow(test_imgs2.numpy())
plt.subplot(2,2,3)
plt.imshow(test_imgs3.numpy())
plt.subplot(2,2,4)

plt.imshow(test_imgs4.numpy())
plt.axis('off')
#plt.subplots_adjust(wspace=0.9, hspace=0.00000000000000000000000001)
plt.show()
a1 = tf.image.resize(prediction1[0],(500, 800)).numpy()
a2 = tf.image.resize(prediction2[0],(500, 800)).numpy()
a3 = tf.image.resize(prediction3[0],(500, 800)).numpy()
a4 = tf.image.resize(prediction4[0],(500, 800)).numpy()
b1 = np.hstack((a1,a2))
b2 = np.hstack((a3,a4))
b3 = np.vstack((b1,b2))
b3.shape
b4 = tf.keras.preprocessing.image.array_to_img(b3)
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.subplot(1,2,1)
plt.imshow(test_imgs[0:1000,0:1600])
plt.subplot(1,2,2)
plt.imshow(b4)
plt.show()
converter = tf.lite.TFLiteConverter.from_keras_model(resunet_model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open('model_tflite.tflite', 'wb').write(tflite_model)