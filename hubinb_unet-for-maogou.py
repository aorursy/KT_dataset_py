import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np

test_image = glob.glob(r'/kaggle/input/cat-dog-tupiandingwei/images/*jpg')
train_image = glob.glob(r'/kaggle/input/cat-dog-tupiandingwei/trimaps/*png')
#做相同的乱序
np.random.seed=2020
index =np.random.permutation(len(train_image))
test_image = np.array(test_image)[index]
train_image = np.array(train_image)[index]
def load_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img,(224,224))
    img = tf.cast(img,tf.float32)
    img = img / 127.5 -1 #归一化到-1~1 
    return img
def load_train(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img,channels=1)
    img = tf.image.resize(img,(224,224))
    img = tf.cast(img,tf.float32)
    img -= 1  #从0开始，0，1，2 
    return img
test_img = tf.data.Dataset.from_tensor_slices(test_image)
train_img = tf.data.Dataset.from_tensor_slices(train_image)
test_img = test_img.map(load_test,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE) #加速
train_img = train_img.map(load_train,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
da_t = tf.data.Dataset.zip((test_img,train_img))
print(da_t)
num = len(train_image)
#print(num)
test_num = int(num*0.2)
train_num = num - test_num
da_train = da_t.skip(test_num)
da_test = da_t.take(test_num)
batch_size = 2
test_step = test_num // batch_size
train_step = train_num // batch_size
da_train = da_train.shuffle(10).repeat().batch(batch_size)
da_test = da_test.batch(batch_size)
model = tf.keras.applications.xception.Xception(include_top=False,
                                                input_shape = (224,224,3))

# model.summary()
layer_name = ['block14_sepconv2_act',
              'block13_sepconv2',
              'block4_sepconv2',
              'block3_sepconv2',
              'block2_sepconv2',           
    ]
layers =[model.get_layer(p).output for p in layer_name]

new_model = tf.keras.models.Model(inputs = model.input,
                                  outputs = layers)
# new_model.summary()
#设置不可训练
new_model.trainable = False
input = tf.keras.layers.Input((224,224,3))
x1,x2,x3,x4,x5 = new_model(input)
print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)
print(x5.shape)
x = tf.keras.layers.Conv2DTranspose(1024,3,2,padding='same')(x1)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.Conv2D(1024,3,padding='same')(x)
x = tf.keras.activations.relu(x)
x = tf.add(x,x2)
print(x.shape)
x = tf.keras.layers.Conv2DTranspose(728,3,2,padding='same')(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.Conv2D(728,3,padding='same')(x)
x = tf.keras.activations.relu(x)
x = tf.add(x,x3)
print(x.shape)
x = tf.keras.layers.Conv2DTranspose(256,3,2,padding='valid')(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.Conv2D(256,3,padding='valid')(x)
x = tf.keras.activations.relu(x)
x = tf.add(x,x4)
print(x.shape)
x = tf.keras.layers.Conv2DTranspose(128,3,2,padding='valid')(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.Conv2D(128,3,padding='valid')(x)
x = tf.keras.activations.relu(x)
x = tf.add(x,x5)
print(x.shape)
x = tf.keras.layers.Conv2DTranspose(64,3,padding='valid',activation='softmax')(x)
print(x.shape)
x = tf.keras.layers.Conv2DTranspose(32,3,2,padding='same',activation='softmax')(x)
print(x.shape)
prediction = tf.keras.layers.Conv2DTranspose(3,3,padding='valid',activation='softmax')(x)
print(prediction.shape)
model = tf.keras.models.Model(inputs = input,
                              outputs = prediction)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ['acc'])
history = model.fit(da_train,
                    epochs = 10,
                    steps_per_epoch = train_step,
                    validation_data=da_test,
                    validation_steps = test_step)
model.save('./hh.h5')
plt.plot(history.epoch,history.history.get('loss'),label='loss')
plt.plot(history.epoch,history.history.get('acc'),label='acc')
plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.legend()
plt.show()