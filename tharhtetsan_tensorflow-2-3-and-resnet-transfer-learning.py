#/kaggle/input/state-farm-distracted-driver-detection/sample_submission.csv
#/kaggle/input/state-farm-distracted-driver-detection/driver_imgs_list.csv
#/kaggle/input/state-farm-distracted-driver-detection/imgs/train/c4/img_16261.jpg


import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance
tf.__version__
sample_path = "/kaggle/input/state-farm-distracted-driver-detection/sample_submission.csv"
imgs_list_path = "/kaggle/input/state-farm-distracted-driver-detection/driver_imgs_list.csv"
train_path = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
driver_imgs_list = pd.read_csv(imgs_list_path)
driver_imgs_list.head()
os.listdir(train_path)
def pair_sort(className,values):
    for j in range(0,len(className)-1):
        for i in range(0,len(className)-1):
            if values[i] > values[i+1]:
                temp =  values[i+1]
                values[i+1] = values[i]
                values[i] = temp

                N_temp =  className[i+1]
                className[i+1] = className[i]
                className[i] = N_temp
    
    return className,values
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')

class_names = np.unique(driver_imgs_list['classname'])
class_image_list = [len(driver_imgs_list[driver_imgs_list['classname'] == current_class]) for current_class in class_names]

class_names,class_image_list=  pair_sort(class_names,class_image_list)

#plt.figure()
plt.suptitle('Number of images per Class')
plt.bar(class_names,class_image_list,color=(0.2, 0.4, 0.6, 0.6))
plt.show()
from matplotlib.pyplot import figure
sub_names = np.unique(driver_imgs_list['subject'])
sub_image_list = [len(driver_imgs_list[driver_imgs_list['subject'] == current_sub]) for current_sub in sub_names]
sub_names,sub_image_list=  pair_sort(sub_names,sub_image_list)

figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

y_pos = np.arange(len(sub_names))
# Create horizontal bars
plt.barh(y_pos, sub_image_list,color=(0.2, 0.4, 0.6, 0.6))
 
# Create names on the y-axis
plt.yticks(y_pos,sub_names )
plt.suptitle('Number of images per subject')

# Show graphic
plt.show()
img_width,img_height = (256,256)
model_input_shape = (img_width,img_height,3)
batch_size = 16
input_image = (img_width,img_height)

def load_image(path):
    read_path = train_path+"/"+path
    image = Image.open(read_path)
    image = image.resize(input_image)
    
    return np.asarray(image)
def show_images(image_ids,class_names):
    pixels = [load_image(path) for path in image_ids]
    
    num_of_images = len(image_ids)
    
    fig, axes = plt.subplots(
        1, 
        num_of_images, 
        figsize=(5 * num_of_images, 5 * num_of_images),
        
    )
   
    
    for i, image_pixels in enumerate(pixels):
        axes[i].imshow(image_pixels)
        axes[i].axis("off")
        axes[i].set_title(class_names[i])
sub_names_imgs = [ current_class+"/"+driver_imgs_list[driver_imgs_list['classname'] == current_class]['img'].values[0] for current_class in class_names]

show_images(sub_names_imgs[:5],class_names[:5])
show_images(sub_names_imgs[5:],class_names[5:])

train_path = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
test_path = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"
x_train = []
y_train = []

x_val = []
y_val = []


split_rate = 0.8
for current_class in class_names:
    select_df = driver_imgs_list[driver_imgs_list['classname'] == current_class ]
    image_list = select_df['img'].values
    train_amount = int(len(image_list)*split_rate)
    train_list = image_list[:train_amount]
    val_list = image_list[train_amount:]
    
    for filename in train_list:
        x_train.append(load_image(current_class+"/"+filename))
        y_train.append(current_class.replace('c',''))

    for filename in val_list:
        x_val.append(load_image(current_class+"/"+filename))
        y_val.append(current_class.replace('c',''))

x_train = np.asarray(x_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
x_val = np.asarray(x_val)
y_val =tf.keras.utils.to_categorical(y_val, num_classes=10)
print("Train x Shape: ",x_train.shape)
print("Test x Shape: ",x_val.shape)

print("Train y Shape: ",y_train.shape)
print("Test y Shape: ",y_val.shape)
base_model  = tf.keras.applications.resnet.ResNet50(include_top = False,
                                                  weights = 'imagenet',
                                                  input_shape = model_input_shape)
base_model.summary()
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)

output =tf.keras.layers.Dense(units = len(class_names),activation = tf.nn.softmax)(x)
model = tf.keras.models.Model(inputs=base_model.inputs, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False),
              metrics=['accuracy'])

model.summary()
num_epochs = 50
def lr_schedule(epoch,lr):
    # Learning Rate Schedule

    lr = lr
    total_epochs = num_epochs

    check_1 = int(total_epochs * 0.9)
    check_2 = int(total_epochs * 0.8)
    check_3 = int(total_epochs * 0.6)
    check_4 = int(total_epochs * 0.4)

    if epoch > check_1:
        lr *= 1e-4
    elif epoch > check_2:
        lr *= 1e-3
    elif epoch > check_3:
        lr *= 1e-2
    elif epoch > check_4:
        lr *= 1e-1

    print("[+] Current Lr rate : {} ".format(lr))
    return lr
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
history = model.fit(
      x = x_train,y=y_train,
      validation_data=(x_val,y_val),
      steps_per_epoch=16,
      batch_size = 8,
      epochs=num_epochs,
    
    callbacks = [lr_callback],
      verbose=1)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].set_title('Training Loss')
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])

ax[1].set_title('Validation Loss')
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])