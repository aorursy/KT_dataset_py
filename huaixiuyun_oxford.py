# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import xml.dom.minidom
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from lxml import etree
from matplotlib.patches import Rectangle
import time
import glob
import os
# for dirname, _, filenames in os.walk('.'):
#     for filename in filenames:
# #         if 'h5' in filename:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)])
!ls /kaggle/input/the-oxfordiiit-pet-dataset/images/images  | wc -l
# 显示有 7393个文件，但事实上只有 7390个 jpg文件
!ls /kaggle/input/the-oxfordiiit-pet-dataset/images/images | grep  'jpg' | wc -l
!ls /kaggle/input/the-oxfordiiit-pet-dataset/images/images | grep  -v 'jpg'
# 这3个文件不会用作训练
! ls /kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls | grep "xml" | wc -l
# 3686个 xml 文件，用作训练集标签
!ls /kaggle/input/the-oxfordiiit-pet-dataset/images/images | head
! ls /kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls | head
dom = xml.dom.minidom.parse("/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls/Abyssinian_1.xml") # or xml.dom.minidom.parseString(xml_string)
pretty_xml_as_string = dom.toprettyxml()
print(pretty_xml_as_string)
# name:  所属的大类别（也是这次分类的类别）;   xmin: 左下角点 x值; ymax: 右上角 y值......
all_image_paths = glob.glob("/kaggle/input/the-oxfordiiit-pet-dataset/images/images/*.jpg")
all_xml_paths = glob.glob("/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls/*.xml")
len(all_image_paths), len(all_xml_paths)
images_names = [ path.split('/')[-1].split('.jpg')[0] for path in all_image_paths ] 
#所有图片文件名，['scottish_terrier_57','Abyssinian_79','pug_4','miniature_pinscher_124','scottish_terrier_45',......]
xmls_names = [ path.split('/')[-1].split('.xml')[0] for path in all_xml_paths ]
#所有xml文件名 ['american_bulldog_173', 'beagle_129', 'Siamese_109', 'pug_110', 'Persian_106',......]

names = list(set(images_names)&set(xmls_names))
nonames = list(  set(images_names) -  set(xmls_names))
# 所有有对应xml标签文件的 文件名， 只有这些文件名开头的jpg文件才有xml文件，这些文件才会当作训练数据
#['havanese_115','staffordshire_bull_terrier_125','leonberger_177','yorkshire_terrier_162','japanese_chin_188',....])
len(names), names[:5], len(nonames), nonames[:5]
train_image_paths =  [image_path for image_path in all_image_paths if image_path.split('/')[-1].split('.jpg')[0] in names]
# 获取所有有xml标签文件的 jpg文件路径
train_image_paths[:5]
# 现在做个排序， 使得 image文件和 xml文件一一对应
train_image_paths.sort(key=lambda path: path.split('/')[-1].split('.jpg')[0])
all_xml_paths.sort(key=lambda path: path.split('/')[-1].split('.xml')[0])
print(train_image_paths[127], all_xml_paths[127] , '\n',train_image_paths[599], all_xml_paths[599])
print(train_image_paths[2375], all_xml_paths[2375] , '\n',train_image_paths[1864], all_xml_paths[1864])
def get_labels(label_path):
    # 从xml文件获取对应的标签
#     xml = open(r'{}'.format(label_path)).read()
    with open(label_path, 'r') as f:
        xml = f.read()
        sel = etree.HTML(xml)
        width = int(sel.xpath('//size/width/text()')[0])
        height = int(sel.xpath('//size/height/text()')[0])
        xmin = int(sel.xpath('//bndbox/xmin/text()')[0])
        ymin = int(sel.xpath('//bndbox/ymin/text()')[0])
        xmax = int(sel.xpath('//bndbox/xmax/text()')[0])
        ymax = int(sel.xpath('//bndbox/ymax/text()')[0])
    return [xmin/width, ymin/height, xmax/width, ymax/height]
RESIZE = 224
# 图片处理
def read_jpg(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def normalize(input_image, resize=RESIZE):
    input_image = tf.image.resize(input_image, [resize, resize])
    input_image = tf.cast(input_image, tf.float32)/127.5 - 1
    return input_image

@tf.function
def load_image(image_path):
    input_image = read_jpg(image_path)
    input_image = normalize(input_image)
    return input_image
labels = [get_labels(path) for path in all_xml_paths]
# labels:   [ [0.555, 0.18, 0.7083333333333334, 0.395], [0.192, 0.21, 0.768, 0.582],......]
labels[:2]
test_path = [  image_path for image_path in all_image_paths if image_path.split('/')[-1].split('.jpg')[0] not in names  ]
test_path[:5]
# xmin_labels, ymin_labels, xmax_label2, ymax_labels = list(zip(*labels))
# type(xmin_labels), len(xmin_labels), xmin_labels[:3]
#xmin_labels  存储了所有 image文件对应的 xmin坐标（进过了get_labels函数处理的，也就是相对坐标）， ymin_labels，xmax_label2，ymax_labels亦如是
index = np.random.permutation(len(train_image_paths))
# 打散顺序
train_images = np.array(train_image_paths)[index]
train_labels = np.array(labels)[index]
# xmin_labels = np.array(xmin_labels)[index]
# ymin_labels = np.array(ymin_labels)[index]
# xmax_label2 = np.array(xmax_label2)[index]
# ymax_labels = np.array(ymax_labels)[index]
train_images[:5], train_labels[:5]
# dataset_labels = tf.data.Dataset.from_tensor_slices(( xmin_labels, ymin_labels,  xmax_label2, ymax_labels))
dataset_labels = tf.data.Dataset.from_tensor_slices(train_labels)
dataset_images = tf.data.Dataset.from_tensor_slices(train_images)

dataset_images = dataset_images.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)   #处理图片
dataset_total = tf.data.Dataset.zip((dataset_images, dataset_labels))

dataset_test = tf.data.Dataset.from_tensor_slices(test_path)
dataset_total
# 分割 训练集和验证集

val_count = int(len(train_image_paths)*0.2)
train_count = len(train_image_paths) - val_count
dataset_train = dataset_total.skip(val_count)
dataset_val = dataset_total.take(val_count)
val_count
SCALE = 224 
BATCH_SIZE = 8
BUFFER_SIZE = 300
TRAIN_STEPS_PER_EPOCH = train_count // BATCH_SIZE
VALIDATION_STEPS = val_count // BATCH_SIZE

train_dataset = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = dataset_val.batch(BATCH_SIZE)

dataset_test = dataset_test.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_dataset = dataset_test.batch(8)
train_dataset, val_dataset, test_dataset
# 从 数据集中挑出一个看看显示是否正确
# def show_img(img, label, idx=0):
#     plt.imshow(tf.keras.preprocessing.image.array_to_img(img[idx]))
#     out1, out2, out3, out4 = label
#     xmin, ymin, xmax, ymax = (out1[idx]).numpy()*RESIZE, (out2[idx]).numpy()*RESIZE, (out3[idx]).numpy()*RESIZE, (out4[idx]).numpy()*RESIZE
#     rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
#     ax = plt.gca()
#     ax.axes.add_patch(rect)
def show_img(img, labels, idx=0):
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[idx]))
    label= labels[idx]
    xmin, ymin, xmax, ymax = (label[0]).numpy()*RESIZE, (label[1]).numpy()*RESIZE, (label[2]).numpy()*RESIZE, (label[3]).numpy()*RESIZE
    rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='red')
    ax = plt.gca()
    ax.axes.add_patch(rect)



for img, label in train_dataset.take(1):
    show_img(img, label, 7)  
for img, label in val_dataset.take(1):
    show_img(img, label, 3)      
# OK， 都没问题， 数据已经准备就绪。 
xception = tf.keras.applications.Xception(weights='imagenet', include_top=False,input_shape=(RESIZE, RESIZE, 3))
inputs = tf.keras.layers.Input(shape=(RESIZE, RESIZE, 3))

h_layer = xception(inputs)
h_layer = tf.keras.layers.GlobalAveragePooling2D()(h_layer)
h_layer = tf.keras.layers.Dense(2048, activation='relu')(h_layer)
h_layer = tf.keras.layers.Dense(256, activation='relu')(h_layer)

# xmin = tf.keras.layers.Dense(1)(h_layer)
# ymin = tf.keras.layers.Dense(1)(h_layer)
# xmax = tf.keras.layers.Dense(1)(h_layer)
# ymax = tf.keras.layers.Dense(1)(h_layer)

# predictions = [xmin, ymin, xmax, ymax]
predictions = tf.keras.layers.Dense(4)(h_layer)
# 因为标签跟原来不一样，这里不在生成 4个输出，而是最后一层 的Dense_unit = 4, 相当于输出了一个列表，这个列表包含4个值

model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse',metrics=['mae'])
EPOCHS = 50
history = model.fit(train_dataset, epochs=EPOCHS,steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=val_dataset)
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
model.save('/kaggle/working/detect_v3.h5')
plt.figure(figsize=(8, 24))
for img, _ in val_dataset.take(1):
    labels= model.predict(img)
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))
        xmin = labels[i][0]*224
        ymin = labels[i][1]*224
        xmax = labels[i][2]*224
        ymax = labels[i][3]*224
        rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='blue')
        ax = plt.gca()
        ax.axes.add_patch(rect)
        
plt.figure(figsize=(8, 24))
for img in test_dataset.take(1):
    labels= model.predict(img)
    print(labels)
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))
        xmin = labels[i][0]*224
        ymin = labels[i][1]*224
        xmax = labels[i][2]*224
        ymax = labels[i][3]*224
        rect = Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color='blue')
        ax = plt.gca()
        ax.axes.add_patch(rect)
        
!ls
