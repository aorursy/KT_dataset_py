import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import os

%matplotlib inline
# 8 int64 values

my_integers=tf.train.Feature(int64_list=tf.train.Int64List(value=np.arange(8)))

my_integers
# A single float value

my_floats=tf.train.Feature(float_list=tf.train.FloatList(value=[3.14]))

my_floats
# A range of bytes. Note that strings must be .encode()'ed

txt='This string must be encoded'

my_txt=tf.train.Feature(bytes_list=tf.train.BytesList(value=[txt.encode()]))

my_txt
features=tf.train.Features(feature={

    'integers': my_integers,

    'pi': my_floats,

    'description': my_txt})

features
tf_record = tf.train.Example(features=features)

tf_record
fname='example1.tfrecord'

with tf.io.TFRecordWriter(fname) as writer:

    writer.write(tf_record.SerializeToString())

print("Size of {} is {}bytes".format(fname, os.path.getsize(fname)))
dataset = tf.data.TFRecordDataset(fname)

raw_example = next(iter(dataset)) # only one example in this file

parsed = tf.train.Example.FromString(raw_example.numpy())

parsed
# strings must be decoded

parsed.features.feature['description'].bytes_list.value[0].decode() 
parsed.features.feature['integers'].int64_list.value[:] # get all data
parsed.features.feature['integers'].int64_list.value[5] # get a single value
parsed.features.feature['pi'].float_list.value[0] # get the value
parsed.features.feature['pi'].float_list.value[:] # get the value as a list
import hashlib

from io import BytesIO

from PIL import Image, ImageFont, ImageDraw

ARTAXOR_PATH = '/kaggle/input/arthropod-taxonomy-orders-object-detection-dataset/'



pickles='/kaggle/input/starter-arthropod-taxonomy-orders-data-exploring/'

objectdf=pd.read_pickle(pickles+'ArTaxOr_objects.pkl')

labels=pd.read_pickle(pickles+'ArTaxOr_labels.pkl')

objectdf.sample(5)
# Fetch attribution string from image EXIF data

def get_attribution(file):

    with Image.open(file) as img:

        exif_data = img._getexif()

    s='Photo: unknown'

    if exif_data is not None:

        if 37510 in exif_data:

            if len(exif_data[37510]) > 0:

                s = exif_data[37510][8:].decode('ascii')

        if 315 in exif_data:

            if len(exif_data[315]) > 0:

                s = 'Photo: ' + exif_data[315]

    return s



# Create example for TensorFlow Object Detection API

def create_tf_example(imagedf, longest_edge=1024):  

    fname = ARTAXOR_PATH+imagedf.file.iloc[0]

    filename=fname.split('/')[-1] # exclude path

    by = get_attribution(fname)

    img = Image.open(fname, "r")

    # resize image if larger that longest edge while keeping aspect ratio

    if max(img.size) > longest_edge:

        img.thumbnail((longest_edge, longest_edge), Image.ANTIALIAS)

    height = img.size[1] # Image height

    width = img.size[0] # Image width

    buf= BytesIO()

    img.save(buf, format= 'JPEG') # encode to jpeg in memory

    encoded_image_data= buf.getvalue()

    image_format = b'jpeg'

    source_id = filename.split('.')[0]

    license = 'CC BY-NC-SA 4.0'

    # A hash of the image is used in some frameworks

    key = hashlib.sha256(encoded_image_data).hexdigest()   

    # object bounding boxes 

    xmins = imagedf.left.values # List of normalized left x coordinates in bounding box (1 per box)

    xmaxs = imagedf.right.values # List of normalized right x coordinates in bounding box

    ymins = imagedf.top.values # List of normalized top y coordinates in bounding box (1 per box)

    ymaxs = imagedf.bottom.values # List of normalized bottom y coordinates in bounding box

    # List of string class name & id of bounding box (1 per box)

    object_cnt = len(imagedf)

    classes_text = []

    classes = []

    for i in range(object_cnt):

        classes_text.append(imagedf.label.iloc[i].encode())

        classes.append(1+imagedf.label_idx.iloc[i])

    # unused features from Open Image 

    depiction = np.zeros(object_cnt, dtype=int)

    group_of = np.zeros(object_cnt, dtype=int)

    occluded = imagedf.occluded.values #also Pascal VOC

    truncated = imagedf.truncated.values # also Pascal VOC

    # Pascal VOC

    view_text = []

    for i in range(object_cnt):

        view_text.append('frontal'.encode())

    difficult = np.zeros(object_cnt, dtype=int)



    tf_record = tf.train.Example(features=tf.train.Features(feature={

        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),

        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),

        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),

        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source_id.encode()])),

        'image/license': tf.train.Feature(bytes_list=tf.train.BytesList(value=[license.encode()])),

        'image/by': tf.train.Feature(bytes_list=tf.train.BytesList(value=[by.encode()])),

        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),

        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode()])),

        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),

        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),

        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),

        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),

        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),

        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),

        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),

        'image/object/depiction': tf.train.Feature(int64_list=tf.train.Int64List(value=depiction)),

        'image/object/group_of': tf.train.Feature(int64_list=tf.train.Int64List(value=group_of)),

        'image/object/occluded': tf.train.Feature(int64_list=tf.train.Int64List(value=occluded)),

        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),

        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult)),

        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=view_text))

    }))

    return tf_record
sample_file='ArTaxOr/Lepidoptera/002b37ac08e1.jpg'

imagedf=objectdf[objectdf.file == sample_file]

tfr=create_tf_example(imagedf)

fname='./image_ex1.tfrecord'

with tf.io.TFRecordWriter(fname) as writer:

    writer.write(tfr.SerializeToString())

print("Size of {} is {}kbytes".format(fname, os.path.getsize(fname)//1024))
# Some helper functions to draw image with object boundary boxes

fontname = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

font = ImageFont.truetype(fontname, 40) if os.path.isfile(fontname) else ImageFont.load_default()



def bbox(img, xmin, ymin, xmax, ymax, color, width, label, score):

    draw = ImageDraw.Draw(img)

    xres, yres = img.size[0], img.size[1]

    box = np.multiply([xmin, ymin, xmax, ymax], [xres, yres, xres, yres]).astype(int).tolist()

    txt = " {}: {}%" if score >= 0. else " {}"

    txt = txt.format(label, round(score, 1))

    ts = draw.textsize(txt, font=font)

    draw.rectangle(box, outline=color, width=width)

    if len(label) > 0:

        if box[1] >= ts[1]+3:

            xsmin, ysmin = box[0], box[1]-ts[1]-3

            xsmax, ysmax = box[0]+ts[0]+2, box[1]

        else:

            xsmin, ysmin = box[0], box[3]

            xsmax, ysmax = box[0]+ts[0]+2, box[3]+ts[1]+1

        draw.rectangle([xsmin, ysmin, xsmax, ysmax], fill=color)

        draw.text((xsmin, ysmin), txt, font=font, fill='white')



def plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by):

    for i in range(len(xmin)):

        color=labels.color[class_label[i]-1]

        bbox(img, xmin[i], ymin[i], xmax[i], ymax[i], color, 5, classes[i].decode(), -1)

    plt.setp(axes, xticks=[], yticks=[])

    axes.set_title(by)

    plt.imshow(img)
# load tfrecord

fname='image_ex1.tfrecord'

dataset = tf.data.TFRecordDataset(fname)

img_example = next(iter(dataset)) 

img_parsed = tf.train.Example.FromString(img_example.numpy())

# only extract features we will actually use

xmin=img_parsed.features.feature['image/object/bbox/xmin'].float_list.value[:]

xmax=img_parsed.features.feature['image/object/bbox/xmax'].float_list.value[:]

ymin=img_parsed.features.feature['image/object/bbox/ymin'].float_list.value[:]

ymax=img_parsed.features.feature['image/object/bbox/ymax'].float_list.value[:]

by=img_parsed.features.feature['image/by'].bytes_list.value[0].decode()

classes=img_parsed.features.feature['image/object/class/text'].bytes_list.value[:]

class_label=img_parsed.features.feature['image/object/class/label'].int64_list.value[:]

img_encoded=img_parsed.features.feature['image/encoded'].bytes_list.value[0]
fig = plt.figure(figsize=(10,10))

axes = axes = fig.add_subplot(1, 1, 1)

img = Image.open(BytesIO(img_encoded))

plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by)
def create_tf_example2(imagedf, longest_edge=1024):  

    # Filename of the image (full path is useful when there are multiple image directories)

    fname = ARTAXOR_PATH+imagedf.file.iloc[0]

    filename=fname.split('/')[-1] # exclude path

    by = get_attribution(fname)

    img = Image.open(fname, "r")

    source_id = filename.split('.')[0]

    # resize image if larger that longest edge while keeping aspect ratio

    if max(img.size) > longest_edge:

        img.thumbnail((longest_edge, longest_edge), Image.ANTIALIAS)

    image_data = np.asarray(img)

    # storing shape will make it easy to reconstruct image later

    image_shape = np.array(image_data.shape)

    # convert to float

    image_data = image_data.reshape(image_data.shape[0]*image_data.shape[1]*image_data.shape[2])

    image_data = image_data.astype(float)/255. # normalize to [0,1]

    # object bounding boxes 

    xmins = imagedf.left.values # List of normalized left x coordinates in bounding box (1 per box)

    xmaxs = imagedf.right.values # List of normalized right x coordinates in bounding box

    ymins = imagedf.top.values # List of normalized top y coordinates in bounding box (1 per box)

    ymaxs = imagedf.bottom.values # List of normalized bottom y coordinates in bounding box

    # List of string class name & id of bounding box (1 per box)

    classes_text = []

    classes = []

    for i in range(len(imagedf)):

        classes_text.append(imagedf.label.iloc[i].encode())

        classes.append(1+imagedf.label_idx.iloc[i])



    tf_record = tf.train.Example(features=tf.train.Features(feature={

        'image/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape)),

        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),

        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source_id.encode()])),

        'image/by': tf.train.Feature(bytes_list=tf.train.BytesList(value=[by.encode()])),

        'image/data': tf.train.Feature(float_list=tf.train.FloatList(value=image_data)),

        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),

        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),

        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),

        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),

        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),

        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))

    }))

    return tf_record
sample_file2='ArTaxOr/Hymenoptera/ab30b4c2f70c.jpg'

imagedf=objectdf[objectdf.file == sample_file2]

tfr2 = create_tf_example2(imagedf)

fname2 = 'image_ex2.tfrecord'

with tf.io.TFRecordWriter(fname2) as writer:

    writer.write(tfr2.SerializeToString())

print("Size of {} is {}kbytes".format(fname2, os.path.getsize(fname2)//1024))
dataset2 = tf.data.TFRecordDataset(fname2)

img_example2 = next(iter(dataset2)) 

img_parsed2 = tf.train.Example.FromString(img_example2.numpy())

# extract features

xmin=img_parsed2.features.feature['image/object/bbox/xmin'].float_list.value[:]

xmax=img_parsed2.features.feature['image/object/bbox/xmax'].float_list.value[:]

ymin=img_parsed2.features.feature['image/object/bbox/ymin'].float_list.value[:]

ymax=img_parsed2.features.feature['image/object/bbox/ymax'].float_list.value[:]

by=img_parsed2.features.feature['image/by'].bytes_list.value[0].decode()

classes=img_parsed2.features.feature['image/object/class/text'].bytes_list.value[:]

class_label=img_parsed2.features.feature['image/object/class/label'].int64_list.value[:]

img_shape=img_parsed2.features.feature['image/shape'].int64_list.value[:]

img_data=img_parsed2.features.feature['image/data'].float_list.value[:]
image2=np.array(img_data).reshape(img_shape) # reshape

image2=image2*255. # scale back to [0, 255] and convert to int

image2=image2.astype(int)

img=Image.fromarray(np.uint8(image2))

fig = plt.figure(figsize=(10,10))

axes = axes = fig.add_subplot(1, 1, 1)

plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by)
filelist=pd.read_pickle(pickles+'ArTaxOr_filelist.pkl')

filelist=filelist.sample(frac=1)

filelist.head()
%%time

import contextlib2



def open_sharded_tfrecords(exit_stack, base_path, num_shards):

    tf_record_output_filenames = [

        '{}-{:05d}-of-{:05d}.tfrecord'.format(base_path, idx, num_shards)

        for idx in range(num_shards)

        ]

    tfrecords = [

        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))

        for file_name in tf_record_output_filenames

    ]

    return tfrecords



num_shards=50

output_filebase='./ArTaxOr'



with contextlib2.ExitStack() as tf_record_close_stack:

    output_tfrecords = open_sharded_tfrecords(tf_record_close_stack, output_filebase, num_shards)

    for i in range(len(filelist)):

        ldf=objectdf[objectdf.id == filelist.id.iloc[i]].reset_index()

        tf_record = create_tf_example(ldf, longest_edge=1280)

        output_shard_index = i % num_shards

        output_tfrecords[output_shard_index].write(tf_record.SerializeToString())
!ls -lh ArTaxOr*.tfrecord
labels=pd.read_pickle(pickles+'ArTaxOr_labels.pkl')

pbfile=open('./ArTaxOr.pbtxt', 'w') 

for i in range (len(labels)): 

    pbfile.write('item {{\n id: {}\n name:\'{}\'\n}}\n\n'.format(i+1, labels.name[i])) 

pbfile.close()
fname='./ArTaxOr-00029-of-00050.tfrecord' 

dataset3 = tf.data.TFRecordDataset(fname)

fig = plt.figure(figsize=(16,18))

idx=1

for raw_record in dataset3.take(6):

    axes = fig.add_subplot(3, 2, idx)

    example = tf.train.Example()

    example.ParseFromString(raw_record.numpy())

    xmin=example.features.feature['image/object/bbox/xmin'].float_list.value[:]

    xmax=example.features.feature['image/object/bbox/xmax'].float_list.value[:]

    ymin=example.features.feature['image/object/bbox/ymin'].float_list.value[:]

    ymax=example.features.feature['image/object/bbox/ymax'].float_list.value[:]

    by=example.features.feature['image/by'].bytes_list.value[0].decode()

    classes=example.features.feature['image/object/class/text'].bytes_list.value[:]

    class_label=example.features.feature['image/object/class/label'].int64_list.value[:]

    img_encoded=example.features.feature['image/encoded'].bytes_list.value[0]

    img = Image.open(BytesIO(img_encoded))

    plot_img(img, axes, xmin, ymin, xmax, ymax, classes, class_label, by)

    idx=idx+1