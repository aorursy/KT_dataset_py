import os

import numpy as np

import cv2

import matplotlib.pyplot as plt

import pandas as pd

BASE_PATH = '../input/siim-isic-melanoma-classification'

hair_images =['ISIC_0078712','ISIC_0080817','ISIC_0082348','ISIC_0109869','ISIC_0155012','ISIC_0159568','ISIC_0164145','ISIC_0194550','ISIC_0194914','ISIC_0202023']

without_hair_images = ['ISIC_0015719','ISIC_0074268','ISIC_0075914','ISIC_0084395','ISIC_0085718','ISIC_0081956']

l = len(hair_images[:8])



fig = plt.figure(figsize=(20,30))



for i,image_name in enumerate(hair_images[:8]):

    

    

    image = cv2.imread(BASE_PATH + '/jpeg/train/' + image_name + '.jpg')

    image_resize = cv2.resize(image,(1024,1024))

    plt.subplot(l, 5, (i*5)+1)

    # Convert the original image to grayscale

    plt.imshow(cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.title('Original : '+ image_name)

    

    grayScale = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)

    plt.subplot(l, 5, (i*5)+2)

    plt.imshow(grayScale)

    plt.axis('off')

    plt.title('GrayScale : '+ image_name)

    

    # Kernel for the morphological filtering

    kernel = cv2.getStructuringElement(1,(17,17))

    

    # Perform the blackHat filtering on the grayscale image to find the hair countours

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    plt.subplot(l, 5, (i*5)+3)

    plt.imshow(blackhat)

    plt.axis('off')

    plt.title('blackhat : '+ image_name)

    

    # intensify the hair countours in preparation for the inpainting 

    ret,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    plt.subplot(l, 5, (i*5)+4)

    plt.imshow(threshold)

    plt.axis('off')

    plt.title('threshold : '+ image_name)

    

    # inpaint the original image depending on the mask

    final_image = cv2.inpaint(image_resize,threshold,1,cv2.INPAINT_TELEA)

    plt.subplot(l, 5, (i*5)+5)

    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.title('final_image : '+ image_name)

       

plt.plot()
def hair_remove(image):

    # convert image to grayScale

    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    

    # kernel for morphologyEx

    kernel = cv2.getStructuringElement(1,(17,17))

    

    # apply MORPH_BLACKHAT to grayScale image

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    

    # apply thresholding to blackhat

    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    

    # inpaint with original image and threshold image

    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)

    

    return final_image
for i,image_name in enumerate(hair_images[:5]):

    

    fig = plt.figure(figsize=(5,5))

    

    image = cv2.imread(BASE_PATH + '/jpeg/train/' + image_name + '.jpg')

    image_resize = cv2.resize(image,(512,512))

    plt.subplot(1, 2, 1)

    plt.imshow(cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.title('Original : '+ image_name)

    

    final_image = hair_remove(image_resize)

    plt.subplot(1, 2, 2)

    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.title('Hair Removed : '+ image_name)

    

    plt.plot()
import numpy as np, pandas as pd, os

import matplotlib.pyplot as plt, cv2

import tensorflow as tf, re, math
!ls /kaggle/input
PATH = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-test/512x512-test/'

IMGS = os.listdir(PATH)

print('There%i test images'%(len(IMGS)))
df = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/marking.csv')

df.rename({'image_id':'image_name'},axis=1,inplace=True)

df.head(5)
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

test.head(5)
# COMBINE TRAIN AND TEST TO ENCODE TOGETHER

cols = test.columns

comb = pd.concat([df[cols],test[cols]],ignore_index=True,axis=0).reset_index(drop=True)
cats = ['patient_id','sex','anatom_site_general_challenge'] 

for c in cats:

    comb[c],mp = comb[c].factorize()

    print(mp)

print('Imputing Age NaN count =',comb.age_approx.isnull().sum())

comb.age_approx.fillna(comb.age_approx.mean(),inplace=True)

comb['age_approx'] = comb.age_approx.astype('int')
# REWRITE DATA TO DATAFRAMES

df[cols] = comb.loc[:df.shape[0]-1,cols].values

test[cols] = comb.loc[df.shape[0]:,cols].values

# LABEL ENCODE TRAIN SOURCE

df.source,mp = df.source.factorize()

print(mp)
test.head(5)
def _bytes_feature(value):

  """Returns a bytes_list from a string / byte."""

  if isinstance(value, type(tf.constant(0))):

    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

  """Returns a float_list from a float / double."""

  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _int64_feature(value):

  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def serialize_example(feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7):

  feature = {

      'image': _bytes_feature(feature0),

      'image_name': _bytes_feature(feature1),

      'patient_id': _int64_feature(feature2),

      'sex': _int64_feature(feature3),

      'age_approx': _int64_feature(feature4),

      'anatom_site_general_challenge': _int64_feature(feature5),

      'source': _int64_feature(feature6),

      'target': _int64_feature(feature7)

  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()
# Write TFRecords - Test



def serialize_example2(feature0, feature1, feature2, feature3, feature4, feature5): 

  feature = {

      'image': _bytes_feature(feature0),

      'image_name': _bytes_feature(feature1),

      'patient_id': _int64_feature(feature2),

      'sex': _int64_feature(feature3),

      'age_approx': _int64_feature(feature4),

      'anatom_site_general_challenge': _int64_feature(feature5),

  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()
SIZE = 687

CT = len(IMGS)//SIZE + int(len(IMGS)%SIZE!=0)

for j in range(CT):

    print(); print('Writing TFRecord %i of %i...'%(j,CT))

    CT2 = min(SIZE,len(IMGS)-j*SIZE)

    with tf.io.TFRecordWriter('test%.2i-%i.tfrec'%(j,CT2)) as writer:

        for k in range(CT2):

            img = cv2.imread(PATH+IMGS[SIZE*j+k])

            img = hair_remove(img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors

            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

            name = IMGS[SIZE*j+k].split('.')[0]

            row = test.loc[test.image_name==name]

            example = serialize_example2(

                img, str.encode(name),

                row.patient_id.values[0],

                row.sex.values[0],

                row.age_approx.values[0],                        

                row.anatom_site_general_challenge.values[0])

            writer.write(example)

            if k%100==0: print(k,', ',end='')
# numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)

CLASSES = [0,1]



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    #if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

    #    numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)

    

def display_batch_of_images(databatch, predictions=None):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = label

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = example['image_name']

    return image, label # returns a dataset of (image, label) pairs



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
# INITIALIZE VARIABLES

IMAGE_SIZE= [512,512]; BATCH_SIZE = 32

AUTO = tf.data.experimental.AUTOTUNE

Test_FILENAMES = tf.io.gfile.glob('test*.tfrec')

print('There are %i test images'%count_data_items(Test_FILENAMES))
TRAINING_FILENAMES = Test_FILENAMES

training_dataset = get_training_dataset()

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)



display_batch_of_images(next(train_batch))