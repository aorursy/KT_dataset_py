import os
import cv2
import tensorflow as tf
random_seed = 42
IMG_SIZE_LIST = [[192,192], [224,224], [331,331], [512,512]]
NUM_IMAGES = 798
DS_NAME_LIST = [
    'oxford_flower_102', 
]
for DS_NAME in DS_NAME_LIST:
    for IMG_SIZE in IMG_SIZE_LIST:
        tfrecord_path = os.path.join(DS_NAME, 'tfrecords-jpeg-{}x{}'.format(IMG_SIZE[0], IMG_SIZE[1]))

        if not os.path.exists(tfrecord_path):
            os.makedirs(tfrecord_path)
        if not DS_NAME.startswith('inaturalist'):
            ds_path = os.path.join('/kaggle/input/externaldatasettpuflower', DS_NAME, DS_NAME)
        else:
            ds_path = os.path.join('/kaggle/input/externaldatasettpuflower', 'inaturalist', 'inaturalist', '{}'.format(DS_NAME.split('list')[1]))
        
        num = 0
        total = 0
        output_path = os.path.join(tfrecord_path, "{:0>2d}-{}x{}-{}.tfrec".format(num, IMG_SIZE[0], IMG_SIZE[1], NUM_IMAGES))
        writer = tf.io.TFRecordWriter(output_path)

        for d in os.listdir(ds_path):
            img_dir = os.path.join(ds_path, d)
            label = int(d.split(']')[0][1:])

            for img in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img)
                
                image = tf.io.read_file(img_path)
                image=tf.image.decode_jpeg(image,channels=3)
                image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0], IMG_SIZE[1])
                image_data = tf.io.encode_jpeg(image).numpy()
                
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                        'class': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                ))

                writer.write(example.SerializeToString())
                total += 1

                if ((total + 1) % NUM_IMAGES) == 0:
                    writer.close()
                    num += 1
                    output_path = os.path.join(tfrecord_path,"{:0>2d}-{}x{}-{}.tfrec".format(num, IMG_SIZE[0], IMG_SIZE[1], NUM_IMAGES))
                    writer = tf.io.TFRecordWriter(output_path)

        writer.close()
