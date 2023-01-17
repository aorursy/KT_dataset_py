import tensorflow as tf
import os


tf.enable_eager_execution()
def data_example_proto(data, label):
    return tf.train.Example(features=tf.train.Features(
        feature={
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tostring()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()]))
            })).SerializeToString()


def tfrecord_from_folder(folder, output):
    with tf.python_io.TFRecordWriter(output) as writer:
        for folder, subfolders, files in tf.gfile.Walk(folder):
            
            label = os.path.basename(folder)
            for file in files:
                image_file = os.path.join(folder, file)
                
                try: image = tf.reshape(tf.image.decode_image(tf.read_file(os.path.join(folder, file))), [-1])
                except: continue
                
                writer.write(data_example_proto(image.numpy(), label))
tfrecord_from_folder('../input/notMNIST_large/notMNIST_large', 'notMNIST.tfrecords')
