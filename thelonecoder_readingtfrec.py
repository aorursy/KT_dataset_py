import tensorflow as tf
import os
import matplotlib.pyplot as plt
path = '../input/melanoma-256x256'
for item in os.listdir(path):
    print(item)
record_file = path + '/train01-2071.tfrec'
raw_dataset = tf.data.TFRecordDataset(record_file)
features = {
      'image': tf.io.FixedLenFeature([], tf.string),
      'image_name': tf.io.FixedLenFeature([],tf.string),
      'patient_id': tf.io.FixedLenFeature([], tf.int64),
      'sex': tf.io.FixedLenFeature([], tf.int64),
      'age_approx': tf.io.FixedLenFeature([], tf.int64),
      'anatom_site_general_challenge':tf.io.FixedLenFeature([], tf.int64),
      'diagnosis':tf.io.FixedLenFeature([], tf.int64),
      'target': tf.io.FixedLenFeature([], tf.int64)
  }
def _parse_function(example):
  # parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example, features)

# use map to apply this operation to each element of dataset
parsed_dataset = raw_dataset.map(_parse_function)
type(parsed_dataset)
for sample in parsed_dataset:
    image = tf.io.decode_image(sample['image'], dtype=tf.dtypes.uint8)
    image_name = sample['image_name']
    sex = sample['sex']
    age_approx = sample['age_approx']
    anatom_site_general_challenge = sample['anatom_site_general_challenge']
    diagnosis = sample['diagnosis']
    target = sample['target']
    
    
    print(f'image_name = {image_name}')
    print(f'sex = {sex}')
    print(f'age_approx = {age_approx}')
    print(f'anatom_site_general_challenge = {anatom_site_general_challenge}')
    print(f'diagnosis = {diagnosis}')
    print(f'target = {target}')
    
    plt.imshow(image)
    break