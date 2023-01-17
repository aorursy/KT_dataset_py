import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image, ImageOps

from scipy.spatial import cKDTree

from skimage.feature import plot_matches

from skimage.measure import ransac

from skimage.transform import AffineTransform

from six import BytesIO

import tensorflow as tf

import tensorflow_hub as hub

from six.moves.urllib.request import urlopen
import csv

import operator

import pandas as pd
#@title The images that will be processed by DELF

def download_and_resize_image(url, filename, new_width=256, new_height=256):

    try:

        response = urlopen(url)

    except Exception as e:

        print(e)

        #print('exception during urlopen')

    #print('url opened')

    image_data = response.read()

    image_data = BytesIO(image_data)

    pil_image = Image.open(image_data)

    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)

    pil_image_rgb = pil_image.convert('RGB')

    pil_image_rgb.save(filename, format='JPEG', quality=90)
def image_input_fn():

    filename_queue = tf.train.string_input_producer([IMAGE], shuffle=False)

    reader = tf.WholeFileReader()

    _, value = reader.read(filename_queue)

    image_tf = tf.image.decode_jpeg(value, channels=3)

    return tf.image.convert_image_dtype(image_tf, tf.float32)
#@title TensorFlow is not needed for this post-processing and visualization

def match_images(temp_dict):

    distance_threshold = 0.8

    

    # Read features.

    locations_1, descriptors_1 = temp_dict['image_1']

    num_features_1 = locations_1.shape[0]

    #print("Loaded image 1's %d features" % num_features_1)

    locations_2, descriptors_2 = temp_dict['image_2']

    num_features_2 = locations_2.shape[0]

    #print("Loaded image 2's %d features" % num_features_2)

    

    # Find nearest-neighbor matches using a KD tree.

    d1_tree = cKDTree(descriptors_1)

    _, indices = d1_tree.query(descriptors_2, distance_upper_bound=distance_threshold)

    

    # Select feature locations for putative matches.

    locations_2_to_use = np.array([locations_2[i,] for i in range(num_features_2) if indices[i] != num_features_1])

    locations_1_to_use = np.array([locations_1[indices[i],] for i in range(num_features_2) if indices[i] != num_features_1])

    

    # Perform geometric verification using RANSAC.

    try:

        _, inliers = ransac((locations_1_to_use, locations_2_to_use),AffineTransform,min_samples=3,residual_threshold=20,max_trials=1000)

    except:

        return 0

    

    #print ("************************************")

    #print (inliers)

    #print('#####################################')

    #print (type(inliers))

    #return(sum(inliers))

    if (inliers is None):

        return 0

    else:

        return sum(inliers)
test_dict = {}
#print(test_dict)

with open('../input/test.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')

    line_count = 0

    for row in csv_reader:

        if line_count == 0:

            print(f'Column names are {", ".join(row)}')

            line_count += 1

        elif not row[1].startswith('http'):

                  #print(line_count)

                  #print('bad url')

                  line_count += 1

        else:

            IMAGE = 'image.jpg'

            #print('good url')

            try:

                  try:

                        download_and_resize_image(str(row[1]),IMAGE)

                  

                  except:

                    print('exception during download')

                  tf.reset_default_graph()

                  tf.logging.set_verbosity(tf.logging.FATAL)

                  m = hub.Module('https://tfhub.dev/google/delf/1')

                  image_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3), name='input_image')

                  module_inputs = {'image': image_placeholder,'score_threshold': 100.0,'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],'max_feature_num': 1000,}

                  module_outputs = m(module_inputs, as_dict=True)

                  #image_input_fn([IMAGE])

                  image_tf = image_input_fn()

                  with tf.train.MonitoredSession() as sess:

                      image = sess.run(image_tf)

                      test_dict[str(row[0])] = sess.run([module_outputs['locations'], module_outputs['descriptors']],feed_dict={image_placeholder: image})

                      #print(test_dict[row[0]])

                      #print (line_count)

                      line_count += 1

                      if(line_count>=50):

                          break

            except:

                

                pass

                  #print('exception occured')

                  

                #pass

    print(f'Processed {line_count} lines from test set.')
index_dict = {}
with open('../input/index.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')

    line_count = 0

    for row in csv_reader:

        if line_count == 0:

            print(f'Column names are {", ".join(row)}')

            line_count += 1

        elif not row[1].startswith('http'):

                  #print(line_count)

                  #print('bad url')

                  line_count += 1

        else:

            IMAGE = 'image.jpg'

            try:

                  download_and_resize_image(str(row[1]),IMAGE)

                  tf.reset_default_graph()

                  tf.logging.set_verbosity(tf.logging.FATAL)

                  m = hub.Module('https://tfhub.dev/google/delf/1')

                  image_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3), name='input_image')

                  module_inputs = {'image': image_placeholder,'score_threshold': 100.0,'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],'max_feature_num': 1000,}

                  module_outputs = m(module_inputs, as_dict=True)

                  #image_input_fn([IMAGE])

                  image_tf = image_input_fn()

                  with tf.train.MonitoredSession() as sess:

                      image = sess.run(image_tf)

                      index_dict[str(row[0])] = sess.run([module_outputs['locations'], module_outputs['descriptors']],feed_dict={image_placeholder: image})

                      #print(test_dict[row[0]])

                      #print (line_count)

                      #print (row[0])

                      #print (index_dict[str(row[0])])

                      line_count += 1

                      if(line_count>=10):

                          break

            except:

                  pass

    print(f'Processed {line_count} lines from index set.')
calc_inliers_dict = {}

temp_dict = {}

index_inlier_dict = {}
query_ct = 1

for key, value in test_dict.items(): 

    #print ('Working on query number')

    #print (query_ct)

    for k, v in index_dict.items():

        temp_dict['image_1'] = value

        temp_dict['image_2'] = v

        #print ('temp dict')

        #print (temp_dict)

        num_inliers = match_images(temp_dict)

        index_inlier_dict[k] = num_inliers

        #print ('index_inlier_dict')

        #print (index_inlier_dict)

        sorted_index_inlier_dict = dict(sorted(index_inlier_dict.items(), key=operator.itemgetter(1),reverse=True))

    calc_inliers_dict[key] = sorted_index_inlier_dict

    query_ct += 1
#print(calc_inliers_dict)
query_images = []

index_matches = []

for key,value in calc_inliers_dict.items():

        #print('one step')

        query_images.append(key)

        matches = ''

        for k,v in value.items():

            if v > 20:

                if matches == '':

                    matches = k

                else:

                    matches = matches + ' ' + k

        index_matches.append(matches)
data_to_submit = pd.DataFrame({

    'id':query_images,

    'images':index_matches

})
data_to_submit.to_csv('csv_to_submit.csv', index = False)
#print(data_to_submit)
#print(query_images)
#print(index_matches)
'''

with open('../results.csv', 'a') as csvFile:

    writer = csv.writer(csvFile)

    writer.writerow(['id','images'])

    for key,value in calc_inliers_dict.items():

        print('one step')

        matches = ''

        for k,v in value.items():

            if v > 6:

                if matches == '':

                    matches = k

                else:

                    matches = matches + ' ' + k

        writer.writerow([key,matches])

                



csvFile.close()

'''
print("Done")