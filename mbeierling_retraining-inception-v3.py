# First I needed to sort the photos into folders. 

# I named them "invasive" and "noninvasive". 

# I used the train_labels.csv file to do that:



import pandas

import os

labels = pandas.read_csv('train_labels.csv')



for index, row in labels.iterrows():

    is_invasive = row['invasive'] == 1

    name = str(row['name'])

    if is_invasive:

        # move file with that name into 'invasive' subfolder

        os.rename(name + '.jpg', 'invasive/' + name + '.jpg')

    else:

        # move file with that name into 'noninvasive' subfolder

        os.rename(name + '.jpg', 'noninvasive/' + name + '.jpg')
# To actually create the submission file, 

# we need to classify the test images 

# and write the predictions into a .csv 

# using the format seen in sample_submission.csv.



import os, sys

import pandas



import tensorflow as tf



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# change this as you see fit

folder_path = sys.argv[1]



# create predictions-table

predictions_list = [];



# get a list of all image-files in folder

images_list = os.listdir(folder_path)



# Unpersists graph from file

with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:

    graph_def = tf.GraphDef()

    graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')



with tf.Session() as sess:

    # Feed the image_data as input to the graph and get first prediction

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')



    for image_path in images_list:

        try:

            # Read in the image_data

            image_data = tf.gfile.FastGFile(folder_path + image_path, 'rb').read()



            predictions = sess.run(softmax_tensor, \

                    {'DecodeJpeg/contents:0': image_data})



            # invasive is the first value, so we access index 0 of the node

            predictions_list.append([image_path[:-4], predictions[0][0]])



            print('%s (score = %.5f)' % (image_path[:-4], predictions[0][0]))



        except:

            print('%s is not a jpg!' % (image_path))



    predictions_table = pandas.DataFrame(predictions_list, columns=['name', 'invasive'])



    predictions_table.to_csv('submission_invasive.csv', index=False)
