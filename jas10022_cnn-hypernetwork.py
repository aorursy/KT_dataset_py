from __future__ import absolute_import, division, print_function

from skimage import transform 

from skimage.color import rgb2gray

from PIL import Image

from resizeimage import resizeimage

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



import os

import pickle

import json

import tensorflow as tf

import numpy as np

import pandas as pd
# open open category to id file

with open('val2019.json') as f:

    values = json.load(f)



#open id to images file

with open('train2019.json') as f:

    images = json.load(f)



annotation_images = pd.DataFrame.from_dict(images["annotations"])

catagory_images = pd.DataFrame.from_dict(images["categories"])

images_id = pd.DataFrame.from_dict(images["images"])



train_data = pd.DataFrame({'ImageId':None,'Class':None,'Family':None,'Genus':None,'Kingdom':None,'Name':None,'Order':None,'Phylum':None}, index = [0])



scaled_features = catagory_images.copy()

col_names = ['class', 'family','kingdom','order','phylum']

features = catagory_images[col_names]

for i in range(0,7):

    labelencoder = LabelEncoder()

    features.values[:,i] = labelencoder.fit_transform(features.values[:,i])



scaled_features[col_names] = features



index = 0;

for t in annotation_images.category_id:

    current_info = pd.DataFrame(scaled_features.loc[t]).T

    

    train_data.loc[index] = [index, current_info["class"].values[0], current_info["family"].values[0], current_info["genus"].values[0], current_info["kingdom"].values[0], current_info["name"].values[0], current_info["order"].values[0], current_info["phylum"].values[0]]

    print(index)

    index += 1



train_data = train_data.assign(File_Name=images_id['file_name'].values)



train_data.to_csv(r'/upperNN.csv', index=False)
lower_train_data = pd.DataFrame({'FileName': images_id['file_name'],'ImageId': annotation_images['id'], 'CatagoryID':annotation_images['category_id']})



lower_train_data = lower_train_data.sort_values(by=['CatagoryID'])



lower_train_data.set_index(keys=['CatagoryID'], drop=False,inplace=True)



# get a list of names

numbers=lower_train_data['CatagoryID'].unique().tolist()



for number in numbers:

    data = lower_train_data.loc[lower_train_data['CatagoryID']==number]



    data.to_csv( str(number) + '.csv', index=False)
# now we can perform a lookup on a 'view' of the dataframe



ordered_features = scaled_features.sort_values(['class', 'family', 'genus', 'kingdom', 'order', 'phylum'])



features = ['class', 'family', 'genus', 'kingdom', 'order', 'phylum']



classt = scaled_features['class'].unique().tolist()

family = scaled_features['family'].unique().tolist()

genus = scaled_features['genus'].unique().tolist()

kingdom = scaled_features['kingdom'].unique().tolist()

order = scaled_features['order'].unique().tolist()

phylum = scaled_features['phylum'].unique().tolist()



for a in kingdom:

    for b in phylum:

        for c in classt:

            for d in order:

                for e in family:

                        current = ordered_features['id'].loc[(ordered_features['kingdom'] == a) & (ordered_features['phylum'] == b) & (ordered_features['class'] == c) & (ordered_features['order'] == d) & (ordered_features['family'] == e)]

                        if len(current) != 0:

                            print(len(current))

                            current.to_csv(str(a) + '_' + str(b) + '_' + str(c) + '_' + str(d) + '_' + str(e) + '.csv', index=False)

#this is how we can create a data file with all the images in a 1,75,75 shape 

#modify the for loop in order to use it and the allImages are all the images file names

#update the .open method to the directory of the train images then the + im

# the Images variable will contain all the picures each resized to 75 by 75



upperNN = pd.read_csv('upperNN .csv')



file_names = upperNN['File_Name']



Images = np.empty([1,75, 75])

i = 0

a = 0

for im in file_names:

        if i >= 5000:

            Images = Images[1:]

            output = open(str(a)+'.pkl', 'wb')

            pickle.dump(Images, output)

            output.close()            

            Images = np.empty([1,75, 75])

            a += 1

            i = 0

        img = Image.open(im, 'r').convert('LA')

        cover = resizeimage.resize_cover(img, [75, 75], validate=False)

        np_im = np.array(cover)

    

        pix_val_flat = np.array([x for sets in np_im for x in sets])

        train_data = pix_val_flat[:,0].astype('float64') /255

        train_data = np.resize(train_data, (1,75,75))

        

        Images = np.concatenate((Images,train_data))

        i += 1

        print(i)

        

Images = Images[1:]

output = open(str(a)+'.pkl', 'wb')

pickle.dump(Images, output)

output.close()

        

Images = np.empty([1,75, 75])

i = 0

a = 54



for num in range(54,65):

    if i == 5:

        Images = Images[1:]

        output = open(str(a)+'.pkl', 'wb')

        pickle.dump(Images, output)

        output.close()            

        Images = np.empty([1,75, 75])

        i = 0

        a += 1

    pkl_file = open(str(num) + '.pkl', 'rb')



    data1 = pickle.load(pkl_file)



    Images = np.concatenate((Images,data1))

    

    pkl_file.close()

    print(i)

    i += 1

Images = Images[1:]

output = open(str(a)+'.pkl', 'wb')

pickle.dump(Images, output)

output.close()
Images = np.empty([1,75, 75])

i = 0

for num in range(0,1010):

    current = pd.read_csv('Lower_NN_Data/' + str(num) + '.csv')

    for id in current['ImageId']:

        file_num = int(id / 25000)

        index = id - (file_num * 25000)

        if file_num == 0:

            im = data1[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 1:

            im = data2[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 2:

            im = data3[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 3:

            im = data4[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 4:

            im = data5[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 5:

            im = data6[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 6:

            im = data7[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 7:

            im = data8[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 8:

            im = data9[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 9:

            im = data10[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

        if file_num == 10:

            im = data11[index].reshape(1,75,75)

            Images = np.concatenate((Images,im))

    Images = Images[1:]

    output = open(str(i) + '.pkl', 'wb')

    pickle.dump(Images, output)

    output.close()

    Images = np.empty([1,75, 75])

    print(i)

    i += 1
tf.logging.set_verbosity(tf.logging.INFO)



def CNN(train_data, train_labels, eval_data, eval_labels, output_nodes_number, model_name, model_type):

      

    if model_type == "lower":

        #tensorflow model function for lowerNN

        def cnn_model(features, labels, mode):

            

            input_layer = tf.reshape(features["x"], [-1, 75, 75,1])

            

            # Convolutional Layer #1

            conv1 = tf.layers.conv2d(

                  inputs=input_layer,

                  filters=32,

                  kernel_size=[5, 5],

                  padding="same",

                  activation=tf.nn.relu)

            

              # Pooling Layer #1

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            

              # Convolutional Layer #2 and Pooling Layer #2

            conv2 = tf.layers.conv2d(

                  inputs=pool1,

                  filters=64,

                  kernel_size=[5, 5],

                  padding="same",

                  activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            

              # Dense Layer

            pool2_flat = tf.reshape(pool2, [-1,  18 * 18 * 64])

            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

            dropout = tf.layers.dropout(

                  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            

              # Logits Layer

            logits = tf.layers.dense(inputs=dropout, units=output_nodes_number)

            

            predicted_classes =tf.argmax(input=logits, axis=1)

            predictions = {

                        'class_ids': predicted_classes[:, tf.newaxis],

                        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),

                        'logits': logits,

                    }

            export_outputs = {

              'prediction': tf.estimator.export.PredictOutput(predictions)

              }

            if mode == tf.estimator.ModeKeys.PREDICT:  

                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

            

              # Calculate Loss (for both TRAIN and EVAL modes)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)            

              # Configure the Training Op (for TRAIN mode)

            if mode == tf.estimator.ModeKeys.TRAIN:

                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

                train_op = optimizer.minimize(

                    loss=loss,

                    global_step=tf.train.get_global_step())

                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            

              # Add evaluation metrics (for EVAL mode)

            eval_metric_ops = {

                  "accuracy": tf.metrics.accuracy(

                      labels=labels, predictions=predictions["class_ids"])

            }

            return tf.estimator.EstimatorSpec(

                  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    

    if model_type == "upper":

        #tensorflow model function for upperNN

        def cnn_model(features, labels, mode):

            

            input_layer = tf.reshape(features["x"], [-1, 75, 75, 1])

            

            # Convolutional Layer #1

            conv1 = tf.layers.conv2d(

                  inputs=input_layer,

                  filters=32,

                  kernel_size=[5, 5],

                  padding="same",

                  activation=tf.nn.relu)

            

              # Pooling Layer #1

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            

              # Convolutional Layer #2 and Pooling Layer #2

            conv2 = tf.layers.conv2d(

                  inputs=pool1,

                  filters=64,

                  kernel_size=[5, 5],

                  padding="same",

                  activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            

              # Dense Layer

            pool2_flat = tf.reshape(pool2, [-1, 18 * 18 * 64])

            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

            dropout = tf.layers.dropout(

                  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            

              # Logits Layer

            logits = tf.layers.dense(inputs=dropout, units=output_nodes_number)

            

            predicted_classes =tf.argmax(input=logits, axis=1)

            predictions = {

                        'class_ids': predicted_classes[:, tf.newaxis],

                        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),

                        'logits': logits,

                    }

            export_outputs = {

              'prediction': tf.estimator.export.PredictOutput(predictions)

              }

            if mode == tf.estimator.ModeKeys.PREDICT:  

                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

            

              # Calculate Loss (for both TRAIN and EVAL modes)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)            

            

              # Configure the Training Op (for TRAIN mode)

            if mode == tf.estimator.ModeKeys.TRAIN:

                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

                train_op = optimizer.minimize(

                    loss=loss,

                    global_step=tf.train.get_global_step())

                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            

              # Add evaluation metrics (for EVAL mode)

            eval_metric_ops = {

                  "accuracy": tf.metrics.accuracy(

                      labels=labels, predictions=predictions["class_ids"])

            }

            return tf.estimator.EstimatorSpec(

                  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    #This is where we need to load up the data for each group



    ModelDir = model_name

    # Create the Estimator

    

    run_config = tf.contrib.learn.RunConfig(

    model_dir=ModelDir,

    keep_checkpoint_max=1)

    

    cnn_classifier = tf.estimator.Estimator(

        model_fn=cnn_model, config=run_config)



    # Set up logging for predictions

    tensors_to_log = {"probabilities": "softmax_tensor"}



    logging_hook = tf.train.LoggingTensorHook(

        tensors=tensors_to_log, every_n_iter=50)



    # Train the model

    train_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x": train_data},

        y=train_labels,

        batch_size=100,

        num_epochs=None,

        shuffle=True)



    #change steps to 20000

    cnn_classifier.train(input_fn=train_input_fn, steps=2000, hooks=[logging_hook])



    # Evaluation of the neural network

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x": eval_data},

        y=eval_labels,

        num_epochs=1,

        shuffle=False)



    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)

    

        #instead of predicting on a test data set we will save the model

    #model_dir = cnn_classifier.export_savedmodel(

       # model_name,

        #serving_input_receiver_fn=serving_input_receiver_fn)



    return  ModelDir
def cnn_model_test(features, labels, mode):

            

            input_layer = tf.reshape(features["x"], [-1, 75, 75,1])

            

            # Convolutional Layer #1

            conv1 = tf.layers.conv2d(

                  inputs=input_layer,

                  filters=32,

                  kernel_size=[5, 5],

                  padding="same",

                  activation=tf.nn.relu)

            

              # Pooling Layer #1

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            

              # Convolutional Layer #2 and Pooling Layer #2

            conv2 = tf.layers.conv2d(

                  inputs=pool1,

                  filters=64,

                  kernel_size=[5, 5],

                  padding="same",

                  activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            

              # Dense Layer

            pool2_flat = tf.reshape(pool2, [-1,  18 * 18 * 64])

            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

            dropout = tf.layers.dropout(

                  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            

              # Logits Layer

            logits = tf.layers.dense(inputs=dropout, units=1)

            

            predicted_classes =tf.argmax(input=logits, axis=1)

            predictions = {

                        'class_ids': predicted_classes[:, tf.newaxis],

                        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),

                        'logits': logits,

                    }

            export_outputs = {

              'prediction': tf.estimator.export.PredictOutput(predictions)

              }

            if mode == tf.estimator.ModeKeys.PREDICT:  

                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

            

              # Calculate Loss (for both TRAIN and EVAL modes)

            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits = logits)

            

              # Configure the Training Op (for TRAIN mode)

            if mode == tf.estimator.ModeKeys.TRAIN:

                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

                train_op = optimizer.minimize(

                    loss=loss,

                    global_step=tf.train.get_global_step())

                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            

              # Add evaluation metrics (for EVAL mode)

            eval_metric_ops = {

                  "accuracy": tf.metrics.accuracy(

                      labels=labels, predictions=predictions["class_ids"])

            }

            return tf.estimator.EstimatorSpec(

                  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#just a start on how to automate creating all the models for the upper NN



#creating a basic model to start the model training

# this code chunk will auto run and create a model trained on all the data for the species classification

# this is all automated and may take a long time to create al 5 models as it has to go through

#25000 images 11 times as the data is split up inot 11 files for each model

# bellow is the code for Upper NN to give to ur dad to run

col_names = ['Class', 'Family','Kingdom','Order','Phylum']

#, 'Family','Kingdom','Order','Phylum'

a = 0

b = 1

output_nodes = 0

for cat in col_names:

    

    pkl_file = open('Data/UpperNN_data/64.pkl', 'rb')

    data1 = pickle.load(pkl_file)

    

    upperNN = pd.read_csv('Data/upperNN.csv')

    Labels = upperNN[cat][249999:265214]

    output_nodes = upperNN[cat].unique().size

    

    

    #this splits the data into training and val data for the model and also reshapes the label data

    X_train, X_val, y_train, y_val = train_test_split(data1, Labels, test_size = 0.05, random_state = 0)

    y_train = np.asarray(y_train).reshape((-1,1))

    y_val = np.asarray(y_val).reshape((-1,1))

    

    model_location = CNN(X_train,y_train,X_val,y_val,output_nodes, str(cat), "upper")



    ##need prediction values



    for i in range(54,64):

        

        pkl_file = open('Data/UpperNN_data/' + str(i) + '.pkl', 'rb')

        data1 = pickle.load(pkl_file)

    

        upperNN = pd.read_csv('Data/upperNN.csv')

        Labels = upperNN[cat][a * 25000:b * 25000]

        a += 1

        b += 1

    

        #this splits the data into training and val data for the model and also reshapes the label data

        X_train, X_val, y_train, y_val = train_test_split(data1, Labels, test_size = 0.05, random_state = 0)

        y_train = np.asarray(y_train).reshape((-1,1))

        y_val = np.asarray(y_val).reshape((-1,1))

                    

        #this session will open up any saved model created in directory and will run prediction on that

        # you can also train with it using the training lines

        with tf.Session() as sess:

          # Restore variables from disk.

            currentCheckpoint = tf.train.latest_checkpoint(cat)

            saver = tf.train.import_meta_graph(currentCheckpoint + ".meta")

            saver.restore(sess, currentCheckpoint)

            print("Model restored.")

                  

            sunspot_classifier = tf.estimator.Estimator(

            model_fn=cnn_model_test, model_dir=cat)

                

                # Set up logging for predictions

                # Log the values in the "Softmax" tensor with label "probabilities"

            tensors_to_log = {"probabilities": "softmax_tensor"}

            logging_hook = tf.train.LoggingTensorHook(

                tensors=tensors_to_log, every_n_iter=50)

                  

                  ## train here

            train_input_fn = tf.estimator.inputs.numpy_input_fn(

                           x={"x": X_train},

                           y=y_train,

                           batch_size=100,

                           num_epochs=None,

                           shuffle=True)

        

                #change steps to 20000

            sunspot_classifier.train(input_fn=train_input_fn, steps=2000)

            

                # Evaluation of the neural network

            eval_input_fn = tf.estimator.inputs.numpy_input_fn(

                    x={"x": X_val},

                    y=y_val,

                    num_epochs=1,

                    shuffle=False)

            

            eval_results = sunspot_classifier.evaluate(input_fn=eval_input_fn)

            print(eval_results)

    if b == 11:

        break
#this will be the lower NN code to run

#this code is very basic where it will look into the Data/Sorted_species files one by one as these

# files contains the groups for all the species and it will train one model for each of the groupings

# once read it will look at all the species ids and then look into the Data/Train_data file to get all 

# the picture data and store itinto train_data for each species

# the code will also take the labels from the Data/Lower_NN_Data files to create the Labels for the training

# then after the code splits the data created into training and validation data to train the model

output_nodes_number = 0



for filename in os.listdir("Data/Sorted_species"):

    if filename.endswith(".csv"):

        data = pd.read_csv("Data/Sorted_species/" + filename, names=['a'])

        output_nodes_number = data.size 

        name = os.path.splitext(filename)[0]

        train_data = np.empty([1,75, 75])

        Labels = np.empty([1])

        for species in data['a']:

                

            pkl_file = open("Data/Train_data/" + str(species) + '.pkl', 'rb')

            data1 = pickle.load(pkl_file)

            train_data = np.concatenate((train_data,data1))

            pkl_file.close()

             

            current_labels = pd.read_csv("Data/Lower_NN_Data/" + str(species) + '.csv')["CatagoryID"]

            Labels = np.concatenate((Labels,current_labels.values))

        

        Labels = Labels[1:]

        train_data = train_data[1:]

         

        labelencoder = LabelEncoder()

        Labels[0:] = labelencoder.fit_transform(Labels[0:])



        X_train, X_val, y_train, y_val = train_test_split(train_data, Labels, test_size = 0.20, random_state = 0)

        y_train = np.asarray(y_train).astype('int32').reshape((-1,1))

        y_val = np.asarray(y_val).astype('int32').reshape((-1,1))

        

        model_location = CNN(X_train,y_train,X_val,y_val,output_nodes_number, name, "lower")
# open open category to id file

with open('test2019.json') as f:

    test = json.load(f)

    

Test_Data = pd.DataFrame.from_dict(test["file_name","id"])



file_names = Test_data['file_name']



Images = np.empty([1,75, 75])

i = 0

for im in file_names:

        img = Image.open(im, 'r').convert('LA')

        cover = resizeimage.resize_cover(img, [75, 75], validate=False)

        np_im = np.array(cover)

    

        pix_val_flat = np.array([x for sets in np_im for x in sets])

        train_data = pix_val_flat[:,0].astype('float64') /255

        train_data = np.resize(train_data, (1,75,75))

        

        Images = np.concatenate((Images,train_data))

        i += 1

        print(i)

        

TestData = Images[1:]
col_names = ['Class', 'Family','Kingdom','Order','Phylum']

pred = np.empty([1])



with tf.Session() as sess:

    test_catagory = pd.DataFrame({'Class':None,'Family':None,'Kingdom':None,'Order':None,'Phylum':None}, index = [0])

    

    for cat in col_names:

        # Restore variables from disk.

        currentCheckpoint = tf.train.latest_checkpoint(cat)

        saver = tf.train.import_meta_graph(currentCheckpoint + ".meta")

        saver.restore(sess, currentCheckpoint)

        print("Model restored.")         

        sunspot_classifier = tf.estimator.Estimator(

            model_fn=cnn_model_test, model_dir=cat)

        # predict with the model and print results

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(

                x={"x": TestData},shuffle=False)

        pred_results = sunspot_classifier.predict(input_fn=pred_input_fn)



        pred_val = np.array([p['class_ids'] for p in pred_results]).squeeze()

        

        test_catagory[cat] = pred_val



    test_catagory = test_catagory[1:]

    

    for pred_val in range(0,len(test_catagory.index)):

        # Restore variables from disk.

        currentCheckpoint = tf.train.latest_checkpoint(test_catagory['Kingdom'][pred_val] + '_' + test_catagory['Phylum'][pred_val] 

                                                       + '_' + test_catagory['Class'][pred_val] + '_' + test_catagory['Order'][pred_val] + '_' 

                                                       + test_catagory['Family'][pred_val] + '.csv')

        saver = tf.train.import_meta_graph(currentCheckpoint + ".meta")

        saver.restore(sess, currentCheckpoint)

        print("Model restored.")         

        sunspot_classifier = tf.estimator.Estimator(

            model_fn=cnn_model_test, model_dir=cat)

        # predict with the model and print results

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(

                x={"x": TestData[pred_val]},shuffle=False)

        pred_results = sunspot_classifier.predict(input_fn=pred_input_fn)



        pred_value = np.array([p['class_ids'] for p in pred_results]).squeeze()

        

        np.concatenate((pred,pred_value))

        

pred = pred[1:]



output1 = pd.DataFrame({'Id':Test_Data['id'],'Prediction':pred})
