import numpy as np
import os
import glob
np.random.seed(1111)
def create_random_csv_files(fault_classes, number_of_files_in_each_class):
    os.mkdir("./random_data/")  # Make a directory to save created files.
    for fault_class in fault_classes:
        for i in range(number_of_files_in_each_class):
            data = np.random.rand(1024,)
            file_name = "./random_data/" + eval("fault_class") + "_" + "{0:03}".format(i+1) + ".csv" # This creates file_name
            np.savetxt(eval("file_name"), data, delimiter = ",", header = "V1", comments = "")
        print(str(eval("number_of_files_in_each_class")) + " " + eval("fault_class") + " files"  + " created.")
create_random_csv_files(["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"], number_of_files_in_each_class = 100)
files = np.sort(glob.glob("./random_data/*"))
print("Total number of files: ", len(files))
print("Showing first 10 files...")
files[:10]
print(files[0])
print(files[0][14:21])
import pandas as pd
import re            # To match regular expression for extracting labels
def data_generator(file_list, batch_size = 20):
    i = 0
    while True:
        if i*batch_size >= len(file_list):  # This loop is used to run the generator indefinitely.
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
            data = []
            labels = []
            label_classes = ["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"]
            for file in file_chunk:
                temp = pd.read_csv(open(file,'r')) # Change this line to read any other type of file
                data.append(temp.values.reshape(32,32,1)) # Convert column data to matrix like data with one channel
                pattern = "^" + eval("file[14:21]")      # Pattern extracted from file_name
                for j in range(len(label_classes)):
                    if re.match(pattern, label_classes[j]): # Pattern is matched against different label_classes
                        labels.append(j)  
            data = np.asarray(data).reshape(-1,32,32,1)
            labels = np.asarray(labels)
            yield data, labels
            i = i + 1
generated_data = data_generator(files, batch_size = 10)
num = 0
for data, labels in generated_data:
    print(data.shape, labels.shape)
    print(labels, "<--Labels")  # Just to see the lables
    print()
    num = num + 1
    if num > 5: break
import tensorflow as tf
def tf_data_generator(file_list, batch_size = 20):
    i = 0
    while True:
        if i*batch_size >= len(file_list):  
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
            data = []
            labels = []
            label_classes = tf.constant(["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"]) # This line has changed.
            for file in file_chunk:
                temp = pd.read_csv(open(file,'r'))
                data.append(temp.values.reshape(32,32,1)) 
                pattern = tf.constant(eval("file[14:21]"))  # This line has changed
                for j in range(len(label_classes)):
                    if re.match(pattern.numpy(), label_classes[j].numpy()):  # This line has changed.
                        labels.append(j)
            data = np.asarray(data).reshape(-1,32,32,1)
            labels = np.asarray(labels)
            yield data, labels
            i = i + 1
check_data = tf_data_generator(files, batch_size = 10)
num = 0
for data, labels in check_data:
    print(data.shape, labels.shape)
    print(labels, "<--Labels")
    print()
    num = num + 1
    if num > 5: break
batch_size = 15
dataset = tf.data.Dataset.from_generator(tf_data_generator,args= [files, batch_size],output_types = (tf.float32, tf.float32),
                                                output_shapes = ((None,32,32,1),(None,)))
num = 0
for data, labels in dataset:
    print(data.shape, labels.shape)
    print(labels)
    print()
    num = num + 1
    if num > 7: break
import shutil
fault_folders = ["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"]
for folder_name in fault_folders:
    os.mkdir(os.path.join("./random_data", folder_name))
for file in files:
    pattern = "^" + eval("file[14:21]")
    for j in range(len(fault_folders)):
        if re.match(pattern, fault_folders[j]):
            dest = os.path.join("./random_data/",eval("fault_folders[j]"))
            shutil.move(file, dest)
glob.glob("./random_data/*")
glob.glob("./random_data/Fault_1/*")[:10] # Showing first 10 files of Fault_1 folder
glob.glob("./random_data/Fault_3/*")[:10] # Showing first 10 files of Falut_3 folder
fault_1_files = glob.glob("./random_data/Fault_1/*")
fault_2_files = glob.glob("./random_data/Fault_2/*")
fault_3_files = glob.glob("./random_data/Fault_3/*")
fault_4_files = glob.glob("./random_data/Fault_4/*")
fault_5_files = glob.glob("./random_data/Fault_5/*")
from sklearn.model_selection import train_test_split
fault_1_train, fault_1_test = train_test_split(fault_1_files, test_size = 20, random_state = 5)
fault_2_train, fault_2_test = train_test_split(fault_2_files, test_size = 20, random_state = 54)
fault_3_train, fault_3_test = train_test_split(fault_3_files, test_size = 20, random_state = 543)
fault_4_train, fault_4_test = train_test_split(fault_4_files, test_size = 20, random_state = 5432)
fault_5_train, fault_5_test = train_test_split(fault_5_files, test_size = 20, random_state = 54321)
fault_1_train, fault_1_val = train_test_split(fault_1_train, test_size = 10, random_state = 1)
fault_2_train, fault_2_val = train_test_split(fault_2_train, test_size = 10, random_state = 12)
fault_3_train, fault_3_val = train_test_split(fault_3_train, test_size = 10, random_state = 123)
fault_4_train, fault_4_val = train_test_split(fault_4_train, test_size = 10, random_state = 1234)
fault_5_train, fault_5_val = train_test_split(fault_5_train, test_size = 10, random_state = 12345)
train_file_names = fault_1_train + fault_2_train + fault_3_train + fault_4_train + fault_5_train
validation_file_names = fault_1_val + fault_2_val + fault_3_val + fault_4_val + fault_5_val
test_file_names = fault_1_test + fault_2_test + fault_3_test + fault_4_test + fault_5_test
print("Number of train_files:" ,len(train_file_names))
print("Number of validation_files:" ,len(validation_file_names))
print("Number of test_files:" ,len(test_file_names))
batch_size = 10
train_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [train_file_names, batch_size], 
                                              output_shapes = ((None,32,32,1),(None,)),
                                              output_types = (tf.float32, tf.float32))

validation_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [validation_file_names, batch_size],
                                                   output_shapes = ((None,32,32,1),(None,)),
                                                   output_types = (tf.float32, tf.float32))

test_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [test_file_names, batch_size],
                                             output_shapes = ((None,32,32,1),(None,)),
                                             output_types = (tf.float32, tf.float32))
from tensorflow.keras import layers
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, activation = "relu", input_shape = (32,32,1)),
    layers.MaxPool2D(2),
    layers.Conv2D(32, 3, activation = "relu"),
    layers.MaxPool2D(2),
    layers.Flatten(),
    layers.Dense(16, activation = "relu"),
    layers.Dense(5, activation = "softmax")
])
model.summary()
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
steps_per_epoch = np.int(np.ceil(len(train_file_names)/batch_size))
validation_steps = np.int(np.ceil(len(validation_file_names)/batch_size))
steps = np.int(np.ceil(len(test_file_names)/batch_size))
print("steps_per_epoch = ", steps_per_epoch)
print("validation_steps = ", validation_steps)
print("steps = ", steps)
model.fit(train_dataset, validation_data = validation_dataset, steps_per_epoch = steps_per_epoch,
         validation_steps = validation_steps, epochs = 10)
test_loss, test_accuracy = model.evaluate(test_dataset, steps = 10)
print("Test loss: ", test_loss)
print("Test accuracy:", test_accuracy)
def create_prediction_set(num_files = 20):
    os.mkdir("./random_data/prediction_set")
    for i in range(num_files):
        data = np.random.randn(1024,)
        file_name = "./random_data/prediction_set/"  + "file_" + "{0:03}".format(i+1) + ".csv" # This creates file_name
        np.savetxt(eval("file_name"), data, delimiter = ",", header = "V1", comments = "")
    print(str(eval("num_files")) + " "+ " files created in prediction set.")
create_prediction_set(num_files = 55)
prediction_files = np.sort(glob.glob("./random_data/prediction_set/*"))
print("Total number of files: ", len(prediction_files))
print("Showing first 10 files...")
prediction_files[:10]
def generator_for_prediction(file_list, batch_size = 20):
    i = 0
    while i <= (len(file_list)/batch_size):
        if i == np.floor(len(file_list)/batch_size):
            file_chunk = file_list[i*batch_size:len(file_list)]
            if len(file_chunk)==0:
                break
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
        data = []
        for file in file_chunk:
            temp = pd.read_csv(open(file,'r'))
            data.append(temp.values.reshape(32,32,1)) 
        data = np.asarray(data).reshape(-1,32,32,1)
        yield data
        i = i + 1
pred_gen = generator_for_prediction(prediction_files,  batch_size = 10)
for data in pred_gen:
    print(data.shape)
batch_size = 10
prediction_dataset = tf.data.Dataset.from_generator(generator_for_prediction,args=[prediction_files, batch_size],
                                                 output_shapes=(None,32,32,1), output_types=(tf.float32))
steps = np.int(np.ceil(len(prediction_files)/batch_size))
predictions = model.predict(prediction_dataset,steps = steps)
print("Shape of prediction array: ", predictions.shape)
predictions
np.argmax(predictions, axis = 1)