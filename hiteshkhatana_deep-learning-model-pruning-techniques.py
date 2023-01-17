import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import time
from datetime import datetime
import zipfile

tf.__version__
#Cifar10 Dataset

data = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = data.load_data()
#normalizing images

train_images = train_images/255.0
test_images = test_images/255.0
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
#converting rgb images to grayscale images to reduce processing time.

def gray_img(images):
    images_grayscale = np.zeros(images.shape[:-1])
    for i in range(0,images.shape[0]):
        images_grayscale[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        
    return images_grayscale
train_images = gray_img(train_images)
test_images = gray_img(test_images)
train_images.shape
# CNN model

model = keras.Sequential()
model.add(keras.Input(shape = (32,32)))
model.add(keras.layers.Reshape(target_shape = (32,32,1)))
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(
  train_images,
  train_labels,
  epochs=8,
  validation_split=0.1,
)
def factors(model , model_name , zip_name):
    start = datetime.now()
    _, model_accuracy = model.evaluate(
        test_images, test_labels, verbose=0)
    end = datetime.now()
    
    t = end-start
    
    model.save(model_name)
    
    size_without_zip = os.path.getsize(model_name)
    
    size_with_zip = get_gzipped_model_size(model_name , zip_name)
    
    return t.total_seconds() , model_accuracy, size_without_zip , size_with_zip
def get_gzipped_model_size(file , zip_name):
    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
        
    return os.path.getsize(zip_name)
time_1  , accuracy_1 ,size_without_zip_1 ,size_with_zip_1 = factors(model , 'Base Model.h5' ,'Base Model.zip')

name = 'basemodel'

print('Accuracy of %s is :%.2f' % (name ,accuracy_1*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_1 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_1 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_1, 'sec')
# tensorflow model optimization kit

!pip install -q tensorflow-model-optimization
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
# Define model for pruning.

model_without_pruning = keras.models.load_model('Base Model.h5')
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model_without_pruning,**pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

model_for_pruning.summary()
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]
  
model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
time_2  , accuracy_2 ,size_without_zip_2 ,size_with_zip_2 = factors(model_for_pruning , 'Model Pruned.h5' ,'Model Pruned.zip')
name = 'basemodel'

print('Accuracy of %s is :%.2f' % (name ,accuracy_1*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_1 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_1 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_1, 'sec')
print('\n')

name = 'pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_2*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_2 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_2 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_2, 'sec')
print('\n')
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

model_for_export.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
time_3  , accuracy_3 ,size_without_zip_3 ,size_with_zip_3 = factors(model_for_export , 'Model stripped.h5' ,'Model Stripped.zip')
name = 'basemodel'

print('Accuracy of %s is :%.2f' % (name ,accuracy_1*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_1 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_1 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_1, 'sec')
print('\n')

name = 'pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_2*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_2 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_2 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_2, 'sec')
print('\n')

name = 'striped model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_3*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_3 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_3 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_3, 'sec')
print('\n')
def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer , **pruning_params)
    return layer

# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
# to the layers of the model.
model_for_dense_layer_pruning = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning_to_dense,
)

model_for_dense_layer_pruning.summary()
model_for_dense_layer_pruning.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
time_4  , accuracy_4 ,size_without_zip_4 ,size_with_zip_4 = factors(model_for_dense_layer_pruning , 'Dense layers pruned.h5' ,'Dense layers pruned.zip')
name = 'dense layer pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_4*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_4 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_4 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_4, 'sec')
print('\n')
name = 'basemodel'

print('Accuracy of %s is :%.2f' % (name ,accuracy_1*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_1 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_1 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_1, 'sec')
print('\n')

name = 'pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_2*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_2 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_2 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_2, 'sec')
print('\n')

name = 'striped model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_3*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_3 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_3 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_3, 'sec')
print('\n')

name = 'dense layer pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_4*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_4 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_4 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_4, 'sec')
print('\n')
model_stripped_pruned_denselayer = tfmot.sparsity.keras.strip_pruning(model_for_dense_layer_pruning)

model_stripped_pruned_denselayer.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
time_5  , accuracy_5 ,size_without_zip_5 ,size_with_zip_5 = factors(model_stripped_pruned_denselayer , 'Dense layers pruned and stripped.h5' ,'Dense layers pruned and stripped.zip')
name = 'basemodel'

print('Accuracy of %s is :%.2f' % (name ,accuracy_1*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_1 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_1 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_1, 'sec')
print('\n')

name = 'pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_2*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_2 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_2 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_2, 'sec')
print('\n')

name = 'striped model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_3*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_3 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_3 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_3, 'sec')
print('\n')

name = 'dense layer pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_4*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_4 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_4 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_4, 'sec')
print('\n')

name = 'dense layer pruned and stripped model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_5*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_5 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_5 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_5, 'sec')
print('\n')
base = keras.models.load_model("Base Model.h5")
base.weights
for layer in base.layers:
    print(layer.name)
from scipy.stats import rankdata

for k in [.20]:
    ranks = {}
    for l in base.layers:
        data = base.get_layer(l.name).get_weights()
        
        if len(data) == 0:
            continue
        else:
            temp = []
            w = data
            for i in range(0,np.array(data).shape[0]):
                ranks[l]=(rankdata(np.abs(w[i]),method='dense') - 1).astype(int).reshape(w[i].shape)
                lower_bound_rank = np.ceil(np.max(ranks[l])*k).astype(int)
                ranks[l][ranks[l]<=lower_bound_rank] = 0
                ranks[l][ranks[l]>lower_bound_rank] = 1
                w[i] = w[i]*ranks[l]
                temp.append(np.array(w[i]))
            print(temp)
            base.get_layer(l.name).set_weights(np.array(temp))
base.save("Weight pruned 20%.h5")
new_model = keras.models.load_model("Weight pruned 20%.h5")
new_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

time_6  , accuracy_6 ,size_without_zip_6 ,size_with_zip_6 = factors(new_model , 'Weight pruned 20%.h5' ,'Weight pruned 20%.zip')

name = 'weight pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_6*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_6 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_6 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_6, 'sec')
print('\n')

name = 'basemodel'

print('Accuracy of %s is :%.2f' % (name ,accuracy_1*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_1 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_1 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_1, 'sec')
base = keras.models.load_model("Base Model.h5")
base.weights
from numpy import linalg as LA


for k in [.10 , .20 , .30 , .40]:
    ranks = {}
    for l in base.layers:
        data = base.get_layer(l.name).get_weights()
        
        if len(data) == 0:
            continue
        else:
            temp = []
            w = data
            for i in range(0,np.array(data).shape[0]):
                norm = LA.norm(w[i],axis=0)
                norm = np.tile(norm,(w[i].shape[0],1))
                ranks[l]=(rankdata(norm,method='dense') - 1).astype(int).reshape(norm.shape)
                lower_bound_rank = np.ceil(np.max(ranks[l])*k).astype(int)
                ranks[l][ranks[l]<=lower_bound_rank] = 0
                ranks[l][ranks[l]>lower_bound_rank] = 1
                w[i] = w[i]*ranks[l]
                temp.append(np.array(w[i]))
            print(temp)
            base.get_layer(l.name).set_weights(np.array(temp))
base.save("Model neuron pruned 40.h5")
new_model2 = keras.models.load_model("Model neuron pruned 40.h5")
new_model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

time_7  , accuracy_7 ,size_without_zip_7 ,size_with_zip_7 = factors(new_model2 , 'Model neuron pruned 40.h5' ,'Model neuron pruned 40.zip')

name = 'basemodel'

print('Accuracy of %s is :%.2f' % (name ,accuracy_1*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_1 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_1 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_1, 'sec')
print('\n')

name = 'pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_2*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_2 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_2 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_2, 'sec')
print('\n')

name = 'striped model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_3*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_3 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_3 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_3, 'sec')
print('\n')

name = 'dense layer pruned model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_4*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_4 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_4 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_4, 'sec')
print('\n')

name = 'dense layer pruned and stripped model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_5*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_5 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_5 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_5, 'sec')
print('\n')


name = 'pruned 20 model'

print('Accuracy of %s is :%.2f' % (name ,accuracy_6*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_6 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_6 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_6, 'sec')
print('\n')


name = 'neuron pruned'

print('Accuracy of %s is :%.2f' % (name ,accuracy_7*100) ,'%')
print('Size of %s without Zip is :' % (name) , size_without_zip_7 ,'Bytes')
print('Size of %s with Zip is :' % (name), size_with_zip_7 ,'Bytes')
print('Time taken for evaluating %s is :' % (name) ,time_7, 'sec')
names = ['basemodel' , 'pruned(tf optimization kit)' , 'pruned-stripped' ,'pruned(only dense layers)','pruned(dense)-stripped' ,'weight-pruned-20%','neuron-pruned']
Accuracy = [accuracy_1 , accuracy_2,accuracy_3,accuracy_4,accuracy_5,accuracy_6,accuracy_7]
Size_without_zip = [size_without_zip_1,size_without_zip_2,size_without_zip_3,size_without_zip_4,size_without_zip_5,size_without_zip_6,size_without_zip_7]
Size_with_zip = [size_with_zip_1,size_with_zip_2,size_with_zip_3,size_with_zip_4,size_with_zip_5,size_with_zip_6,size_with_zip_7]
Time_for_eval = [time_1,time_2,time_3,time_4,time_5,time_6,time_7]
import pandas as pd
ar = []

for j in range(0,len(names)):
    k = []
    for i in [names ,Accuracy , Size_without_zip , Size_with_zip , Time_for_eval]:
        k.append(i[j])
    ar.append(k)
result = pd.DataFrame(ar, columns=  ['Names' , 'Accuracy' , 'Size_without_zip' , 'Size_with_zip' ,'Time_for_eval'])
import matplotlib.pyplot as plt
def plotting_result(Y):
    fig = plt.figure(figsize = (15,6)) 
    # creating the bar plot
    plt.barh(result['Names'] ,result[Y],) 
    #plt.xlabel("Models")
    plt.xlabel(Y)
    plt.title("%s with different actions on model" % (Y))
    plt.show() 
plotting_result('Accuracy')
plotting_result('Size_without_zip')
plotting_result('Size_with_zip')
plotting_result('Time_for_eval')
