import numpy as np 

import pandas as pd 

import tensorflow as tf



alphabet=pd.read_csv("../input/handwritten_data_785.csv")

images=alphabet.iloc[:,1:].values

labels=alphabet.iloc[:,0].values.ravel()



print("Data size",images.shape,"Label size",labels.shape)
images=images.astype(np.float)

images=np.multiply(images,1.0/255.0)
#do one hot encoding for labels

def one_hot(labels,num_unique_label=26):

    num_labels=labels.shape[0]

    index_offset=np.arange(num_labels)*num_unique_label

    labels_one_hot=np.zeros((num_labels,num_unique_label))

    labels_one_hot.flat[index_offset+labels.ravel()]

    return labels_one_hot   
new_labels=one_hot(labels)
#TODO

#Divide data set into training data and test data

#Maybe using cross-validation method



#Training:  use Neural Network --- Tensorflow