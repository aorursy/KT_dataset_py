#The external libraries the that will be used to help in the project
import numpy as np # A linear algebra library for python which has many data structures for example;numpy arrays which are much more faster than normal python lists
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt # A library that makes crearing graphs/plots.. much easier
import seaborn as sns #statistical data visualization



dataset = pd.read_csv('../input/forest-cover-type-dataset/covtype.csv') #load The dataset
print(dataset.shape) #show the shape of the data set,returns a tuple dataset[0]=number of data entries and dataset[1] is the number of features
                     # can be 3d for other datasets, forexample;(X,Y,Z) where X is the number of data entries ,Y could be a timestamp ,Z number of features in the timestamp
dataset.head(10) #A preview of the first 10 instances in the dataset
dataset.info() #More information about the datatypes of the dataset
for column in dataset: #for each column/feature in the dataset see the unique values and the number of there occurences 
    print(dataset[str(column)].value_counts())
    
#from the following we can deduce that the featue 'Soil_Type14' is ussles as its almost constant
dataset.median() # The median value for each feature
dataset.describe(include='all') #show statistics for each feature in tha data set 

#count: the total number of values for each feature (not N/A) it can be seen that currenly no feature is missing since they all have same count
#mean: the mean of all the values of each feature
#std: the standard deviation of each feature (it is the average of how far/close are all the values to each other 
#     a low std will indicate that most of the values are near the mean value)
#min: the smallest value each feature in the dataset
#25%,50%,75% :It describes the distribution of teh data set values for each feature. 50 should be The median value. 25, 75 are the border of the upper/lower quarter of the data.
#max: the largest value each feature in the dataset
for column in dataset:  #Get a more visual idea about the distribution of the values of each feature
    dataset[str(column)].hist() #get the values of the current feature
    print("Feature :",column) #print The name of the feature
    plt.show()# show the histogram of the value distribution of the current feature

dataset.corr() # The relation between each feature and another based on  the dataset A correlation of 1 means that the features are highly 
#related ,when one value increase the other one also increases in the same direction,A corrlation of 0 means that the features are not related at all
#A corelation of  -1 (inversly correlated) means that when the value of a feature increases the other increases also but in the opposite direction(decreases)
f = plt.figure(figsize=(50, 40)) #create a new figure of size 50x40
plt.matshow(dataset.corr(), fignum=f.number,vmin=-1, vmax=1) # plot the correlation matrix using the pandas correlation matrix with minimum value=-1 and max =1
plt.xticks(range(dataset.shape[1]), dataset.columns, fontsize=10, rotation=45)# plot the X axis using the column names
plt.yticks(range(dataset.shape[1]), dataset.columns, fontsize=10)# plot the X axis using the column names
cb = plt.colorbar() #show the color bar which represents the values for each color
cb.ax.tick_params(labelsize=10) #show the number values for each color
plt.title('Correlation Matrix', fontsize=25)# Add a tittle to the figure
#get the names of columns with high correlation between them
high_correlation=[] #an empty python list to store the features with high correlation between them
correlation_matrix=dataset.corr()# the correlation matrix from pandas
for column_1 in correlation_matrix: #iterate through the corrleation matrix
    for column_2 in correlation_matrix:
        if(column_1!=column_2):# not including the correlation between a feature and itself
            if(abs(correlation_matrix[column_1][column_2])>0.6 and (column_2,column_1,correlation_matrix[column_1][column_2])not in high_correlation):#checking if the features have high correlation and also discluding duplicates
                high_correlation.append((column_1,column_2,correlation_matrix[column_1][column_2]))#adding the features with high corelation between tham with the values of the correlation as a tuple to the list
                
for i in high_correlation:#prining the high correlation beweenfeatures with the correlation value
    print(i)
import tensorflow as tf
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
features=dataset.iloc[:,:54].to_numpy().astype('float32')
print(features.shape)

labels=dataset.iloc[:,54].to_numpy().astype('float32')
print(labels.shape)

y_binary = to_categorical(labels)
print(y_binary.shape)
print(y_binary[0])
model=Sequential()
model.add(Dense(128,input_dim=features[0].shape[0]))
model.add(Dense(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(units=8, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002, 0.5),metrics=['accuracy'])
model.summary()
model.fit(x=features, y=y_binary, batch_size=64, epochs=10, verbose=1)