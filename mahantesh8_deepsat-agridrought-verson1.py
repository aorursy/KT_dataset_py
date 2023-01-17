import numpy as np
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
from skimage import color
!pip install mahotas
import mahotas as mt
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings("ignore")




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data_path="/kaggle/input/deepsat-sat6/X_train_sat6.csv"
train_label_path="/kaggle/input/deepsat-sat6/y_train_sat6.csv"
test_data_path="/kaggle/input/deepsat-sat6/X_test_sat6.csv"
test_lable_path="/kaggle/input/deepsat-sat6/y_test_sat6.csv"
def data_read(data_path, nrows):
    data=pd.read_csv(data_path, header=None, nrows=nrows)
    data=data.values ## converting the data into numpy array
    return data
##Read training data
train_data=data_read(train_data_path, nrows=500)
print("Train data shape:" + str(train_data.shape))

##Read training data labels
train_data_label=data_read(train_label_path,nrows=500)
print("Train data label shape:" + str(train_data_label.shape))
print()

##Read test data
test_data=data_read(test_data_path, nrows=100)
print("Test data shape:" + str(test_data.shape))


##Read test data labels
test_data_label=data_read(test_lable_path,nrows=100)
print("Test data label shape:" + str(test_data_label.shape))

#label converter
# [1,0,0,0,0,0]=building
# [0,1,0,0,0,0]=barren_land
# [0,0,1,0,0,0]=trees
# [0,0,0,1,0,0]=grassland
# [0,0,0,0,1,0]=road
# [0,0,0,0,0,1]=water


def label_conv(label_arr):
    labels=[]
    for i in range(len(label_arr)):
        
        if (label_arr[i]==[1,0,0,0,0,0]).all():
            labels.append("Building")  
            
        elif (label_arr[i]==[0,1,0,0,0,0]).all():  
            labels.append("Barren_land")  
            
        elif (label_arr[i]==[0,0,1,0,0,0]).all():
            labels.append("Tree") 
            
        elif (label_arr[i]==[0,0,0,1,0,0]).all():
            labels.append("Grassland")
            
        elif (label_arr[i]==[0,0,0,0,1,0]).all():
            labels.append("Road") 
            
        else:
            labels.append("Water")
    return labels
train_label_convert=label_conv(train_data_label)##train label conveter
test_label_convert=label_conv(test_data_label) ##test label converter


def data_visualization(data, label, n):
    ##data: training or test data
    ##lable: training or test labels
    ## n: number of data point, it should be less than or equal to no. of data points
    fig = plt.figure(figsize=(14, 14))
    ax = []  # ax enables access to manipulate each of subplots
    rows, columns=4,4
    for i in range(columns*rows):
        index=np.random.randint(1,n)
        img= data[index].reshape([28,28,4])[:,:,:3] ##reshape input data to rgb image
        ax.append( fig.add_subplot(rows, columns, i+1) ) # create subplot and append to ax
        ax[-1].set_title("Class:"+str(label[index]))  # set class
        plt.axis("off")
        plt.imshow(img)

    plt.subplots_adjust(wspace=0.1,hspace=0.5)
    plt.show()  # finally, render the plot
data_visualization(train_data,train_label_convert, n=500)
data_visualization(test_data,test_label_convert, n=100)

#  texture_features=["Angular Second Moment","Contrast","Correlation","Sum of Squares: Variance","Inverse Difference Moment",
#                    "Sum Average","Sum Variance","Sum Entropy","Entropy","Difference Variance","Difference Entropy",
#                    "Information Measure of Correlation 1","Information Measure of Correlation 2""Maximal Correlation Coefficient"]

#https://gogul09.github.io/software/texture-recognition #references for texture feature calculations

def feature_extractor(input_image_file):
    
        tex_feature=[]
        hsv_feature=[]
        ndvi_feature=[]
        arvi_feature=[]

        for df_chunk in pd.read_csv(input_image_file ,header=None,chunksize = 5000):

            df_chunk=df_chunk.astype("int32")
            data=df_chunk.values


            ################data for HSV and Texture feature##############
            img=data.reshape(-1,28,28,4)[:,:,:,:3]
            #############################################################

            ######################Data for NDVI and ARVI#################

            NIR=data.reshape(-1,28,28,4)[:,:,:,3]
            Red=data.reshape(-1,28,28,4)[:,:,:,2]
            Blue=data.reshape(-1,28,28,4)[:,:,:,0]
            #############################################################

            for i in range(len(data)):

                #######Texture_feature####################################
                textures = mt.features.haralick(img[i])
                ht_mean= textures.mean(axis=0)
                tex_feature.append(ht_mean)
                ##########################################################

                #######hsv_feature#########################################
                img_hsv = color.rgb2hsv(img[i]) # Image into HSV colorspace
                h = img_hsv[:,:,0] # Hue
                s = img_hsv[:,:,1] # Saturation
                v = img_hsv[:,:,2] # Value aka Lightness
                hsv_feature.append((h.mean(),s.mean(),v.mean()))
                ###########################################################

                ##########Calculation of NDVI Feature######################
                NDVI=(NIR[i]-Red[i])/(NIR[i]+Red[i])
                ndvi_feature.append(NDVI.mean())
                ############################################################

                ###################Calculation of ARVI#####################
                a_1=NIR[i] -(2*Red[i]-Blue[i])
                a_2=NIR[i] +(2*Red[i]+Blue[i])
                arvi=a_1/a_2
                arvi_feature.append(arvi.mean())
                #######################################################

        features=[]
        for i in range(len(tex_feature)):
            h_stack=np.hstack((tex_feature[i], hsv_feature[i], ndvi_feature[i], arvi_feature[i]))
            features.append(h_stack)
            
        return features
train_data_features=feature_extractor(train_data_path)
# saving train data features
feature=pd.DataFrame(train_data_features, columns=["feature"+ str(i) for i in range(len(train_data_features[0]))])
feature.to_csv("train_feature_deepstat_6.csv")
#test data features extraction
test_data_features=feature_extractor(test_data_path)
feature_test=pd.DataFrame(test_data_features, columns=["feature"+ str(i) for i in range(len(train_data_features[0]))])
feature_test.to_csv("test_feature_deepsat_6.csv")
from sklearn.preprocessing import StandardScaler 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from time import time
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def data_read_(data_path):
    df=pd.read_csv(data_path, index_col=[0])
    return df

def label_read(data_path):
    df=pd.read_csv(data_path, header=None)
    return df


train_feature_deepstat_6=data_read_(data_path="/kaggle/input/extracted-feature/train_feature_deepstat_6.csv")
train_label=label_read(data_path=train_label_path)
print("Training data shape: ",train_feature_deepstat_6.shape)
print("Training label shape: ",train_label.shape)
train_feature_deepstat_6.head()
train_label.head()
test_feature_deepsat_6=data_read_(data_path="/kaggle/input/extracted-feature/test_feature_deepsat_6.csv")
test_label=label_read(data_path=test_lable_path)
print("Training data shape: ",test_feature_deepsat_6.shape)
print("Training label shape: ",test_label.shape)
test_feature_deepsat_6.head()
test_label.head()
sc=StandardScaler()
#fit the training data
fit=sc.fit(train_feature_deepstat_6)
##transform the train and test data
train_data_stn=fit.transform(train_feature_deepstat_6)
test_data_stn=fit.transform(test_feature_deepsat_6)
model=Sequential()

#layer1
model.add(Dense(units=50,input_shape=(train_data_stn.shape[1],),use_bias=True))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))

#layer2
model.add(Dense(units=50, use_bias=True))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))

#layer3
model.add(Dense(units=6, activation="softmax"))


##ADD early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
#tensorboard=TensorBoard(log_dir='logs/{}'.format(time()))

#compile the model
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

#model.fit(train_data_stn, train_label.values, validation_split=0.15, batch_size=512, epochs=500,callbacks=[es, mc,tensorboard]) 
model.fit(train_data_stn, train_label.values, validation_split=0.15, batch_size=512, epochs=500,callbacks=[es]) 
Accuracy_on_test_data=model.evaluate(test_data_stn, test_label.values)[1]
print("Accuracy on test data: ",Accuracy_on_test_data)
#label converter
# [1,0,0,0,0,0]=building
# [0,1,0,0,0]=barren_land
# [0,0,1,0,0,0]=tree
# [0,0,0,1,0,0]=grassland
# [0,0,0,0,1,0]=road
# [0,0,0,0,0,1]=water


##Building confusion matrix

y_pred=model.predict_classes(test_data_stn)
y_true=np.argmax(test_label.values, axis=1)
cm=confusion_matrix(y_target=y_true, y_predicted=y_pred)

plot_confusion_matrix(cm,class_names=["Building","Barren_land","Tree","Grassland","Road","Water"],figsize=(6,6) )
plt.show()
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)


dic={}
precision_=[]
recall_=[]
precision_macro_average_=[]
for label in range(6):
    precision_.append(precision(label, cm))
    recall_.append(recall(label, cm))
    
dic["Precision"]= precision_
dic["Recall"]= recall_

plt.figure(figsize=(6,6))
ax=sns.heatmap(pd.DataFrame(dic, index=["building","barren_land","tree","grassland","road","water"]),annot=True,cbar=False)
plt.yticks(rotation=0)
ax.xaxis.tick_top() # x axis on top

plt.show()