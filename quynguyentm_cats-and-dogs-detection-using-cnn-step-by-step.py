#Import required packages
#Ref1: https://tiendv.wordpress.com/2016/12/25/convolutional-neural-networks/
#Ref2: https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b
#Ref3: http://tflearn.org/models/dnn/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
import os
import cv2 #Package for image processing: Load image and resize...
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn #Package for training model CNN
from random import shuffle
from tqdm import tqdm #Package to show image loading processing
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
#Setup some values for model : Learning rate, Size of input image, Dir of test and train data
Learningrate=0.001
TRAIN_DIR="../input/DogvsCattrain"
TEST_DIR="../input/Dogvscattest"
Img_size=50
MODEL_NAME = 'dogs-vs-cats-convnet'
#Create one-hot matrix for label data
def create_label(image_name): #Image_nme with structure dog.1,cat.2
    word_label=image_name.split('.')[0] #Spilt by "." before extract 3 characters before "."
    if word_label =='cat': #N if wordlabel is dog, we will reuturn [1,0]
        return np.array([1,0])
    elif word_label =='dog': 
        return np.array([0,1])
create_label('dog.1')
#Def function to create training data
def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path=os.path.join(TRAIN_DIR,img) #extract all dir from trainingdata
        img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE) #Load Image with dir before and convert to matrix, apply grays for all image before
        img_data=cv2.resize(img_data,(Img_size,Img_size))
        training_data.append([np.array(img_data),create_label(img)]) #Append all Featrure and label
    
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
        
#Def function for testing data
def create_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path=os.path.join(TEST_DIR,img)
        img_num=img.split('.')[0]
        img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_data=cv2.resize(img_data,(Img_size,Img_size))
        testing_data.append([np.array(img_data),img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
#Load data
train_data=create_train_data()
test_data=create_test_data()
#Split data to train and test using train_test_split Sklearn
from sklearn.model_selection import train_test_split
train,test=train_test_split(train_data,test_size=0.2, random_state=42)
#Extract X_train, Y_train...
X_train = np.array([i[0] for i in train]).reshape(-1, Img_size, Img_size, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, Img_size, Img_size, 1)
y_test = [i[1] for i in test]
y_train
tf.reset_default_graph()
convnet = input_data(shape=[None, Img_size, Img_size, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu') #The important in neural net, dùng các cửa sổ trượt
#là các kenel, filter hay feature detector để lấy ra các đặc trưng trên mỗi vùng ảnh với kích thước của nó được khai báo như trên
convnet = max_pool_2d(convnet, 5) #Extract max value, đặc tính nổi trội nhất của vùng dữ liệu để làm giảm kích thước và tăng
#tính đại diện
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu') #Sau khi các tầng được phân tách và thực hiện conv_2d Và chọn ra max_pool 
#sẽ được kết nối lại với nhau
convnet = dropout(convnet, 0.8) #Loại bỏ việc học lẫn nhau giữa các neural
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=Learningrate, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
          validation_set=({'input': X_test}, {'targets': y_test}), 
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#Epoch số vòng thực hiện lại cả quá trình, Snapshot step- Số step sau mỗi lần snapshot
# Manually save model
model.save("C:/Anaconda3/envs/tensorflow/CatdectionPj/Catdetection.tfl")
# Load a model
model.load("C:/Anaconda3/envs/tensorflow/CatdectionPj/Catdetection.tfl")
#Testing the result of model after training
prediction = model.predict(X_test)[0]
#Test a sample data
path1=os.path.join("..input/Dogvscattest/anhtestmeo.jpg") #Dir of pic
#Load image and tranform to matrix before resize
sample=cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
sample=cv2.resize(sample,(Img_size,Img_size))
#Reshape to 50*50 like the model
sample.shape
data=sample.reshape(Img_size, Img_size,1)
#Prediction and test
prediction = model.predict([data])[0]
prediction
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(sample, cmap="gray")

print(f"cat: {prediction[0]}, dog: {prediction[1]}")
