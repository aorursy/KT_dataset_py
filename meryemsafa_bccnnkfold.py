from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
import tensorflow as tf
import cv2 #opencv
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

%load_ext tensorboard
import datetime
!rm -rf ./logs/ 

!mkdir /kaggle/working/histories/
!mkdir /kaggle/working/Models/
!ls

no_angles = 360
url = '/kaggle/input/mias-mammography/all-mias/'

def save_dictionary(path,data):
        print('saving catalog...')
        #open('u.item', encoding="utf-8")
        import json
        with open(path,'w') as outfile:
            json.dump(str(data), fp=outfile)
        # save to file:
        print(' catalog saved')

def read_image():
    import cv2
    info = {}
    for i in range(322):
        if i<9:
            image_name='mdb00'+str(i+1)
        elif i<99:
            image_name='mdb0'+str(i+1)
        else:
            image_name = 'mdb' + str(i+1)
        # print(image_name)
        image_address= url+image_name+'.pgm'
        img = cv2.imread(image_address, 0)
        # print(i)
        img = cv2.resize(img, (64,64))   #resize image

        rows, cols = img.shape
        info[image_name]={}
        for angle in range(no_angles):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)    #Rotate 0 degree
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            info[image_name][angle]=img_rotated
    return (info)

import os #Operating System
import sys #System

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))    

def read_lable():
    filename = url+'Info.txt'
    text_all = open(filename).read()
    #print(text_all)
    lines=text_all.split('\n')
    info={}
    for line in lines:
        words=line.split(' ')
        if len(words)>3:
            if (words[3] == 'B'):
                info[words[0]] = {}
                for angle in range(no_angles):
                    info[words[0]][angle] = 0
            if (words[3] == 'M'):
                info[words[0]] = {}
                for  angle in range(no_angles):
                    info[words[0]][angle] = 1
    return (info)

class Model:

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.model = Sequential()
    
    def createModel2(self):
        from keras.layers import Conv2D, MaxPooling2D
        self.model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', tf.keras.metrics.AUC(num_thresholds=200,
    curve="ROC",summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    label_weights=None), tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
), tf.keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
),])
        return self.model
        
    
    def createModel1(self):
        self.model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (64, 64, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size = 3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        self.model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        self.model.add(Conv2D(128, kernel_size = 4, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation='softmax'))
        # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(num_thresholds=200,
    curve="ROC",summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    label_weights=None), tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
), tf.keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
),])
        return self.model

    def createModel(self):
        #model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(self.rows, self.cols, 1)))
        self.model.add(Activation('relu'))
        
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        
        self.model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
        self.model.add(Activation('relu'))
        
        self.model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
        self.model.add(Activation('relu'))
        
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        
        #self.model.add(Activation('softmax'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        #self.model.add(Activation('tanh'))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        #self.model.add(Activation('softmax'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        #self.model.add(Activation('softmax'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(num_thresholds=200,
    curve="ROC",summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    label_weights=None), tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
), tf.keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
),])
        #self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Accuracy()])#,    tf.keras.metrics.CosineSimilarity(), tf.keras.metrics.LogCoshError()])
        return self.model

    #def TrainModel(self, model, x_train, y_train, epochs, counter):
    def TrainModel(self, x_train, y_train, x_test ,y_test , epochs, counter):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0,write_graph=True, write_images=False)
        history = self.model.fit(x_train, y_train,epochs=epochs, batch_size=32, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
        save_dictionary('/kaggle/working/histories/history'+str(counter)+'.dat', history.history)
        #print(history.history.keys())
        """plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show() """
       
        print('YAY')
        listOfKeys = list()
        for i in history.history.keys():
            listOfKeys.append(i)
        #for i in range(5):
        #    print(listOfKeys[i])
        for st in listOfKeys:
            print(st,end ='\n')
        for i in range(5):
            plt.plot(history.history[listOfKeys[i]])
            plt.plot(history.history[listOfKeys[i+4]])            
            plt.title('model '+listOfKeys[i])
            plt.ylabel(listOfKeys[i])
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
    
    def Classify(self, image_address):
        import cv2
        img = cv2.imread(image_address, 0)
        img = cv2.resize(img, (64, 64))  # resize image
        img=np.reshape(img,(1,64,64,1))  # reshare image
        return self.model.predict(img)[0][0]
        
        
from sklearn.model_selection import KFold #sklearn en general => model de selection parmis eux on trouve Kfold pour selectionner les parties de data pour l'entrainement
import numpy as np #numpy vient pour manipuler les type structural => les matrices, les tensors 

class Data:
    def __init__(self):
        self.no_angles = 360 # 360°
        self.X = np.array([])
        self.Y = np.array([])
        self.splt = 0

    def SplitingData(self):
        from sklearn.model_selection import train_test_split
        import numpy as np
        lable_info=read_lable()
        image_info=read_image()
        #print(image_info[1][0])
        ids=lable_info.keys()   #ids = acceptable labeled ids
        #print(type(ids))
        del lable_info['Truth-Data:']       
        #print(lable_info)
        #print(ids)
        X=[]
        Y=[]
        for id in ids:
            for angle in range(self.no_angles):
                X.append(image_info[id][angle])
                Y.append(lable_info[id][angle])
        self.X=np.array(X)
        self.Y=np.array(Y)
        return [X,Y]
    
    def myprint(s):
        with open('modelsummary.txt','w+') as f:
            print(s, file=f)

    def kfold_Split(self, n_split, epoch):
        counter = 0
        self.splt = n_split
        for train_index,test_index in KFold(n_split).split(self.X):
            x_train,x_test = self.X[train_index],self.X[test_index]
            y_train,y_test = self.Y[train_index],self.Y[test_index]
            rows, cols = x_train[0].shape
            (a, b, c) = x_train.shape
            x_train = np.reshape(x_train, (a, b, c, 1))
            (a, b, c) = x_test.shape
            x_test = np.reshape(x_test, (a, b, c, 1))
            first = Model(rows, cols)
            model = first.createModel()
            #model = first.createModel1()
            print("Begin Training...")
            # Train Model using 
            counter += 1
            #first.TrainModel(model, x_train, y_train, epoch, counter) # n epoch
            first.TrainModel(x_train, y_train, x_test, y_test, epoch, counter) # n epoch
            model.save('/kaggle/working/Models/Model'+str(counter)+'.h5')
            # Confusion Matrix
            from sklearn.metrics import confusion_matrix
            #model.predict(y_test)
            print(np.size(y_test))
            y_pred=model.predict_classes(x_test)
            classes = ['Benign','Malignant']
            print(y_pred)
            con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
            con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
            con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)
            figure = plt.figure(figsize=(8, 8))
            sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

            #cm=confusion_matrix(y_test,y_pred_class)
            #print(cm)
            #print('Model evaluation ', model.evaluate(x_test,y_test))
            #print(model.summary()) #Summarise the whole model and params in one table
            #model.summary(print_fn=myprint)
    
            
    def Classify(self, image_address):
        from keras.models import load_model #charger les models
        import cv2
        import keras
        img = cv2.imread(image_address, 0)
        img = cv2.resize(img, (64, 64))  # resize image
        img=  np.reshape(img,(1,64,64,1))  # reshare image
        #return self.model.predict(img)[0][0]
        labels = ['B', 'M']
        l = list()
        predictedLabels = {}
        for i in range(1, self.splt+1):
            model_address = '/kaggle/working/Models/Model'+str(i)+'.h5'
            print(model_address)
            my_model=load_model(model_address)
            #print(my_model.Classify(image_address))
            xnew = my_model.predict(img)#[0][0]
            #y_proba = my_model.predict(img)
            #y_classes = keras.np_utils.probas_to_classes(y_proba)
            y_prob = my_model.predict(img) 
            y_classes = y_prob.argmax(axis=-1)
            y_label = my_model.predict_classes(img, verbose=1)[0][0]
            #l.append(y_prob)
            #l.append(labels[1 if y_prob > 0.5 else 0])
            #predictedLabels[y_prob[0][0]] = labels[1 if y_prob > 0.5 else 0]
            predictedLabels[y_prob[0][0]] = labels[y_label]
            #print('prediction : ',xnew)
            #print('class : ',my_model.predict_classes(img, verbose=1))
            #print('y_classesmal : ',y_classes)
        #print('Classifying..')
        return predictedLabels
        
def main():
    #while 1:
    #   choice = input('press 1 to train, 2 to classify and 3 to exit :')
       data = Data()
    #   if choice == '1':
       print('Spliting data...')
       X,Y = data.SplitingData()
       print('Applying KFold and Trainning...')
       data.kfold_Split(4,40) # first number is split number second one is the number of epochs => kfold_split(split, epoch)
    #  elif choice == '2':
       L = list()
       address  = '/kaggle/input/mias-mammography/all-mias/mdb167.pgm'
       address1 = '/kaggle/input/mias-mammography/all-mias/mdb059.pgm'
       address2 = '/kaggle/input/mias-mammography/all-mias/mdb063.pgm'
       address3 = '/kaggle/input/mias-mammography/all-mias/mdb080.pgm'
       print('finished')
       #L.append(data.Classify(address))
       #L.append(data.Classify(address1))
       #L.append(data.Classify(address2))
       #L.append(data.Classify(address3))
       #for i in L:
       #     print(i,end=' ')
    #   elif choice == '3':
    #        break
    #   else:
    #        print('please choose one of the given choices !')
main()
    
!ls logs/fit
!zip -r fitting.zip logs/fit 