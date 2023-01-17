import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.applications import imagenet_utils

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model, load_model, Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

import numpy as np

import matplotlib.pyplot as plt

import time

from PIL import Image

from sklearn.model_selection import train_test_split

import cv2

import os

from tqdm import tqdm

import random as rd
def process_data(source_dir, subject,image_size,t_split, v_split,rand_seed):

    # read in data from each class then split into train, test,valid data sets

    data=[]

    labels=[]

    file_names=[]

    class_list=os.listdir(source_dir) # list of flower types daisy etc

    class_list.remove('brain_tumor_dataset')

    print (class_list)

    leave=True    

    c_count=-1

    for c in class_list:

        c_count=c_count +1

        c_path=os.path.join(source_dir,c) #path to class dir - ie c:\temp\flowers\daisy

        partition=c_path.split(r'/')

        p_len=len(partition)

        desc=partition[p_len-2] + '-' + partition[p_len-1]

        # read in the data for each class 

        f_list=os.listdir(c_path) # path to images in class directory

        for f in tqdm(f_list, desc=desc, unit='files', leave=leave):

            index=f.rfind('.')

            ext=f[index+1:]

            if ext in ['jpg', 'jpe', 'png', 'jpeg', 'gif']:

                f_path=os.path.join(c_path, f)                        

                img=cv2.imread(f_path,1)

                image_array = Image.fromarray(img , 'RGB')

                resize_img = image_array.resize((image_size ,image_size))

                data.append(np.array(resize_img))   #data is a list of image arrays for the entire data set

                labels.append(c_count) # c_count is the label associated with the array image, 

                file_names.append(f)

    # at this point all image data is in a list of arrays, labels exist for each image

    net_split=(t_split + v_split)/100

    v_share=t_split/(t_split + v_split)

    train_data , x , train_labels , y = train_test_split(data,labels,test_size =net_split,random_state =rand_seed)

    train_files, a =train_test_split(file_names, test_size=net_split, random_state=rand_seed)

    val_data , test_data , val_labels , test_labels = train_test_split(x,y,test_size = v_share,random_state = rand_seed)

    val_files, test_files=train_test_split(a,test_size=v_share, random_state=rand_seed)

    print_data(train_labels, test_labels, val_labels, class_list)

    train_data=np.array(train_data)

    train_labels-np.array(train_labels)

    train_files=np.array(train_files)

    test_data=np.array(test_data)

    test_labels=np.array(test_labels)

    test_files=np.array(test_files)

    val_data=np.array(val_data)

    val_labels=np.array(val_labels)

    val_files=np.array(val_files)

    data_set=[train_data, train_labels, test_data, test_labels, val_data, val_labels,test_files, class_list]

    return data_set    
def print_data(train_labels, test_labels, val_labels, class_list):

    train_list=list(train_labels)

    test_list=list(test_labels)

    val_list=list(val_labels)

    print('{0:9s}Class Name{0:10s}Class No.{0:4s}Train Files{0:7s}Test Files{0:5s}Valid Files'.format(' '))

    for i in range(0, len(class_list)):

        c_name=class_list[i]

        tr_count=train_list.count(i)

        tf_count=test_list.count(i)

        v_count=val_list.count(i)

        print('{0}{1:^25s}{0:5s}{2:3.0f}{0:9s}{3:4.0f}{0:15s}{4:^4.0f}{0:12s}{5:^3.0f}'.format(' ',

                                                                                               c_name,i,tr_count,

                                                                                               tf_count,v_count))

    print('{0:30s} ______________________________________________________'.format(' '))

    msg='{0:10s}{1:6s}{0:16s}{2:^3.0f}{0:8s}{3:3.0f}{0:15s}{4:3.0f}{0:13s}{5}\n'

    print(msg.format(' ', 'Totals',len(class_list),len(train_labels),len(test_labels),len(val_labels)))
def get_steps(train_data, test_data,val_data,batch_size):

    length=train_data.shape[0]

    if length % batch_size==0:

        tr_steps=int(length/batch_size)

    else:

        tr_steps=int(length/batch_size) + 1

    length=val_data.shape[0]

    if length % batch_size==0:

        v_steps=int(length/batch_size)

    else:

        v_steps=int(length/batch_size) + 1

    length=test_data.shape[0]

    batches=[int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80]

    batches.sort(reverse=True)

    t_batch_size=batches[0]

    t_steps=length/t_batch_size        

    return [tr_steps,t_steps, v_steps, t_batch_size]
def make_model(source_dir,output_dit,class_list, image_size, subject,model_size, rand_seed,lr_factor):

    size=len(class_list)

    check_file = os.path.join(output_dir, 'tmp.h5')

        

    if model_size=='L':

        # mobile = keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape)

        mobile = tf.keras.applications.mobilenet.MobileNet()        

        #remove last 5 layers of model and add dense layer with 128 nodes and the prediction layer with size nodes

        # where size=number of classes

        x=mobile.layers[-6].output

        x=Dense(128, kernel_regularizer = regularizers.l2(l = 0.015), activation='relu')(x)

        x=Dropout(rate=.5, seed=rand_seed)(x)

        predictions=Dense (size, activation='softmax')(x)

        model = Model(inputs=mobile.input, outputs=predictions)

        for layer in model.layers:

            layer.trainable=True

        model.compile(Adam(lr=lr_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        

    else:

        if model_size=='M':

            fm=2

        else:

            fm=1

        model = Sequential()

        model.add(Conv2D(filters = 4*fm, kernel_size = (3, 3), activation ='relu', padding ='same', name = 'L11',

                         kernel_regularizer = regularizers.l2(l = 0.015),input_shape = (image_size, image_size, 3)))

        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name ='L12'))

        model.add(BatchNormalization(name = 'L13'))

        model.add(Conv2D(filters = 8*fm, kernel_size = (3, 3), activation ='relu',

                         kernel_regularizer = regularizers.l2(l = 0.015), padding ='same', name = 'L21')) 

        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name ='L22'))

        model.add(BatchNormalization(name = 'L23'))

        model.add(Conv2D(filters = 16*fm, kernel_size = (3, 3), activation ='relu',

                         kernel_regularizer = regularizers.l2(l = 0.015), padding ='same', name ='L31')) 

        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name ='L32'))

        model.add(BatchNormalization(name = 'L33'))

        if fm==2:

            model.add(Conv2D(filters = 32*fm, kernel_size = (3, 3), activation ='relu',

                             kernel_regularizer = regularizers.l2(l = 0.015),padding ='same', name ='L41')) 

            model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name ='L42'))

            model.add(BatchNormalization(name = 'L43'))

            model.add(Conv2D(filters = 64*fm, kernel_size = (3, 3), activation ='relu', 

                             kernel_regularizer = regularizers.l2(l = 0.015),padding ='same', name ='L51')) 

            model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name ='L52'))

            model.add(BatchNormalization(name = 'L53'))

            

        model.add(Flatten())

        model.add(Dense(256 *fm,kernel_regularizer = regularizers.l2(l = 0.015), activation='relu', name ='Dn1'))

        model.add(Dropout(rate=.5))

        model.add(Dense(size, activation = 'softmax', name ='predict'))

        model.compile(Adam(lr=lr_rate),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min', verbose=1)

    checkpoint = ModelCheckpoint(check_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

    lrck=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=1,

                                           verbose=1, mode='min', min_delta=0.000001, cooldown=1, min_lr=1.0e-08)

    callbacks=[checkpoint,lrck, early_stop]

    return [model, callbacks]
def make_generators(data_sets, batch_size,t_batch_size,rand_seed):

    #data_set[0]=train data,[1]train labels,[2]=test data,[3]=test labels,[4]=value data,[5]=val labels,[6]=test files

    train_datagen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                             horizontal_flip=True,

                             samplewise_center=True,

                             samplewise_std_normalization=True)

    train_gen=train_datagen.flow(data_sets[0],data_sets[1], batch_size=batch_size, seed=rand_seed)

    val_datagen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                             samplewise_center=True,

                             samplewise_std_normalization=True)

    val_gen=val_datagen.flow(data_sets[4], data_sets[5], batch_size=batch_size, seed=rand_seed)

    test_datagen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                             samplewise_center=True,

                             samplewise_std_normalization=True)

    test_gen=test_datagen.flow(data_sets[2], data_sets[3], batch_size=t_batch_size, shuffle=False)

    return [train_gen, test_gen, val_gen]
def train(model, train_gen, val_gen,tr_steps,v_steps, epochs,callbacks,sub):

    start=time.time()

    data = model.fit_generator(generator = train_gen, validation_data= val_gen,

                       steps_per_epoch=tr_steps, epochs=epochs, 

                       validation_steps=v_steps, callbacks = callbacks)

    stop=time.time()

    duration = stop-start

    hrs=int(duration/3600)

    mins=int((duration-hrs*3600)/60)

    secs= duration-hrs*3600-mins*60

    msg='The training cycle took  {0} hours {1} minutes and {2:6.2f} seconds'

    print(msg.format(hrs, mins,secs))

    return data

    
def tr_plot(tacc,vacc,tloss,vloss):

    #Plot the training and validation data

    Epoch_count=len(tloss)

    Epochs=[]

    for i in range (0,Epoch_count):

        Epochs.append(i+1)

    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    axes[0].plot(Epochs,tloss, 'r', label='Training loss')

    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )

    axes[0].set_title('Training and Validation Loss')

    axes[0].set_xlabel('Epochs')

    axes[0].set_ylabel('Loss')

    axes[0].legend()

    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')

    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')

    axes[1].set_title('Training and Validation Accuracy')

    axes[1].set_xlabel('Epochs')

    axes[1].set_ylabel('Accuracy')

    axes[1].legend()

    plt.tight_layout

    plt.style.use('fivethirtyeight')

    plt.show()

 
def display_pred(source_dir,output_dir,pred,t_files,t_labels,class_list,subject, model_size):

    # t_files are the test files, t_labels are the class label associated with the test file

    # class_list is a list of classes

    trials=len(t_files)   # number of predictions made should be same as len(t_files)

    errors=0

    prob_list=[]

    true_class=[]

    pred_class=[]

    file_list=[]

    x_list=[]

    index_list=[]

    pr_list=[]

    error_msg=''

    for i in range (0,trials):

        p_c_num=pred[i].argmax()  #the index with the highest prediction value

        if p_c_num !=t_labels[i]: #if the predicted class is not the same as the test label it is an error

            errors=errors + 1

            file_list.append(t_files[i])  # list of file names that are in error

            true_class.append(class_list[t_labels[i]]) # list classes that have an eror

            pred_class.append(class_list[p_c_num]) #class the prediction selected

            prob_list.append(100 *pred[i][p_c_num])# probability of the predicted class

            add_msg='{0:^24s}{1:5s}{2:^20s}\n'.format(class_list[t_labels[i]], ' ', t_files[i])

            error_msg=error_msg + add_msg

            

    accuracy=100*(trials-errors)/trials

    print('\n There were {0} errors in {1} trials for an accuracy of {2:7.3f}'.format(errors, trials,accuracy,),flush=True)

    if errors<=25:

        msg='{0}{1:^24s}{0:3s}{2:^20s}{0:3s}{3:20s}{0:3s}{4}'

        print(msg.format(' ', 'File Name', 'True Class', 'Predicted Class', 'Probability'))

        for i in range(0,errors):

            msg='{0}{1:^24s}{0:3s}{2:^20s}{0:3s}{3:20s}{0:5s}{4:^6.2f}'

            print (msg.format(' ',file_list[i], true_class[i], pred_class[i], prob_list[i]))

    else:

        print('with {0} errors the full error list will not be printed'.format(errors))    

    acc='{0:6.2f}'.format(accuracy)

    if model_size=='L':

        ms='Large'

    elif model_size=='M':

        ms= 'Medium'

    else:

        ms= 'Small'

    header='Classification subject: {0} There were {1} errors in {2} tests for an accuracy of {3} using a {4} model\n'.format(subject,errors,trials,acc,ms)

    header= header +'{0:^24s}{1:5s}{2:^20s}\n'.format('CLASS',' ', 'FILENAME') 

    error_msg=header + error_msg

    file_name='error list-' + model_size + acc +'.txt'

    print('\n file {0} containing the list of errors has been saved to {1}'.format(file_name, output_dir))

    file_path=os.path.join(output_dir,file_name)

    f=open(file_path, 'w')

    f.write(error_msg)

    f.close()

    for c in class_list:

        count=true_class.count(c)

        x_list.append(count)

        pr_list.append(c)

    for i in range(0, len(x_list)):  # only plot classes that have errors

        if x_list[i]==0:

            index_list.append(i)

    for i in sorted(index_list, reverse=True):  # delete classes with no errors

        del x_list[i]

        del pr_list[i]      # use pr_list - can't change class_list must keep it fixed

    fig=plt.figure()

    fig.set_figheight(len(pr_list)/4)

    fig.set_figwidth(6)

    plt.style.use('fivethirtyeight')

    for i in range(0, len(pr_list)):

        c=pr_list[i]

        x=x_list[i]

        plt.barh(c, x, )

        plt.title('Errors by class')

    plt.show()

    time.sleep(5.0)

    

    return accuracy        
def save_model(output_dir,subject, accuracy,r_model, image_size):

    # save the model with the  subect-accuracy.h5

    acc=str(accuracy)[0:5]

    tempstr=subject + '-' +str(image_size) + '-' + acc + '.h5'

    model_save_path=os.path.join(output_dir,tempstr)

    r_model.save(model_save_path)    
def make_predictions(output_dir, test_gen, t_steps):

    # the best model was saved as a file need to read it in and load it since it is not available otherwise

    test_gen.reset()

    msg='Training has completed, now loading saved best model and processing test set to see how accurate the model is'

    print (msg,flush=True)

    model_path=os.path.join(output_dir,'tmp.h5')

    model=load_model(model_path)                      # load the saved model with lowest validation loss

    pred=model.predict_generator(test_gen, steps=t_steps,verbose=1) # make predictions on the test set

    return [pred, model]
def wrapup (source_dir,output_dir,subject, accuracy, r_model,image_size,run_num, model_size):

    if accuracy >= 95:

        msg='With an accuracy of {0:5.2f} the results appear satisfactory and the program will terminate'

        print(msg.format(accuracy),flush=True)

        return [False, None]

    elif accuracy >=85 and accuracy < 95:

        if run_num<2:

            msg='With an accuracy of {0:5.2f} our results are mediocure, will run 10 more epochs to see if accuracy improves '

            print (msg.format(accuracy),flush=True)

            return [True, 10]

        else:

            print('Final accuracy of {0} is still mediocure- program terminating'.format(accuracy))

            if model_size !='L':

                print('try running again with model_size=L to get a more accurate result')

            return [False, None]

    else:

        if run_num<2:

            msg='With an accuracy  of {0:5.2f} the results are poor, will run for 15 more epochs to see if accuracy improves'

            print (msg.format( accuracy),flush=True)

            msg='if running more epochs does not improve test set accuracy you may want to get more training data '

            msg=msg + 'or perhaps crop your images so the desired subject takes up most of the image'

            print (msg, flush=True)

            return [True, 15]

        else:

            print('Final accuracy of {0} is still poor - program is terminating'.format(accuracy))

            if model_size !='L':

                print('try running again with model_size=L to get a more accurate result')

            return [False, None] 
def TF2_classify(source_dir,output_dir,mode,subject, t_split=10, v_split=5, epochs=20,batch_size=80,

                 lr_rate=.002,lr_factor=.8,image_size=224,rand_seed=128,model_size='L'):

    model_size=model_size.upper()

    mode=mode.lower()

    if model_size=='L':

        image_size=224              # for the large model image size must be 224

    data_sets=process_data(source_dir, subject,image_size,t_split, v_split,rand_seed)

    #data_set[0]=train data,[1]train labels,[2]=test data,[3]=test labels,[4]=value data,[5]=val labels,[6]=test files

    # data_sets[7]=class_list

    steps=get_steps(data_sets[0],data_sets[2],data_sets[4],batch_size)

    # tr_steps=steps[0]  t_steps=steps[1]   v_steps=steps[2] t_batch_size=steps[3]

    model_data=make_model(source_dir,output_dir,data_sets[7], image_size, subject, model_size,rand_seed,lr_factor)

    # model=model_data[0]  callbacks=model_data[1]

    gens=make_generators(data_sets,batch_size,steps[3], rand_seed)

    # train_gen = gens[0]  test_gen=gens[1]   val_gen=gens[2]

    run_num=0

    run=True

    tacc=[]

    tloss=[]

    vacc=[]

    vloss=[]

    while run:

        run_num=run_num +1

        results=train(model_data[0], gens[0], gens[2],steps[0],steps[2], epochs,model_data[1],subject)

        tacc_new=results.history['accuracy']

        tloss_new=results.history['loss']

        vacc_new =results.history['val_accuracy']

        vloss_new=results.history['val_loss']

        for d in tacc_new:  # need to append new data from training to plot all epochs

            tacc.append(d)

        for d in tloss_new:

            tloss.append(d)

        for d in vacc_new:

            vacc.append(d)

        for d in vloss_new:

            vloss.append(d)

        tr_plot(tacc,vacc,tloss,vloss)

        predict=make_predictions(output_dir, gens[1], steps[1],)

        # pred= predict[0]  r_model=predict[1]

        accuracy=display_pred(source_dir,output_dir,predict[0],data_sets[6],data_sets[3],data_sets[7], subject, model_size)

        decide=wrapup(source_dir,output_dir,subject, accuracy, predict[1], image_size,run_num, model_size)

        run=decide[0]

        epochs=decide[1] 
import os

d_list=os.listdir('/kaggle/input')

for d in d_list:

    d_path=os.path.join('/kaggle/input', d)

    class_list=os.listdir(d_path)

    print(class_list)

    for c in class_list:

        c_path=os.path.join(d_path, c)

        print(c,c_path)

source_dir='/kaggle/input/brain-mri-images-for-brain-tumor-detection'

output_dir='/kaggle/working'

subject='brain tumors'

t_split=8

v_split=10

epochs=30

batch_size=40

lr_rate=.0015

lr_factor=.8

image_size=224

rand_seed=64

model_size='L'

mode='all'



TF2_classify(source_dir,output_dir,mode,subject, t_split=t_split, v_split=v_split, epochs=epochs,batch_size=batch_size,

         lr_rate=lr_rate,lr_factor=lr_factor,image_size=image_size,rand_seed=rand_seed, model_size=model_size)