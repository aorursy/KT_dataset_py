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

import numpy as np

import matplotlib.pyplot as plt

import time

import os

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

# set ansi color values

Cblu ='\33[34m'

Cend='\33[0m'   # sets color back to default 

Cred='\033[91m'

Cblk='\33[39m'

Cgreen='\33[32m'

Cyellow='\33[33m'
def get_paths(source_dir,output_dir,mode,subject):

    # NOTE if running on kaggle these directories will need to be amended

    # if all files are in a single kaggle input directory change the string 'consolidated'

    # to match the directory name used in the database.

    #if files are seperated in training, test and validation directories change the strings

    # 'train', 'test' and 'valid' to match the directory names used in the database

    if mode =='ALL':

        # all data is in a single directory must be split into train, test, valid data sets

        train_path=os.path.join(source_dir)    

        classes=os.listdir(train_path) 

        class_num=len(classes)

        test_path=None        

        valid_path=None

       

    else:

        # data is seperated in 3 directories train, test, valid

        test_path=os.path.join(source_dir,'test')

        classes=os.listdir(test_path)

        class_num=len(classes)  #determine number of class directories in order to set leave value intqdm    

        train_path=os.path.join(source_dir, 'train')

        valid_path=os.path.join(source_dir,'valid')

                  

    # save the class dictionary as a text file so it can be used by predictor.py in the future

    #saves file as subject.txt  structure is similar to a python dictionary

    msg=''

    for i in range(0, class_num):

        msg=msg + str(i) + ':' + classes[i] +','

    id=subject  + '.txt'   

    dict_path=os.path.join (output_dir, id)

    f=open(dict_path, 'w')

    f.write(msg)

    f.close()    

    return [train_path, test_path, valid_path,classes]

      

    

   
def make_model(classes,lr_rate, height,width,model_type, rand_seed):

    size=len(classes)

    if model_type=='MOBILENET':

        weights='imagenet'

        if height==224:

            Top=True            

            cut=-2

        else:

            Top=False

            cut=-1            

        mobile = tf.keras.applications.mobilenet.MobileNet( include_top=Top,

                                                           input_shape=(height,width,3),

                                                           pooling='avg', weights=weights,

                                                           alpha=1, depth_multiplier=1)

        x=mobile.layers[cut].output

        x=Dense(256, kernel_regularizer = regularizers.l2(l = 0.015), activation='relu')(x)

        x=Dropout(rate=.5,name= 'drop2', seed=rand_seed)(x)

        predictions=Dense (size, activation='softmax')(x)

        model = Model(inputs=mobile.input, outputs=predictions)

        #model.summary()

        for layer in model.layers:

            layer.trainable=True

        model.compile(Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        

    elif model_type=='NASNET':    

        weights='imagenet'

        in_shape=(height, width,3)

        if height==224:

            include_top=True            

            cut=-2

            pool_type=None 

            size=1000

            tensor=None

        else:

            print('**************height= ', height)

            include_top=False

            cut=-1 

            pool_type='ave'

            tensor=input_shape

             

        mobile=keras.applications.nasnet.NASNetMobile(  )

        x=mobile.layers[cut].output       

        x=Dense(128, kernel_regularizer = regularizers.l2(l = 0.015), activation='relu')(x)

        x=Dropout(rate=.5,name= 'drop2', seed=rand_seed)(x)

        predictions=Dense (2, activation='softmax')(x)

        model = Model(inputs=mobile.input, outputs=predictions)  

        #model.summary()

        for layer in model.layers:

            layer.trainable=True

        model.compile(Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        #model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr_rate),loss='binary_crossentropy', metrics=['accuracy'])

        

    return model
def make_generators( paths, mode, batch_size, v_split, classes, height, width, rand_seed, model_size):

    #paths[0]=train path,paths[1]=test path paths[2]= valid path paths[3]=classes

    v_split=v_split/100.0

    file_names=[]

    labels=[] 

    # determine batch_size for test images determine numberof test files

    count=0

    train_batch_size = 80

    dir_list=os.listdir(paths[0]) # lists content of  directory

    for d in dir_list:# d is one of the class sub directories

        d_path=os.path.join(paths[0],d)

        if os.path.isdir(d_path):                        

            file_list=os.listdir(d_path)

            for f in file_list:      # f is a file in a class directory

                f_path=os.path.join(d_path, f)

                if os.path.isfile(f_path):

                    count=count + 1 

    train_steps=int(count/80)+1

    if mode == 'SEP':

        test_batch_size, test_steps=get_batch_size(paths[1], model_size, None)

        valid_batch_size, valid_steps=get_batch_size(paths[2], model_size, None)

        train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(paths[0],

                target_size=(height, width), batch_size=train_batch_size, seed=rand_seed, class_mode='categorical', color_mode='rgb')

        

        valid_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input) .flow_from_directory(paths[2], 

                target_size=(height, width), batch_size=valid_batch_size,

                seed=rand_seed, class_mode='categorical',color_mode='rgb', shuffle=False)

        

        test_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(paths[1],

                target_size=(height, width), batch_size=test_batch_size, class_mode='categorical',color_mode='rgb',

                seed=rand_seed, shuffle=False )

        for file in test_gen.filenames:

            file_names.append(file)            

        for label in test_gen.labels:

            labels.append(label)

        

        return [train_gen, test_gen, valid_gen, file_names, labels, [train_steps, test_steps, valid_steps]]

                  

    else:

        

        # all data is in a single directory there are no test images use validation images as test images

        train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                             validation_split=v_split).flow_from_directory(paths[0],

                                                                                    target_size=(height, width),

                                                                                    batch_size=train_batch_size,

                                                                                    subset='training',seed=rand_seed)

        valid_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                                    validation_split=v_split).flow_from_directory(paths[0],

                                                                                    target_size=(height, width),

                                                                                    batch_size=train_batch_size,

                                                                                    subset='validation',

                                                                                    seed=rand_seed, shuffle=False)

        

        

        for file in valid_gen.filenames:

            file_names.append(file)

        for label in valid_gen.labels:

            labels.append(label)

        length=len(labels)

        valid_batch_size, valid_steps=get_batch_size(None, model_size, length)

    return [train_gen, valid_gen, valid_gen, file_names, labels, [train_steps, valid_steps, valid_steps] ]
def get_batch_size(dir, model_size, mode): 

    max_batch_size = 80 if model_size=='MOBILENET' else 40

    count=0

    if mode!=None:

        count=mode

    else:

        dir_list=os.listdir(dir) # lists content of  directory

        for d in dir_list:       # d is one of the class sub directories

            d_path=os.path.join(dir,d)

            if os.path.isdir(d_path):                        

                file_list=os.listdir(d_path)

                for f in file_list:      # f is a file in a class directory

                    f_path=os.path.join(d_path, f)

                    if os.path.isfile(f_path):

                        count=count + 1

    factors=[]

    # find out if number of good files is divisable

    for i in range (1, int(count/2) +2):        

        if count % i ==0:

            factors.append(i) 

    # find the largest factor that is less than or equal to 80   

    end=len(factors)-1 

    for i in range(end, -1, -1):

        if factors[i]<=max_batch_size:

            batch_size=int(factors[i])

            steps=int(count/batch_size)

            break    

    return (batch_size, steps)
def train(model, callbacks, train_gen, val_gen,steps_list, epochs,start_epoch):

    # steps_list[0]=training steps, steps_list[2]=validations steps

    start=time.time()

    data = model.fit_generator(generator = train_gen,

                               validation_data= val_gen, epochs=epochs, initial_epoch=start_epoch,

                               validation_steps=steps_list[2],callbacks = callbacks, verbose=1)

    stop=time.time()

    duration = stop-start

    hrs=int(duration/3600)

    mins=int((duration-hrs*3600)/60)

    secs= duration-hrs*3600-mins*60

    msg='{0}Training took\n {1} hours {2} minutes and {3:6.2f} seconds {4}'

    print(msg.format(Cblu,hrs, mins,secs,Cend))

    return data

    
def tr_plot(tacc,vacc,tloss,vloss):

    #Plot the training and validation data

    Epoch_count=len(tloss)

    Epochs=[]

    for i in range (0,Epoch_count):

        Epochs.append(i+1)

    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss

    val_lowest=vloss[index_loss]

    index_acc=np.argmax(vacc)

    val_highest=vacc[index_acc]

    plt.style.use('fivethirtyeight')

    sc_label='best epoch= '+ str(index_loss+1)

    vc_label='best epoch= '+ str(index_acc + 1)

    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    axes[0].plot(Epochs,tloss, 'r', label='Training loss')

    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )

    axes[0].scatter(index_loss+1,val_lowest, s=150, c= 'blue', label=sc_label)

    axes[0].set_title('Training and Validation Loss')

    axes[0].set_xlabel('Epochs')

    axes[0].set_ylabel('Loss')

    axes[0].legend()

    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')

    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')

    axes[1].scatter(index_acc+1,val_highest, s=150, c= 'blue', label=vc_label)

    axes[1].set_title('Training and Validation Accuracy')

    axes[1].set_xlabel('Epochs')

    axes[1].set_ylabel('Accuracy')

    axes[1].legend()

    plt.tight_layout

    #plt.style.use('fivethirtyeight')

    plt.show()

    

 
def display_pred(output_dir, pred, file_names, labels, subject, model_size,classes, kaggle):    

    trials=len(labels)

    errors=0

    e_list=[]

    prob_list=[]

    true_class=[]

    pred_class=[]

    x_list=[]

    index_list=[]

    pr_list=[]

    error_msg=''    

    for i in range (0,trials):

        p_class=pred[i].argmax()

        if p_class !=labels[i]: #if the predicted class is not the same as the test label it is an error

            errors=errors + 1

            fname=os.path.basename(file_names[i])

            e_list.append(fname)  # list of file names that are in error

            true_class.append(classes[labels[i]]) # list classes that have an eror

            pred_class.append(classes[p_class]) #class the prediction selected

            prob_list.append(100 *pred[i][p_class])# probability of the predicted class

            add_msg='{0:^24s}{1:5s}{2:^20s}\n'.format(classes[labels[i]], ' ', file_names[i])

            error_msg=error_msg + add_msg

            

    accuracy=100*(trials-errors)/trials

    print('{0}\n There were {1} errors in {2} trials for an accuracy of {3:7.3f}{4}'.format(Cblu,errors, trials,accuracy,Cend))

    if kaggle==True and errors<26:

        ans='Y'

    else:

        ans='N'

    if kaggle==False:

        ans=input('To see a listing of prediction errors enter Y to skip press Enter\n ')

    if ans== 'Y' or ans  =='y':

        msg='{0}{1}{2:^20s}{1:3s}{3:^20s}{1:3s}{4:^20s}{1:5s}{5}{6}'

        print(msg.format(Cblu, ' ', 'File Name', 'True Class', 'Predicted Class', 'Probability', Cend))

        for i in range(0,errors):

            msg='{0}{1:^20s}{0:3s}{2:^20s}{0:3s}{3:^20s}{0:5s}{4:^6.2f}'

            print (msg.format(' ',e_list[i], true_class[i], pred_class[i], prob_list[i]))

    if kaggle==True:

        ans='Y'

    else:

        ans=input('\nDo you want to save the list of error files?. Enter Y to save or press Enter to not save  ')

    if ans=='Y' or ans=='y':

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

        file_id='error list-' + model_size + acc +'.txt'

        file_path=os.path.join(output_dir,file_id)

        f=open(file_path, 'w')

        f.write(error_msg)

        f.close()

    for c in classes:

        count=true_class.count(c)

        x_list.append(count)

        pr_list.append(c)

    for i in range(0, len(x_list)):  # only plot classes that have errors

        if x_list[i]==0:

            index_list.append(i)

    for i in sorted(index_list, reverse=True):  # delete classes with no errors

        del x_list[i]

        del pr_list[i]      # use pr_list - can't change class_list must keep it fixed

    if errors !=0:

        fig=plt.figure()

        fig.set_figheight(len(pr_list)/4)

        fig.set_figwidth(6)

        plt.style.use('fivethirtyeight')

        for i in range(0, len(pr_list)):

            c=pr_list[i]

            x=x_list[i]

            plt.barh(c, x, )

            plt.title( subject +' Classification Errors on Test Set')

        if errors>0:

            plt.show()

    if kaggle==False:

        ans=input('Press Enter to continue')

    return accuracy        
def save_model(output_dir,subject, accuracy, height, width, model, weights):

    # save the model with the  subect-accuracy.h5

    acc=str(accuracy)[0:5]

    tempstr=subject + '-' +str(height) + '-' + str(width) + '-' + acc + '.h5'

    model.set_weights(weights)

    model_save_path=os.path.join(output_dir,tempstr)

    model.save(model_save_path)    
def make_predictions( model, weights, test_gen, lr):

    config = model.get_config()

    pmodel = Model.from_config(config)  # copy of the model

    pmodel.set_weights(weights) #load saved weights with lowest validation loss

    pmodel.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])    

    print('Training has completed. Now loading test set to see how accurate the model is')

    results=pmodel.evaluate(test_gen, verbose=0)

    print('Model accuracy on Test Set is {0:7.2f} %'.format(results[1]* 100))

    predictions=pmodel.predict_generator(test_gen, verbose=0)     

    return predictions
def wrapup (output_dir,subject, accuracy, height,width, model, weights,run_num, kaggle):

    if accuracy >= 95:

        msg='{0} With an accuracy of {1:5.2f} % the results appear satisfactory{2}'

        print(msg.format(Cgreen, accuracy, Cend))

        if kaggle:

            save_model(output_dir, subject, accuracy, height, width , model, weights)

            print ('*********************Process is completed******************************')            

            return [False, None]        

    elif accuracy >=85 and accuracy < 95:

        if kaggle:

            if run_num==2:

                save_model(output_dir, subject, accuracy, height, width , model, weights)

                print ('*********************Process is completed******************************')

                return [False, None]

            else:

                print('running for 6 more epochs to see if accuracy improves')

                return[True,6] # run for 8 more epochs

        else:

            msg='{0}With an accuracy of {1:5.2f} % the results are mediocure. Try running more epochs{2}'

            print (msg.format(Cblu, accuracy,Cend))

    else:

        if kaggle:

            if run_num==2:

                save_model(output_dir, subject, accuracy, height,width , model, weights)

                print ('*********************Process is completed******************************')

                return [False, None]

            else:

                print('Running for 8 more epochs to see if accuracy improves')

                return[True,8] # run for 8 more epochs

        else:

            msg='{0} With an accuracy  of {1:5.2f} % the results would appear to be unsatisfactory{2}'

            print (msg.format(Cblu, accuracy, Cend))

            msg='{0}You might try to run for more epochs or get more training data '

            msg=msg + 'or perhaps crop your images so the desired subject takes up most of the image{1}'

            print (msg.format(Cblu, Cend))

    

    tryagain=True

    if kaggle==False:

        while tryagain==True:

            ans=input('To continue training from where it left off enter the number of additional epochs or enter H to halt  ')

            if ans =='H' or ans == 'h':

                run=False

                tryagain=False

                save_model(output_dir, subject, accuracy, height,width , model, weights)                                      

                print ('*********************Process is completed******************************')

                return [run,None]

            else:

                try:

                    epochs=int(ans)

                    run=True

                    tryagain=False

                    return [run,epochs]

                except ValueError:

                    print('{0}\nyour entry {1} was neither H nor an integer- re-enter your response{2}'.format(Cred,ans,Cend))

def set_dim(w,h):

    wh_list=[224,160, 128, 96 ]

    if w  in wh_list and h in wh_list and w==h:

        return(w,w)

    else:

        x=h if h>w else w

        # find closest value to what is in the list

        delta_min=np.inf

        for s in wh_list:

            delta =abs( x-s)

            if delta< delta_min:

                delta_min=delta

                h=s    

    return (h,h)
def TF2_classify(source_dir, output_dir, mode, subject, v_split=5, epochs=20, batch_size=80,

                 lr_rate=.002, height=224, width=224, rand_seed=128, model_size='V1', kaggle=False):

    model_size=model_size.upper()

    width, height =set_dim(width, height)    

    mode=mode.upper() 

    height, width = set_dim(height, width)

    paths=get_paths(source_dir,output_dir,mode,subject)

    #paths[0]=train path,paths[1]=test path paths[2]= valid path paths[3]=classes 

    gens=make_generators( paths, mode, batch_size, v_split, paths[3], height, width, rand_seed, model_size)

    #gens[0]=train generator gens[1]= test generator  gens[2]= validation generator

    #gens[3]=test file_names  gens[4]=test labels gens[5]=[train_steps, test_steps, valid_steps]

    model=make_model(paths[3],lr_rate, height, width, model_size, rand_seed) 

    class val(tf.keras.callbacks.Callback):

        # functions in this class adjust the learning rate 

        lowest_loss=np.inf

        best_weights=model.get_weights()

        lr=float(tf.keras.backend.get_value(model.optimizer.lr))

        epoch=0

        highest_acc=0

        

        def __init__(self):

            super(val, self).__init__()

            self.lowest_loss=np.inf

            self.best_weights=model.get_weights()

            self.lr=float(tf.keras.backend.get_value(model.optimizer.lr))

            self.epoch=0

            self.highest_acc=0

            

        def on_epoch_end(self, epoch, logs=None): 

            val.lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))

            val.epoch=val.epoch +1            

            v_loss=logs.get('val_loss')

            v_acc=logs.get('accuracy')

            if val.highest_acc<v_acc:

                val.highest_acc=v_acc

                val.best_weights=model.get_weights()

            if v_acc<=.95 and v_acc<val.highest_acc:

                lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))

                ratio=v_acc/val.highest_acc  # add a factor to lr reduction

                new_lr=lr * .5 * ratio

                tf.keras.backend.set_value(model.optimizer.lr, new_lr)

                msg='{0}\n current accuracy {1:7.4f} % is below 95 % and below the highest accuracy of {2:7.4f}, reducing lr to {3:11.9f}{4}'

                print(msg.format(Cyellow, v_acc* 100, val.highest_acc, new_lr,Cend))   

            if val.lowest_loss > v_loss:

                msg='{0}\n validation loss improved,saving weights with validation loss= {1:7.4f}\n{2}'

                print(msg.format(Cgreen, v_loss, Cend))

                val.lowest_loss=v_loss

                val.best_weights=model.get_weights()

                

            else:

                 if v_acc>.95 and val.lowest_loss<v_loss:

                        # reduce learning rate based on validation loss> val.best_loss

                        lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))

                        ratio=val.lowest_loss/v_loss  # add a factor to lr reduction

                        new_lr=lr * .7 * ratio

                        tf.keras.backend.set_value(model.optimizer.lr, new_lr)

                        msg='{0}\n current loss {1:7.4f} exceeds lowest loss of {2:7.4f}, reducing lr to {3:11.9f}{4}'

                        print(msg.format(Cyellow, v_loss, val.lowest_loss, new_lr,Cend))

           

    callbacks=[val()]

    run_num=0

    run=True

    tacc=[]

    tloss=[]

    vacc=[]

    vloss=[]

    start_epoch=0

    while run:

        run_num=run_num +1

        if run_num==1:

            print(' Starting Training Cycle')

        else:

            print('Resuming training from epoch {0}'.format(start_epoch))

        results=train(model,callbacks, gens[0], gens[2],gens[5], epochs,start_epoch) 

        # returns data from training the model - append the results for plotting

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

        last_epoch=results.epoch[len(results.epoch)-1] # this is the last epoch run

        tr_plot(tacc,vacc,tloss,vloss) # plot the data on loss and accuracy

        bestw=val.best_weights  # these are the saved weights with the lowest validation loss

        lr_rate=val.lr 

        predictions=make_predictions(model, bestw, gens[1], lr_rate)

        accuracy=display_pred(output_dir, predictions, gens[3], gens[4], subject, model_size, paths[3], kaggle)        

        model_path=os.path.join(source_dir, 'autism-224-224-95.71.h5')        

        decide=wrapup(output_dir,subject, accuracy, height, width, model, bestw,run_num, kaggle)

        run=decide[0]        

        if run==True:

            epochs=last_epoch + decide[1]+1

            start_epoch=last_epoch +1

        

           


source_dir=r'/kaggle/input/beauty-detection-data-set'

subject='beauty'

v_split=8

epochs=20

batch_size=80

lr_rate=.00005

height=224

width=224

rand_seed=100

model_size='MOBILENET'

mode='sep'

kaggle=True  # added to deal with fact that kaggle 'commit' does not allow user entry

              # set to True if you are doing a kaggle commit

if kaggle:

    output_dir=r'/kaggle/working'

    



TF2_classify(source_dir, output_dir, mode,subject, v_split= v_split, epochs=epochs,batch_size= batch_size,

         lr_rate= lr_rate,height=height, width=width,rand_seed=rand_seed, model_size=model_size, kaggle=kaggle)
