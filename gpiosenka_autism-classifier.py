import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model, load_model, Sequential

import numpy as np

import matplotlib.pyplot as plt

import time

import os

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
def save_model(output_dir,subject, accuracy, image_size,model_size, model, weights):

    # save the model with the  subect-accuracy.h5

    acc=str(accuracy)[0:5]

    id=subject + '-' + model_size + '-' +str(image_size) + '-' + acc + '.h5'    

    model.set_weights(weights)

    model_save_path=os.path.join(output_dir,id)    

    model.save(model_save_path)  
def make_predictions( model, weights, test_gen, lr):

    config = model.get_config()

    pmodel = Model.from_config(config)  # copy of the model

    pmodel.set_weights(weights) #load saved weights with lowest validation loss

    pmodel.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])    

    print('Training has completed. Now loading test set to see how accurate the model is')

    results=pmodel.evaluate(test_gen, verbose=0)

    print('Model accuracy on Test Set is {0:7.2f} %'.format(results[1]* 100))

    #predictions=pmodel.predict_generator(test_gen, verbose=0)     

    return results[1] * 100
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
def train(model, callbacks, generators, epochs,start_epoch):

    # steps_list[0]=training steps, steps_list[2]=validations steps

    start=time.time()

    data = model.fit_generator(generator = generators[0], validation_data= generators[2], 

                               epochs=epochs, initial_epoch=start_epoch,

                               callbacks = callbacks, verbose=1)

    #data=model.fit(x=generators[0],  epochs=epochs, verbose=1, 

    #               callbacks=callbacks,  validation_data=generators[2], shuffle=True,  initial_epoch=start_epoch) 

       

    stop=time.time()

    duration = stop-start

    hrs=int(duration/3600)

    mins=int((duration-hrs*3600)/60)

    secs= duration-hrs*3600-mins*60

    msg=f'Training took\n {hrs} hours {mins} minutes and {secs:6.2f} seconds'

    print_in_color(msg, (0,0,255),(0,0,0))

    return data
def make_model(classes,lr_rate, image_size,model_size,dropout, rand_seed):

    size=len(classes)

    mobile = tf.keras.applications.mobilenet.MobileNet( include_top=False,

                                                           input_shape=(image_size,image_size,3),

                                                           pooling='avg', weights='imagenet',

                                                           alpha=1, depth_multiplier=1)

    x=mobile.layers[-1].output

    x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

    if model_size=='S':

        x=Dense(128, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

                bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)

        x=Dropout(rate=dropout, seed=rand_seed)(x) 

    elif model_size=='M':

        x=Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

                bias_regularizer=regularizers.l1(0.006),activation='relu')(x)

        x=Dropout(rate=dropout, seed=rand_seed)(x) 

        x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

        x=Dense(16, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

                bias_regularizer=regularizers.l1(0.006),activation='relu')(x)

        x=Dropout(rate=dropout, seed=rand_seed)(x)

    else:

        x=Dense(1024, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

                bias_regularizer=regularizers.l1(0.006),activation='relu')(x)

        x=Dropout(rate=dropout, seed=rand_seed)(x) 

        x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

        x=Dense(128, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

                bias_regularizer=regularizers.l1(0.006),activation='relu')(x)

        x=Dropout(rate=dropout, seed=rand_seed)(x)

        x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

        x=Dense(16, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

                bias_regularizer=regularizers.l1(0.006),activation='relu')(x)

        x=Dropout(rate=dropout, seed=rand_seed)(x)        

    x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

    predictions=Dense (len(classes), activation='softmax')(x)

    model = Model(inputs=mobile.input, outputs=predictions)    

    for layer in model.layers:

        layer.trainable=True

    model.compile(Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    
def make_generators( paths, mode, split, classes, image_size, rand_seed):

    #paths[0]=train path,paths[1]=test path paths[2]= valid path paths[3]=classes

    split=split/100.0    

    batch_size=80    

    if mode == 'SEP': 

        train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(paths[0],

                target_size=(image_size, image_size), batch_size=batch_size, seed=rand_seed, class_mode='categorical', color_mode='rgb')

        

        valid_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input) .flow_from_directory(paths[2], 

                target_size=(image_size, image_size), batch_size=batch_size,

                seed=rand_seed, class_mode='categorical',color_mode='rgb', shuffle=False)

        

        test_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(paths[1],

                target_size=(image_size, image_size), batch_size=batch_size, class_mode='categorical',color_mode='rgb',

                seed=rand_seed, shuffle=False )

        file_names=test_gen.filenames          

        labels=test_gen.labels               

        return [[train_gen, test_gen, valid_gen], file_names, labels]                  

    else:        

        # all data is in a single directory there are no test images use validation images as test images

        train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                             validation_split=split).flow_from_directory(paths[0],

                                                                                    target_size=(image_size, image_size),

                                                                                    batch_size=batch_size,

                                                                                    subset='training',seed=rand_seed)

        valid_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                                    validation_split=split).flow_from_directory(paths[0],

                                                                                    target_size=(image_size, image_size),

                                                                                    batch_size=batch_size,

                                                                                    subset='validation',

                                                                                    seed=rand_seed, shuffle=False)

        file_names= valid_gen.filenames

        labels= valid_gen.labels

    return [[train_gen, valid_gen, valid_gen], file_names, labels]
def get_paths(source_dir,output_dir,mode,subject,classes): 

    class_count=len(classes)

    if mode =='ALL':

        # all data is in a single directory must be split into train, test, valid data sets 

        train_path=source_dir

        test_path=None        

        valid_path=None       

    else:

        # data is seperated in 3 directories train, test, valid

        test_path=os.path.join(source_dir,'test')

        train_path=os.path.join(source_dir, 'train')

        valid_path=os.path.join(source_dir,'valid')                  

    # save the class dictionary as a text file so it can be used by predictor.py in the future

    #saves file as subject.txt  structure is similar to a python dictionary

    msg=''

    for i in range(0, class_count):

        msg=msg + str(i) + ':' + classes[i] +','

    id=subject +'-' + str(class_count)  + '.txt'   

    dict_path=os.path.join (output_dir, id)

    f=open(dict_path, 'w')

    f.write(msg)

    f.close()    

    return [train_path, test_path, valid_path]

      
def check_inputs(source_dir, out_dir, split, image_size,model_size,dropout):

    status=True

    if os.path.isdir(source_dir)==False:

        msg=f'The source directory you specified {source_dir} does not exist - program terminating'

        status=False

    elif len(os.listdir(source_dir))<2:

        msg=f'directory {source_dir} must have at least 2 sub directories'

        status=False

    elif image_size not in [224,160,128,96]:

        msg=f'the images size you specified {image_size} is not one of 224,160,128 or 96 - program terminating' 

        status=False

    elif dropout <0.0 or dropout>1.0:

        msg=f'The drop out value you specified {dropout} must be a value between 0.0 and 1.0 - program terminating'

        status=False

    elif model_size not in ['L', 'M', 'S']:

        msg=f'The model size you specified {model_size} must be L, M, or S'

        status=False

    elif os.path.isdir(out_dir)==False:

        msg=f'The output directory you specified {out_dir} does not exist - program terminating'

        status=False

    else:        

        status=True

        msg=f'ERROR you must have a test, train and a valid subdirectory in {source_dir}'

        source_list=os.listdir(source_dir)       

        if ('test' in source_list and 'train' not in source_list) or ('test' in source_list and 'valid' not in source_list):            

            status=False                

        elif ('train' in source_list and 'test' not in source_list) or ('train' in source_list and 'valid' not in source_list):                      

            status=False                

        elif ('valid' in source_list and 'train' not in source_list) or ('valid' in source_list and 'test' not in source_list):                      

            status=False                 

        else: 

            if 'test' in source_list:

                test_path=os.path.join(source_dir,'test')

                train_path=os.path.join(source_dir,'train')

                valid_path=os.path.join(source_dir,'valid')

                test_list=sorted(os.listdir(test_path))

                train_list=sorted(os.listdir(train_path))

                valid_list=sorted(os.listdir(valid_path))            

                if train_list != test_list or train_list != valid_list or test_list !=valid_list:

                    status=False

                    msg='class directories must have identical names in the train, test and valid directories- program terminating'

                elif len(test_list) <2 or len(valid_list)<2 or len(train_list)<2:

                    status=False

                    msg=' the train, test and valid directories must have at least 2 class sub directories - program terminating'

            else:

                if len(os.listdir(source_dir))<2:

                    msg=f'The must be at least 2 subdirectories in {source_dir}'

                    status=False

                elif split==None or split<1 or split>100:

                    msg=f'the split parameter you specied {split} must be between 0 and 100 when class directories are in the source directory- program terminating'

                    status=False

                else:

                    mode='ALL'

                

    if status==False:

        print_in_color(msg, (255,0,0), (0,0,0))

        return (False, None,None)

    else:

        source_list=os.listdir(source_dir)

        if 'test' in source_list:

            mode='SEP'

            class_path=os.path.join(source_dir, 'test')

            classes=os.listdir(class_path)

        else:

            mode='ALL'

            classes=source_list

        return(True, mode, classes)
def TF2_classify(source_dir,out_dir,subject, split, epochs,lr_rate,image_size, model_size, dropout, rand_seed,dwell, kaggle):

    model_size=model_size.upper()

    if kaggle:

        out_dir=r'/kaggle/working'

    status, mode, classes=check_inputs(source_dir, out_dir,split, image_size,model_size, dropout) 

    if status==False:

        return False

    paths=get_paths(source_dir,output_dir,mode,subject,classes)

    generators, file_names,labels= make_generators( paths, mode, split, classes, image_size, rand_seed)

    model=make_model(classes,lr_rate, image_size,model_size,dropout, rand_seed)

    

                           

    class lradjust(tf.keras.callbacks.Callback):

        # functions in this class adjust the learning rate 

        lowest_loss=np.inf

        best_weights=model.get_weights()

        lr=float(tf.keras.backend.get_value(model.optimizer.lr))

        epoch=0

        tr_highest_acc=0        

        def __init__(self):

            super(lradjust, self).__init__()

            self.lowest_loss=np.inf

            self.best_weights=model.get_weights()

            self.lr=float(tf.keras.backend.get_value(model.optimizer.lr))

            self.epoch=0

            self.tr_highest_acc=0            

        def on_epoch_end(self, epoch, logs=None):                     

            lradjust.lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))

            lradjust.epoch=lradjust.epoch +1            

            v_loss=logs.get('val_loss')  # get the validation loss for this epoch

            tr_acc=logs.get('accuracy')  # get the training accuracy for this epoch

            if lradjust.tr_highest_acc<tr_acc:  # check if the accuracy for this epoch is the highest accuracy thus far

                lradjust.tr_highest_acc=tr_acc  # replace the highest accuracy with the accuracy for this epoch

                if lradjust.tr_highest_acc<.95:  # check if accuracy exceecs .95 if it is then save the weights

                    lradjust.best_weights=model.get_weights()

                    msg=f' \n saving weights with new highest accuracy of  {lradjust.tr_highest_acc:7.4f} '

                    print_in_color(msg, (255, 255,0), (0,0,0))

            if tr_acc<=.95 and tr_acc<lradjust.tr_highest_acc:

                # reduce lr because training accuracy went below highest accuracy

                lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))

                ratio=tr_acc/lradjust.tr_highest_acc  # add a factor to lr reduction

                new_lr=lr * .5 * ratio

                tf.keras.backend.set_value(model.optimizer.lr, new_lr)

                msg=f'\n current accuracy {tr_acc:7.4f} is below the highest accuracy of {lradjust.tr_highest_acc:7.4f},reducing learning rate to {new_lr:11.9f}'

                print_in_color(msg, (255,0,0),(0,0,0)) 

            if lradjust.lowest_loss > v_loss and lradjust.tr_highest_acc>.95:

                #accuracy is above 95% and the new validation loss is the lowest loss thus far so save the weights

                msg=f'\n validation loss improved to {v_loss:7.4f} from {lradjust.lowest_loss:7.4f} saving weights'

                print_in_color(msg, (0,255,0), (0,0,0))

                lradjust.lowest_loss=v_loss

                lradjust.best_weights=model.get_weights()

                

            else:

                 if tr_acc>.95 and lradjust.lowest_loss<v_loss:

                        # reduce learning rate based on validation loss> val.best_loss

                        lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))

                        ratio=lradjust.lowest_loss/v_loss  # add a factor to lr reduction

                        new_lr=lr * .7 * ratio

                        tf.keras.backend.set_value(model.optimizer.lr, new_lr)

                        msg=f'\n current validation loss {v_loss:7.4f} exceeds lowest loss of {lradjust.lowest_loss:7.4f}, reducing lr to {new_lr:11.9f}'

                        print_in_color(msg, (255,0,0), (0,0,0))

                        if dwell:

                            model.set_weights(lradjust.best_weights)  # ignore the new weights and load the best weights

                            msg='\nsetting weights back to best weights'

                            print_in_color(msg,( 0,0,255),(0,0,0))

                

    callbacks=[lradjust()] 

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

        results=train(model,callbacks, generators, epochs,start_epoch) 

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

        bestw=lradjust.best_weights  # these are the saved weights with the lowest validation loss

        lr_rate=lradjust.lr 

        accuracy=make_predictions( model, bestw, generators[1], lr_rate)

        save_model(output_dir,subject, accuracy, image_size,model_size, model, bestw)        

        return True

        

            
def print_in_color(txt_msg,fore_tupple,back_tupple,):

    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple 

    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)

    rf,gf,bf=fore_tupple

    rb,gb,bb=back_tupple

    msg='{0}' + txt_msg

    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 

    print(msg .format(mat))

    print('\33[0m') # returns default print color to back to black

    return
source_dir=r'/kaggle/input/autistic-children-data-set-traintestvalidate'

output_dir=r'/kaggle/working'

subject='autism'

split=8

epochs=30

lr_rate=.002

image_size=224

model_size='s'

dropout=.5

rand_seed=225

dwell=True

kaggle=True

status=TF2_classify(source_dir,output_dir,subject, split, epochs,lr_rate,image_size, model_size, dropout, rand_seed,dwell, kaggle)

print (f'status is {status}')