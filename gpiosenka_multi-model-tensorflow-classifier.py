import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization

from tensorflow.keras.optimizers import Adam, Adamax

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model, load_model, Sequential

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import os

from PIL import Image

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

logging.getLogger('tensorflow').setLevel(logging.FATAL)
dir=r'../input/autistic-children-data-set-traintestvalidate'

train_dir=os.path.join(dir,'train')

test_dir=os.path.join(dir,'test')

val_dir=os.path.join(dir,'valid')
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
classes=os.listdir(train_dir) # class names are the names of the sub directories

class_count=len(classes) # determine number of classes

batch_size=56 # set training batch size

rand_seed=123

start_epoch=0 # specify starting epoch

epochs=15 # specify the number of epochs to run

fine_tune_epochs=10

img_width=224 # use 224 X 224 images compatible with mobilenet model

img_height=224

bands=3

lr=.04 # specify initial learning rate

dropout=.3

metrics=['accuracy']

model_type='DenseNet201'

freeze=True # if false full model will be trained no fine tuning, if True basemodel weights are frozen model will be fine tuned

patience=2 # how many epoch to wait before reducing learning rate when there is no improvement in performance

threshold=.95 # accuracy level that must be achieve to start adjusting learning rate based on validation loss

factor=.5 # value that the current learning rate  will be multiplied by when there is nopreformance improvement

print_all=False # if True all predictions are printed out, if False only predictions misclassified are printed out
def get_bs(dir,b_max):

    # dir is the directory containing the samples, b_max is maximum batch size to allow based on your memory capacity

    # you only want to go through test and validation set once per epoch this function determines needed batch size ans steps per epoch

    length=0

    dir_list=os.listdir(dir)

    for d in dir_list:

        d_path=os.path.join (dir,d)

        length=length + len(os.listdir(d_path))

    batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=b_max],reverse=True)[0]  

    return batch_size,int(length/batch_size)
valid_batch_size, valid_steps=get_bs(val_dir, 100)

test_batch_size, test_steps=get_bs(test_dir,100)
train_gen=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input, horizontal_flip=True).flow_from_directory(

        train_dir,  target_size=(img_width, img_height), batch_size=batch_size, seed=rand_seed, class_mode='categorical', color_mode='rgb',)





valid_gen=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) .flow_from_directory(val_dir, 

                    target_size=(img_width, img_height), batch_size=valid_batch_size,

                    class_mode='categorical',color_mode='rgb', shuffle=False)

test_gen=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(test_dir,

                    target_size=(img_width, img_height), batch_size=test_batch_size,

                    class_mode='categorical',color_mode='rgb', shuffle=False )

file_names=test_gen.filenames  # save list of test files names to be used later

tlabels=test_gen.labels # save test labels to be used later

class_dict=test_gen.class_indices # dictionary of {key=label: value=label index }
images,labels=next(train_gen)

plt.figure(figsize=(20, 20))

length=len(labels)

if length<25:

    r=length

else:

    r=25

for i in range(r):

    plt.subplot(5, 5, i + 1)

    image=(images[i]+1 )/2

    plt.imshow(image)

    index=int(labels[i][1])

    plt.title(classes[index], color='white')

    plt.axis('off')

plt.show()
class Models:         

    def make_model(self, model_type, classes, width, height, bands, lr, freeze, dropout, metrics ):

        self.model_type=model_type

        self.classes=classes

        self.width=width

        self.height=height

        self.bands=bands

        self.lr=lr   

        self.freeze=freeze

        self.dropout=dropout

        self.metrics=metrics

        img_shape=(self.width, self.height,self.bands)

        model_list=['Mobilenet','MobilenetV2', 'VGG19','InceptionV3', 'ResNet50V2', 'NASNetMobile', 'DenseNet201']

        if self.model_type not in model_list:

            msg=f'ERROR the model name you specified {self.model_type} is not an allowed model name'

            print_in_color(msg, (255,0,0),(55,65,80))

            return None       

        if self.model_type=='Mobilenet':

            base_model=tf.keras.applications.mobilenet.MobileNet( include_top=False, input_shape=img_shape, pooling='max', weights='imagenet',dropout=self.dropout) 

        elif self.model_type=='MobilenetV2':

            base_model=tf.keras.applications.MobileNetV2( include_top=False, input_shape=img_shape, pooling='max', weights='imagenet')        

        elif self.model_type=='VGG19':

            base_model=tf.keras.applications.VGG19( include_top=False, input_shape=img_shape, pooling='max', weights='imagenet' )

        elif self.model_type=='InceptionV3':

            base_model=tf.keras.applications.InceptionV3( include_top=False, input_shape=img_shape, pooling='max', weights='imagenet' )

        elif self.model_type=='NASNetMobile':

            base_model=tf.keras.applications.NASNetMobile( include_top=False, input_shape=img_shape, pooling='max', weights='imagenet' )

        elif self.model_type=='DenseNet201':

            base_model=tf.keras.applications.densenet.DenseNet201( include_top=False, input_shape=img_shape, pooling='max', weights='imagenet' )

        else:

            base_model=tf.keras.applications.ResNet50V2( include_top=False, input_shape=img_shape, pooling='max', weights='imagenet')

        

        if self.freeze:

            for layer in base_model.layers[:-20]:#train top 20 layers of base model

                layer.trainable=False            

        msg=f'The model selected is {model_type}'

        print_in_color(msg, (0,255,0), (55,65,80))

        x=base_model.output

        x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

        x = Dense(128, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

                bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)

        x=Dropout(rate=dropout, seed=rand_seed)(x) 

        #x=Dense(self.classes * 400, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

       #         bias_regularizer=regularizers.l1(0.006),activation='relu')(x)

       # x=Dropout(self.dropout)(x)

       # x=Dense(self.classes * 200,kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),

       #         bias_regularizer=regularizers.l1(0.006), activation='relu')(x)

       # x=Dropout(self.dropout)(x)

        output=Dense(self.classes, activation='softmax')(x)

        model=Model(inputs=base_model.input, outputs=output)

        model.compile(Adamax(lr=self.lr), loss='categorical_crossentropy', metrics=self.metrics) 

        return model
model=Models().make_model(model_type,len(classes), img_width, img_height , bands, lr, freeze, dropout, metrics)
class LRA(keras.callbacks.Callback):

    best_weights=model.get_weights() # set a class vaiable so weights can be loaded after training is completed

    def __init__(self, patience, threshold, factor, model_name, freeze):

        super(LRA, self).__init__()

        self.patience=patience # specifies how many epochs without improvement before learning rate is adjusted

        self.threshold=threshold # specifies training accuracy threshold when lr will be adjusted based on validation loss

        self.factor=factor # factor by which to reduce the learning rate

        self.lr=float(tf.keras.backend.get_value(model.optimizer.lr)) # get the initiallearning rate and save it in self.lr

        self.highest_tracc=0.0 # set highest training accuracy to 0

        self.lowest_vloss=np.inf # set lowest validation loss to infinity

        self.count=0

        if freeze==True:

            msg=f' Starting training using model  { model_name} with weights frozen to imagenet weights initializing LRA callback'

        else:

             msg=f' Starting training using model  { model_name} training all layers initializing LRA callback'            

        print_in_color (msg, (244, 252, 3), (55,65,80))        

    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch

        lr=float(tf.keras.backend.get_value(self.model.optimizer.lr)) # get the current learning rate

        v_loss=logs.get('val_loss')  # get the validation loss for this epoch

        acc=logs.get('accuracy')  # get training accuracy        

        if acc < self.threshold: # if training accuracy is below threshold adjust lr based on training accuracy

            if acc>self.highest_tracc: # training accuracy improved in the epoch

                msg= f'\n training accuracy improved from  {self.highest_tracc:7.4f} to {acc:7.4f} learning rate held at {lr:9.6f}'

                print_in_color(msg, (0,255,0), (55,65,80))

                self.highest_tracc=acc # set new highest training accuracy

                LRA.best_weights=model.get_weights() # traing accuracy improved so save the weights

                count=0 # set count to 0 since training accuracy improved

                if v_loss<self.lowest_vloss:

                    self.lowest_vloss=v_loss                    

            else:  # training accuracy did not improve check if this has happened for patience number of epochs if so adjust learning rate

                if self.count>=self.patience -1:

                    self.lr= lr* self.factor # adjust the learning by factor

                    tf.keras.backend.set_value(model.optimizer.lr, self.lr) # set the learning rate in the optimizer

                    self.count=0 # reset the count to 0

                    if v_loss<self.lowest_vloss:

                        self.lowest_vloss=v_loss

                    msg=f'\nfor epoch {epoch +1} training accuracy did not improve for {self.patience } consecutive epochs, learning rate adjusted to {self.lr:10.8f}'

                    print_in_color(msg, (255,0,0), (55,65,80))

                else:

                    self.count=self.count +1 # increment patience counter

                    msg=f'\nfor  epoch {epoch +1} training accuracy did not improve, patience count incremented to {self.count}'

                    print_in_color(msg, (255,255,0), (55,65,80))

        else: # training accuracy is above threshold so adjust learning rate based on validation loss

            if v_loss< self.lowest_vloss: # check if the validation loss improved

                msg=f'\n for epoch {epoch+1} validation loss improved from  {self.lowest_vloss:7.4f} to {v_loss:7.4}, saving best weights'

                print_in_color(msg, (0,255,0), (55,65,80))

                self.lowest_vloss=v_loss # replace lowest validation loss with new validation loss                

                LRA.best_weights=model.get_weights() # validation loss improved so save the weights

                self.count=0 # reset count since validation loss improved               

            else: # validation loss did not improve

                if self.count>=self.patience-1:

                    self.lr=self.lr * self.factor # adjust the learning rate

                    msg=f' \nfor epoch {epoch+1} validation loss failed to improve for {self.patience} consecutive epochs, learning rate adjusted to {self.lr:10.8f}'

                    self.count=0 # reset counter

                    print_in_color(msg, (255,0,0), (55,65,80))

                    tf.keras.backend.set_value(model.optimizer.lr, self.lr) # set the learning rate in the optimizer

                else: 

                    self.count =self.count +1 # increment the patience counter

                    msg=f' \nfor epoch {epoch+1} validation loss did not improve patience count incremented to {self.count}'

                    print_in_color(msg, (255,255,0), (55,65,80))
def tr_plot(tr_data, start_epoch):

    #Plot the training and validation data

    tacc=tr_data.history['accuracy']

    tloss=tr_data.history['loss']

    vacc=tr_data.history['val_accuracy']

    vloss=tr_data.history['val_loss']

    Epoch_count=len(tacc)+ start_epoch

    Epochs=[]

    for i in range (start_epoch ,Epoch_count):

        Epochs.append(i+1)   

    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss

    val_lowest=vloss[index_loss]

    index_acc=np.argmax(vacc)

    acc_highest=vacc[index_acc]

    plt.style.use('fivethirtyeight')

    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)

    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)

    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))

    axes[0].plot(Epochs,tloss, 'r', label='Training loss')

    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )

    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)

    axes[0].set_title('Training and Validation Loss')

    axes[0].set_xlabel('Epochs')

    axes[0].set_ylabel('Loss')

    axes[0].legend()

    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')

    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')

    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)

    axes[1].set_title('Training and Validation Accuracy')

    axes[1].set_xlabel('Epochs')

    axes[1].set_ylabel('Accuracy')

    axes[1].legend()

    plt.tight_layout

    #plt.style.use('fivethirtyeight')

    plt.show()

callbacks=[LRA(patience=patience, threshold=threshold, factor=factor, model_name=model_type, freeze=freeze)]

results=model.fit(x=train_gen,  epochs=epochs, verbose=1, callbacks=callbacks,  validation_data=valid_gen,

               validation_steps=valid_steps,  shuffle=True,  initial_epoch=start_epoch)
tr_plot(results, start_epoch)
model.set_weights(LRA.best_weights)

acc=model.evaluate( test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps)[1]* 100

msg=f'accuracy on the test set is {acc:5.2f} %'

print_in_color(msg, (0,255,0),(55,65,80))

preds=model.predict(test_gen, batch_size=test_batch_size, verbose=0, steps=test_steps)
save_dir=r'../'

save_loc=os.path.join(save_dir, str(acc)[:str(acc).rfind('.')+3] + '.h5')

model.save(save_loc)
def show_images( test_dir, preds, acc, labels, file_names, max_images ):

    errors=np.ceil(len(file_names)*(1-acc/100))

    error_list=[]

    pred_class_list=[]

    true_class_list=[]

    if errors>0:

        if errors<=max_images:

            msg=f'The {int(errors)} images that were misclassified are shown below'

        else: msg=f'Only {max_images} of the {int(errors)} images that were misclassified are shown below'

        print_in_color(msg,(0,255,0),(55,65,80))

        rows=int(np.ceil(errors/5))

        height=rows * 5        

        plt.figure(figsize=(20,height ))

        j=1

        for i,p in enumerate(preds):

            if j <=max_images:

                if np.argmax(p) != labels[i]:

                    img_path=os.path.join(test_dir,file_names[i] )

                    plt.subplot(rows, 5, j)                

                    j=j + 1

                    img = Image.open(img_path)

                    img=img.resize((224,224))     

                    plt.axis('off')

                    title=file_names[i][:18]

                    plt.title(title)

                    plt.imshow(np.asarray(img)) 

            else:

                break

        plt.show()

    else:

        msg='With 100% accuracy there are no misclassified images to show'

        print_in_color(msg,(0,255,0), (55,65,80))

    return error_list
error_list=show_images( test_dir, preds, acc, tlabels, file_names,50 )
if freeze:

    for layer in model.layers: # make top 30 layers trainable

        layer.trainable==True

    start_epoch=epochs

    total_epochs=start_epoch + fine_tune_epochs

    data=model.fit(x=train_gen,  epochs=total_epochs, verbose=1, callbacks=callbacks,  validation_data=valid_gen,  initial_epoch=start_epoch)

    tr_plot(data, start_epoch)

    save_loc=os.path.join(save_dir, str(acc)[:str(acc).rfind('.')+3] + '.h5')

    model.save(save_loc)

    model.set_weights(LRA.best_weights)

    acc=model.evaluate( test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps)[1]* 100

    msg=f'accuracy of fine tuned model on the test set is {acc:5.2f} %'

    print_in_color(msg, (0,255,0),(55,65,80))

    preds=model.predict(test_gen, batch_size=test_batch_size, verbose=0, steps=test_steps)

    error_list=show_images( test_dir, preds, acc, tlabels, file_names,50 )

else:

    preds=model.predict(test_gen, batch_size=test_batch_size, verbose=0, steps=test_steps)

    
new_dict={} 

for key in class_dict: # set key in new_dict to value in class_dict and value in new_dict to key in class_dict

    value=class_dict[key]

    new_dict[value]=key

    b=' '

msg='{0:2s}{1:15s}{0:9s}{2:10s}{0:15s}{3:9s},{0:8s}{4}'.format(' ', 'Predicted Class', 'True Class', 'File Name', 'Errors are in Red') # adjust spacing based on your class names

print_in_color(msg, (0,255,0),(55,65,80))

error_file_list=[]

for i, p in enumerate(preds):

    pred_index=np.argmax(p) # get the index that has the highest probability

    if pred_index == tlabels[i]:

        foreground=(255,255,255)

    else:

        foreground=(255,0,0)

        error_file_list.append(file_names[i])

    pred_class=new_dict[pred_index]  # find the predicted class based on the index

    true_class=new_dict[tlabels[i]] # use the test label to get the true class of the test file

    file=file_names[i]

    msg=f' {pred_class:^15s}        {true_class:^15s}     {file:^25s}'

    if print_all==True or( print_all==False and foreground==(255,0,0)):

        print_in_color(msg, foreground, (55,65,80))
for f in error_file_list:

    print (f)