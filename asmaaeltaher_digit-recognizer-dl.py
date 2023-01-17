import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.optimizers import RMSprop



# visualization
import seaborn as sns
import matplotlib.pyplot as plt, matplotlib.image as mpimg
%matplotlib inline

#load train data 
train = pd.read_csv('../input/digit-recognizer/train.csv')

# sprate imgs from labels
train_imgs = train.drop(['label'], axis=1)
train_labels = train['label']
def prepare_imgs(imgs):
    #reshape to (28,28) 
    imgs = np.reshape(imgs.to_numpy(), (-1,28,28))
    #scale the values to be [0,1]
    imgs = imgs/255.
    
    return imgs
    
    
def prepare_lbls(lbls):
    #labels to categories
    lbl_cats = to_categorical(lbls, num_classes = 10)
    
    return lbl_cats


def split_data(X, Y, tst_sz, val_sz=0):
    # to train & tst
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, Y, test_size =tst_sz, random_state=1)
    
    # to train & test & validation
    if val_sz != 0:
        X_val, X_tst, y_val, y_tst = train_test_split(X_tst, y_tst, test_size =0.5, random_state=1)
        return X_trn, X_val, X_tst, y_trn, y_val, y_tst
    
    return X_trn, X_tst, y_trn, y_tst


def prepare(imgs, lbls=0, tst_sz=0, val_sz=0, to_tensor=0):
    # prepare imgs
    X = prepare_imgs(imgs)
    
    # prepare labels if lbls has values
    if type(lbls)==pd.Series:
        Y = prepare_lbls(lbls)
    
    
    # if we want to split the data 
    if tst_sz!=0:
        # split to 2 parts
        if val_sz==0:
            X_trn, X_tst, y_trn, y_tst = split_data(X, Y, tst_sz, val_sz=0)
            # to tensor
            if to_tensor == 1:
                X_trn = tf.convert_to_tensor(X_trn[...,None])
                X_tst = tf.convert_to_tensor(X_tst[...,None])
                y_trn = tf.convert_to_tensor(y_trn)
                y_tst = tf.convert_to_tensor(y_tst)
            return X, Y, X_trn, X_tst, y_trn, y_tst
        # split to 3 parts
        else:
            X_trn, X_val, X_tst, y_trn, y_val, y_tst = split_data(X, Y, tst_sz, val_sz=1)
            # to tensor
            if to_tensor == 1:
                X_trn = tf.convert_to_tensor(X_trn[...,None])
                X_val = tf.convert_to_tensor(X_val[...,None])
                X_tst = tf.convert_to_tensor(X_tst[...,None])
                y_trn = tf.convert_to_tensor(y_trn)
                y_val = tf.convert_to_tensor(y_val)
                y_tst = tf.convert_to_tensor(y_tst)
            return X, Y, X_trn, X_val, X_tst, y_trn, y_val, y_tst
        
    #if we will work with the total dataset
    if to_tensor == 1:
        X = tf.convert_to_tensor(X[...,None])
        
    if type(lbls)!=pd.Series:
        return X
    return X, Y
# prepare and split data
X, Y, X_trn, X_val, X_tst, y_trn, y_val, y_tst = prepare(train_imgs, train_labels, tst_sz=0.2, val_sz=1, to_tensor=1)
def vis_sample(imgs, lbls):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Sample Data', fontsize=16)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.reshape(imgs[i], (28,28)))
        plt.title(lbls[i].argmax())
        plt.axis("off")
        plt.show;
        
def lbls_hist(lbls):
    hist,bin_edges = np.histogram(lbls)

    plt.figure(figsize=[10,8])
    plt.bar(bin_edges[:-1], hist, width = 0.5, color='#0504aa',alpha=0.7)
    plt.xlabel('Label',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Labels Distribution Histogram',fontsize=15)
    plt.show();
    
def explore_data(imgs, lbls):
    # shape and dtype
    print(f'Data shape is: {train_imgs.shape} and the data type is {type(train_imgs)}')
    
    # sample data vis
    vis_sample(imgs, lbls)
    
    # labels hist
    lbls_hist(lbls.argmax(1))
explore_data(X, Y)
def data_aug(X):
    print("In Data Genrator Step")
    datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range=10, 
            zoom_range = 0.1, 
            width_shift_range=0.1,  
            height_shift_range=0.1,  
            horizontal_flip=False,  
            vertical_flip=False)  

    datagen.fit(X)
    return datagen
    

def create_model(input_shape):
    print("In Model Creating Step")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), activation='relu',input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5,5), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation= 'relu'),
        tf.keras.layers.Dense(10,activation= 'softmax')
    ])
    
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    print(model.summary())
    return model
    

def fit_model(model, X_trn, y_trn, n_epochs, X_tst, X_val=0, y_val=0, val_split=0, datagen=0):
    if datagen == 0:
        # without aug
        if val_split != 0:
            # using val split 
            print("In Model Fitting Step (using val split)")
            history = model.fit(
                X_trn, y_trn, validation_split =val_split, 
                epochs=n_epochs
            )
        else:
            #using val set
            print("In Model Fitting Step (using val set)")
            history = model.fit(
                X_trn, y_trn, validation_data = (X_val, y_val), 
                epochs=n_epochs
            )
    else:
        # using aug 
        if  type(X_val)==tf.python.framework.ops.EagerTensor:
            print("In Model Fitting Step (using val set & fit generator)")
            history = model.fit_generator(
                datagen.flow(X_trn, y_trn), validation_data = (X_val, y_val) ,
                epochs=n_epochs
            )
        else :
            print("In Model Fitting Step (without val & using fit generator)")
            history = model.fit_generator(
                datagen.flow(X_trn, y_trn),
                epochs=n_epochs
            )
        
    print("In Model Prediction Step")
    y_preds = model.predict(X_tst) 
    return history, y_preds


def vis_accuracy(history, n_epochs):
    print("In Accuracy Visualization Step")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(n_epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show();
    

def check_accuracy(y_true, y_preds):
    print(f'The model accuracy is {accuracy_score(y_tst, (y_preds > 0.5))}.')
    

def check_confusion(y_true, y_preds):
    plt.figure(figsize=(16, 8))
    plt.title('Confusion Matrix')
    sns.heatmap(confusion_matrix(y_true.numpy().argmax(1), y_preds.argmax(1)))
    plt.show();
    

def run_model(X_trn, y_trn, n_epochs, X_tst, y_tst=0, X_val=0, y_val=0, val_split=0, aug=0, final_model =0):
    #create model
    model = create_model(X_trn.shape[1:])
    
    # create data augmentation layer
    datagen = 0
    if aug == 1:
        datagen = data_aug(X_trn)
    
    #fit & predict
    history, y_preds = fit_model(model, X_trn, y_trn, n_epochs, X_tst, X_val, y_val, val_split, datagen)
    
    #if it's not the final model "we have true y"
    if final_model == 0:
        #vis accuracy curve
        vis_accuracy(history, n_epochs)
        #check confusion matrix
        check_confusion(y_tst, y_preds)
        #check total accuracy
        check_accuracy(y_tst, y_preds)
    
    return y_preds
    
# train using val split
y_preds = run_model(X_trn, y_trn, 10, X_tst, y_tst, val_split=0.2)
# tain using val & test sets
tst_y_preds_aug = run_model(X_trn, y_trn, 10, X_tst, y_tst, X_val, y_val, val_split=0, aug=1)
#load test data
test = pd.read_csv('../input/digit-recognizer/test.csv')
# prepare full training data 
X, y = prepare(train_imgs, train_labels, tst_sz=0, val_sz=0, to_tensor=1)
X_test = prepare(test, to_tensor=1)
# train in full train data and pridect on test data 
test_y_preds = run_model(X, y, 30, X_test, aug=1, final_model=1)
len(test_y_preds)
submission = pd.DataFrame({"ImageId": list(range(1,len(test_y_preds)+1)),
                         "Label": test_y_preds.argmax(1)})

submission.to_csv("Submission.csv",index=False)
