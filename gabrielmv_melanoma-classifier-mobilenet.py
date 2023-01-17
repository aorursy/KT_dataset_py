from keras.models import Model

from keras.layers import Dense, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from keras.applications.mobilenet import MobileNet, preprocess_input

import numpy as np

import itertools

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt





def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()



def plot_training_curves(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    

    epochs = range(1, len(acc) + 1)

    

    plt.plot(epochs, loss, 'r', label='Training loss')

    plt.plot(epochs, val_loss, 'g', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    plt.figure()

    

    plt.plot(epochs, acc, 'r', label='Training acc')

    plt.plot(epochs, val_acc, 'g', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.figure()

    

    plt.show()



def print_results(cm):

    tp = cm[0, 0]

    tn = cm[1, 1]

    fn = cm[0, 1]

    fp = cm[1, 0]

    

    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100

    

    sensitivity = (tp / (tp + fn)) * 100

    

    specificity = (tn / ( tn + fp )) * 100

    

    print ('Accuracy: ',  accuracy)



    print ('Sensitivity: ', sensitivity)

    

    print ('Specificity: ',  specificity)



def fine_tune_mobile_net(train_batches, train_steps, class_weights, valid_batches, val_steps, file_path):

    mobile = MobileNet() # mobile.summary()

    

    x = mobile.layers[-6].output

    x = Dropout(0.5)(x)

    predictions = Dense(2, activation='softmax')(x)    

    

    model = Model(inputs = mobile.input, outputs = predictions)

    

    for layer in mobile.layers:

        layer.trainable = False

        

    model.compile(Adam(lr = 0.003), 

                  loss = 'categorical_crossentropy', 

                  metrics = ['accuracy'])

    

    model.fit_generator(train_batches,

                          steps_per_epoch = train_steps,

                          class_weight = class_weights,

                          validation_data = valid_batches,

                          validation_steps = val_steps,

                          epochs = 50,

                          verbose = 1,

                          callbacks = callbacks)

    



    model.load_weights(file_path)

    print ('*** Fine Tunning MobileNet ***')



    for layer in model.layers[:-23]:

        layer.trainable = False

        

    model.compile(Adam(lr = 0.003), 

                  loss = 'categorical_crossentropy', 

                  metrics = ['accuracy'])



    history = model.fit_generator(train_batches,

                          steps_per_epoch = train_steps,

                          class_weight = class_weights,

                          validation_data = valid_batches,

                          validation_steps = val_steps,

                          epochs = 75,

                          verbose = 1,

                          callbacks = callbacks)



    return model, history



def save_model(model, file_path):

    model_json = model.to_json()

    with open('model.json', 'w') as json_file:

        json_file.write(model_json)

    



file_path = 'weights-mobilenet-2.0.h5'



callbacks = [

        ModelCheckpoint(file_path, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max'),

        ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 8, verbose = 1, mode = 'min', min_lr = 0.00001),

        EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 15, verbose = 1)

        ]



training_path = '../input/dermmel/DermMel/train_sep'

validation_path = '../input/dermmel/DermMel/valid'

test_path = '../input/dermmel/DermMel/test'



num_train_samples = 10682

num_val_samples = 3562

num_test_samples = 3561



train_batch_size = 16

val_batch_size = 16

test_batch_size = 16



train_steps = np.ceil(num_train_samples / train_batch_size)

val_steps = np.ceil(num_val_samples / val_batch_size)

test_steps = np.ceil(num_val_samples / val_batch_size)



class_weights = {

        0: 4.1, # melanoma

        1: 1.0 # non-melanoma

}



train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(training_path,

                                    target_size = (224, 224),

                                    batch_size = val_batch_size,

                                    class_mode = 'categorical')



valid_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(validation_path,

                                        target_size = (224, 224),

                                        batch_size = val_batch_size,

                                        class_mode = 'categorical')



test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test_path,

                                          target_size = (224, 224),

                                          batch_size = test_batch_size,

                                          class_mode = 'categorical',

                                          shuffle = False)





model, history = fine_tune_mobile_net(train_batches, train_steps, class_weights, valid_batches, val_steps, file_path)





save_model(model, file_path)

model.load_weights(file_path)



test_labels = test_batches.classes



predictions = model.predict_generator(test_batches, steps = val_steps, verbose = 1)



cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_confusion_matrix(cm, ['melanoma', 'non-melanoma'])

print_results(cm)
plot_training_curves(history)