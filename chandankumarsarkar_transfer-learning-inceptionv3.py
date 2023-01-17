import tensorflow as tf

import numpy as np

import glob

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sn

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import load_img, img_to_array
train_img_gen = ImageDataGenerator(rescale=1/255., validation_split=0.2)

test_img_gen = ImageDataGenerator(rescale=1/255.)
# 80% of the Training data used for Training and 20% used for Validation

train_data = train_img_gen.flow_from_directory(directory='../input/intel-image-classification/seg_train/seg_train', target_size=(150, 150), batch_size=128, class_mode='binary', shuffle=True, subset='training')
valid_data = train_img_gen.flow_from_directory(directory='../input/intel-image-classification/seg_train/seg_train', target_size=(150, 150), batch_size=128, class_mode='binary', shuffle=True, subset='validation')
test_data = test_img_gen.flow_from_directory(directory='../input/intel-image-classification/seg_test/seg_test', target_size=(150, 150), batch_size=1, class_mode='binary', shuffle=False)
class_labels = train_data.class_indices

class_labels
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
base_model.trainable=False
out_incep = base_model.output
gavpooling = tf.keras.layers.GlobalAveragePooling2D()(out_incep)
d1 = tf.keras.layers.Dense(units = 64, activation = 'relu')(gavpooling)
output = tf.keras.layers.Dense(units = 6, activation = 'softmax')(d1)
model = tf.keras.models.Model(base_model.input, output)
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), 

              loss=tf.keras.losses.sparse_categorical_crossentropy,

              metrics=tf.keras.metrics.sparse_categorical_accuracy)
train_data.reset()

history = model.fit(x=train_data, epochs=5, validation_data=valid_data)
# Training Loss and Validation Loss versus Epochs plot. At 5 epochs validation loss almost reaches plateau/flattens.

plt.plot([1,2,3,4,5], history.history['loss'], 'bo--', label='Train Loss')

plt.plot([1,2,3,4,5], history.history['val_loss'], 'ro--', label='Val Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
# Training Accuracy and Validation Accuracy versus Epochs plot. At 5 epochs validation accuracy almost reaches plateau/flattens.

plt.plot([1,2,3,4,5], history.history['sparse_categorical_accuracy'], 'bo--', label='Train Accuracy')

plt.plot([1,2,3,4,5], history.history['val_sparse_categorical_accuracy'], 'ro--', label='Val Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()

plt.show()
# Accuracy aghieved on the Test Data is better than the Validation and it is very close to the Training accuracy

test_data.reset()

eval_loss, eval_metrics = model.evaluate(test_data)
# Printing the Classification Report on Test Data

test_data.reset()

preds = model.predict(test_data)

test_preds = np.argmax(preds, axis=1)

print(classification_report(test_data.classes, test_preds, target_names=test_data.class_indices.keys()))
# Printing the Confusion Matrix on Test Data

cm = confusion_matrix(test_data.classes, test_preds)

sn.heatmap(cm, annot=True, cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())

plt.show()
# # Based on above results on Test Data it is evident that the model has most of the errors in the 

# Between 'Buildings' and 'Streets'

# Between 'Glacier' and 'Mountain'

# Between 'Glacier' and 'Sea'
# Dict to capture the labels and their text names

labels_dict = {0: 'buildings',

 1: 'forest',

 2: 'glacier',

 3: 'mountain',

 4: 'sea',

 5: 'street'}
# To investigate the mismatched classes in more detail, we will open the images which got incorrect predictions

# function takes input the names of actual class and predicted class

def show_missclassified (actual_classname, predicted_classname, test_inputdata, test_predictions):



    count = 0

    filepaths = []



    for i in range(test_inputdata.classes.shape[0]):

        if labels_dict.get(test_inputdata.classes[i]) == actual_classname and labels_dict.get(test_predictions[i]) == predicted_classname:

            count +=1

            filepaths.append(test_inputdata.filepaths[i])



    

    test_images = []

    for p in filepaths:

        img = load_img(path=p, grayscale=False, target_size=(150, 150))

        img = img_to_array(img, data_format='channels_last')

        img = img / 255.

        test_images.append(img)

    test_images = np.array(test_images)



    print('Actual Class: {0},  Predicted Class: {1}'.format(actual_classname, predicted_classname))

    print('Number of missclassified images: {}'.format(count))

    

    if count > 0:

        ncols = 7 if count > 7 else count

        nrows = int(count/ncols)+1

        fig, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15,15))





        img_count = 0

        for i in range(nrows):

            for j in range(ncols):

                if img_count < count:

                    ax[i,j].imshow(test_images[img_count], cmap=plt.cm.binary)

                    ax[i,j].set_title('pred = '+predicted_classname)

                    ax[i,j].axis('off')

                    img_count +=1

        plt.show()

    return
# To see the images with Actual Class Label: street,  Predicted Class Label: buildings

show_missclassified('street', 'buildings', test_data, test_preds)
# To see the another example of images with Actual Class Label: glacier,  Predicted Class Label: mountain

show_missclassified('glacier', 'mountain', test_data, test_preds)
# Loading the prediction folder images

pred_img_paths = glob.glob(pathname='../input/intel-image-classification/seg_pred/seg_pred'+'/*.jpg')

pred_images = []

for p in pred_img_paths:

    img = load_img(path=p, grayscale=False, target_size=(150, 150))

    img = img_to_array(img, data_format='channels_last')

    img = img / 255.

    pred_images.append(img)

pred_images = np.array(pred_images)
# Loaded input shape

pred_images.shape
# Capture the file names of the images in the Pred folder

pred_file_names = [tf.strings.split(i, sep='/')[-1].numpy() for i in pred_img_paths]
# Predict the classes for the input images

pred_results = model.predict(pred_images, batch_size=1)
# Plot sample results from the predictions.  Format = Predicted Class Name (File Name)

nrows = 7

ncols = 10

fig, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=(30,20))



img_start = 6000

img_count = 0

for i in range(nrows):

    for j in range(ncols):

        ax[i,j].imshow(pred_images[img_start + img_count], cmap=plt.cm.binary)

        ax[i,j].axis('off')

        ax[i,j].set_title(labels_dict.get(np.argmax(pred_results[img_start + img_count]))+' ('+str(pred_file_names[img_start + img_count])+')')

        img_count +=1

plt.show()