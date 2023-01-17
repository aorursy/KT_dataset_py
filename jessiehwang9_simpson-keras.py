import numpy as np
import cv2
import pandas as pd
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools 
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.cluster import KMeans
import itertools 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns

simpson_images = []
labels = [] 
for simpson_dir_path in glob.glob("/Users/hshen/Dropbox/data2/train/*"):
    simpson_label = simpson_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(simpson_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        simpson_images.append(image)
        labels.append(simpson_label)
simpson_images = np.array(simpson_images)
labels = np.array(labels)
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
id_to_label_dict
def plot_image_grid(images, nb_rows, nb_cols, figsize=(5, 5)):
    assert len(images) == nb_rows*nb_cols, "Number of images should be the same as (nb_rows*nb_cols)"
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)
    
    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            # axs[i, j].xaxis.set_ticklabels([])
            # axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].axis('off')
            axs[i, j].imshow(images[n])
            n += 1        
plot_image_grid(simpson_images[0:100], 10, 10)
label_ids = np.array([label_to_id_dict[x] for x in labels])
scaler = StandardScaler()
images_scaled = scaler.fit_transform([i.flatten() for i in simpson_images])
pca = PCA(n_components=50)
pca_result = pca.fit_transform(images_scaled)
#Train Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids, test_size=0.25, random_state=42)
#Define confusion matrix

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(X_train, y_train)
test_predictions = forest.predict(X_test)
precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with RandomForest: {0:.6f}".format(precision))
auc_test = roc_auc_score(y_test,  test_predictions)
print('AUC Test: %.2f' % auc_test)
probs=forest.predict_proba(X_test)
preds=probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, preds)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of Random Forest')
plt.legend()
plt.show()
cm = confusion_matrix(y_test, test_predictions)
cm_plot_labels = ['homer_simpson', 'lisa_simpson']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
#train svm
svm_clf = svm.SVC(gamma='auto', kernel='linear', probability=True)
svm_clf = svm_clf.fit(X_train, y_train) 
test_predictions1 = svm_clf.predict(X_test)
precision = accuracy_score(test_predictions1, y_test) * 100
print("Accuracy with SVM: {0:.6f}".format(precision))
auc_test = roc_auc_score(y_test,  test_predictions1)
print('AUC Test of SVM: %.2f' % auc_test)
probs=svm_clf.predict_proba(X_test)
preds=probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, preds)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of SVM')
plt.legend()
plt.show()
cm = confusion_matrix(y_test, test_predictions1)
cm_plot_labels = ['homer_simpson', 'lisa_simpson']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
train_path = '/Users/hshen/Dropbox/data2/train'
valid_path = '/Users/hshen/Dropbox/data2/val'
test_path = '/Users/hshen/Dropbox/data2/test'
train_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(train_path, target_size=(224,224), classes=['homer_simpson', 'lisa_simpson'], batch_size=500)
valid_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path, target_size=(224,224), classes=['homer_simpson', 'lisa_simpson'], batch_size=100)
test_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, target_size=(224,224), classes=['homer_simpson', 'lisa_simpson'], batch_size=100)
# plots images with Labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

sample_training_images, labels = next(train_data_gen)
plots(sample_training_images, titles=labels)
# Build and train CNN
total_train=2880
total_val=359
epochs=15
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history1 = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train //700,
    epochs=5,
    validation_data=valid_data_gen,
    validation_steps=total_val //100
)
acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']

loss = history1.history['loss']
val_loss = history1.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
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
plt.show()
model1 = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='sigmoid')
])
model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history2 = model1.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train //700,
    epochs=15,
    validation_data=valid_data_gen,
    validation_steps=total_val //100
)
acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']

loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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
plt.show()
#underfitting
#Visualize training images
sample_training_images, _ = next(train_data_gen)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(sample_training_images[:5])
#Apply horizontal flip
train_data_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True).flow_from_directory(train_path, target_size=(224,224), classes=['homer_simpson', 'lisa_simpson'], batch_size=500,shuffle=True)
valid_data_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True).flow_from_directory(valid_path, target_size=(224,224), classes=['homer_simpson', 'lisa_simpson'], batch_size=100,shuffle=True)
test_data_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True).flow_from_directory(test_path, target_size=(224,224), classes=['homer_simpson', 'lisa_simpson'], batch_size=100,shuffle=True)
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# Re-use the same custom plotting function defined and used
# above to visualize the training images
plotImages(augmented_images)
model2 = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='sigmoid')
])
model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history3 = model2.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train //700,
    epochs=15,
    validation_data=valid_data_gen,
    validation_steps=total_val //100
)
acc = history3.history['accuracy']
val_acc = history3.history['val_accuracy']

loss = history3.history['loss']
val_loss = history3.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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
plt.show()
#Put it all together
image_gen_train = ImageDataGenerator(rescale=1./255,
                                    rotation_range=45,
                                    width_shift_range=.15,
                                    height_shift_range=.15,
                                    horizontal_flip=True,
                                    zoom_range=0.5)
train_data_gen = image_gen_train.flow_from_directory(batch_size=700,
                                                    directory=train_path,
                                                    shuffle=True,
                                                    target_size=(224,224),
                                                    )
valid_data_gen=image_gen_train.flow_from_directory(batch_size=700,
                                                    directory=valid_path,
                                                    shuffle=True,
                                                    target_size=(224,224),
                                                    )
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
model3 = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='sigmoid')
])
model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history4 = model3.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train //700,
    epochs=15,
    validation_data=valid_data_gen,
    validation_steps=total_val //100
)
acc = history4.history['accuracy']
val_acc = history4.history['val_accuracy']

loss = history4.history['loss']
val_loss = history4.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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
plt.show()
#Build Fine-tuned VGG16 model
vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()
type(vgg16_model)
model4 = Sequential()
for layer in vgg16_model.layers:
    if layer.name != 'predictions':
        model4.add(layer)
model4.summary()
for layer in model4.layers:
    layer.trainable = False
model4.add(Dense(2, activation='softmax'))
model4.summary()
#Train the fine-tuned VGG16 model
model4.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history=model4.fit_generator(train_data_gen, steps_per_epoch=5,
                   validation_data=valid_data_gen, validation_steps=2, epochs=5, verbose=2)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
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
plt.show()
#Predict using fine-tuned VGG16 model by picture
image='/Users/hshen/Dropbox/kaggle_simpson_testset/homer_simpson_2.jpg'
my_image1=plt.imread(image)
img=plt.imshow(my_image1)
from skimage.transform import resize
my_image_resized1 = resize(my_image1, (224,224,3)) 
img1= plt.imshow(my_image_resized1) 
import numpy as np
probabilities1 = model3.predict(np.array( [my_image_resized1,] ))
probabilities1
number_to_class=['homer_simpson','lisa_simpson']
index = np.argsort(probabilities1[0,:])
print("Most likely class:", number_to_class[index[1]], "-- Probability:", probabilities1[0,index[1]])
print("Most likely class:", number_to_class[index[0]], "-- Probability:", probabilities1[0,index[0]])
#To save this model 
model.save('my_model.h5')
