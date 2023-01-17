import os
import numpy as np
import pandas as pd
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline
os.listdir('../input')
base_dir = 'base_dir'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

nv = os.path.join(train_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(train_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(train_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(train_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(train_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(train_dir, 'df')
os.mkdir(df)

nv = os.path.join(val_dir, 'nv')
os.mkdir(nv)
mel = os.path.join(val_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(val_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(val_dir, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(val_dir, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(val_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(val_dir, 'df')
os.mkdir(df)
df_data = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
df_data.head()
df = df_data.groupby('lesion_id').count()
df = df[df['image_id'] == 1]
df.reset_index(inplace=True)
df.head()
def identify_duplicates(x):
    unique_list = list(df['lesion_id'])
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'
    
df_data['duplicates'] = df_data['lesion_id']
df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)
df_data.head()
df_data['duplicates'].value_counts()
df = df_data[df_data['duplicates'] == 'no_duplicates']
df.shape
y = df['dx']
_, df_val = train_test_split(df, test_size=0.17, random_state=101, stratify=y)
df_val.shape
df_val['dx'].value_counts()
def identify_val_rows(x):
    val_list = list(df_val['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

df_data['train_or_val'] = df_data['image_id']
df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)
df_train = df_data[df_data['train_or_val'] == 'train']

print(len(df_train))
print(len(df_val))
df_train['dx'].value_counts()
df_val['dx'].value_counts()
df_data.set_index('image_id', inplace=True)
folder_1 = os.listdir('../input/skin-cancer-mnist-ham10000/ham10000_images_part_1')
folder_2 = os.listdir('../input/skin-cancer-mnist-ham10000/ham10000_images_part_2')

train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

for image in train_list:
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
        src = os.path.join('../input/skin-cancer-mnist-ham10000/ham10000_images_part_1', fname)
        dst = os.path.join(train_dir, label, fname)
        shutil.copyfile(src, dst)

    if fname in folder_2:
        src = os.path.join('../input/skin-cancer-mnist-ham10000/ham10000_images_part_2', fname)
        dst = os.path.join(train_dir, label, fname)
        shutil.copyfile(src, dst)

for image in val_list:
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
        src = os.path.join('../input/skin-cancer-mnist-ham10000/ham10000_images_part_1', fname)
        dst = os.path.join(val_dir, label, fname)
        shutil.copyfile(src, dst)

    if fname in folder_2:
        src = os.path.join('../input/skin-cancer-mnist-ham10000/ham10000_images_part_2', fname)
        dst = os.path.join(val_dir, label, fname)
        shutil.copyfile(src, dst)
print(len(os.listdir('base_dir/train_dir/nv')))
print(len(os.listdir('base_dir/train_dir/mel')))
print(len(os.listdir('base_dir/train_dir/bkl')))
print(len(os.listdir('base_dir/train_dir/bcc')))
print(len(os.listdir('base_dir/train_dir/akiec')))
print(len(os.listdir('base_dir/train_dir/vasc')))
print(len(os.listdir('base_dir/train_dir/df')))
print(len(os.listdir('base_dir/val_dir/nv')))
print(len(os.listdir('base_dir/val_dir/mel')))
print(len(os.listdir('base_dir/val_dir/bkl')))
print(len(os.listdir('base_dir/val_dir/bcc')))
print(len(os.listdir('base_dir/val_dir/akiec')))
print(len(os.listdir('base_dir/val_dir/vasc')))
print(len(os.listdir('base_dir/val_dir/df')))
class_list = ['mel','bkl','bcc','akiec','vasc','df']

for item in class_list:
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)
    img_class = item
    img_list = os.listdir('base_dir/train_dir/' + img_class)

    for fname in img_list:
        src = os.path.join('base_dir/train_dir/' + img_class, fname)
        dst = os.path.join(img_dir, fname)
        shutil.copyfile(src, dst)

    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    batch_size = 50
    aug_datagen = datagen.flow_from_directory(
        path,
        save_to_dir=save_path,
        save_format='jpg',
        target_size=(224,224),
        batch_size=batch_size)
    
    num_aug_images_wanted = 6000

    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted-num_files)/batch_size))

    for i in range(0,num_batches):
        imgs, labels = next(aug_datagen)

    shutil.rmtree('aug_dir')
print(len(os.listdir('base_dir/train_dir/nv')))
print(len(os.listdir('base_dir/train_dir/mel')))
print(len(os.listdir('base_dir/train_dir/bkl')))
print(len(os.listdir('base_dir/train_dir/bcc')))
print(len(os.listdir('base_dir/train_dir/akiec')))
print(len(os.listdir('base_dir/train_dir/vasc')))
print(len(os.listdir('base_dir/train_dir/df')))
print(len(os.listdir('base_dir/val_dir/nv')))
print(len(os.listdir('base_dir/val_dir/mel')))
print(len(os.listdir('base_dir/val_dir/bkl')))
print(len(os.listdir('base_dir/val_dir/bcc')))
print(len(os.listdir('base_dir/val_dir/akiec')))
print(len(os.listdir('base_dir/val_dir/vasc')))
print(len(os.listdir('base_dir/val_dir/df')))
def plots(ims, figsize=(12,6), rows=5, interp=False, titles=None): # 12,6
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
            
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
            
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
plots(imgs, titles=None)
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
datagen = ImageDataGenerator(preprocessing_function=tensorflow.keras.applications.mobilenet.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

test_batches = datagen.flow_from_directory(valid_path,
                                           target_size=(image_size,image_size),
                                           batch_size=1,
                                           shuffle=False)
mobile = tensorflow.keras.applications.mobilenet.MobileNet()
mobile.summary()
type(mobile.layers)
len(mobile.layers)
x = mobile.layers[-6].output
x = Dropout(0.20)(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=predictions)
model.summary()
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(Adam(lr=0.01), 
              loss='categorical_crossentropy', 
              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

print(valid_batches.class_indices)
class_weights = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 3.0,
    5: 1.0,
    6: 1.0,
}
filepath = "mobilenet.h5"

checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_top_3_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', 
                              factor=0.5, 
                              patience=2, 
                              verbose=1, 
                              mode='max', 
                              min_lr=0.00001)                     
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_batches, 
                              steps_per_epoch=train_steps, 
                              class_weight=class_weights,
                              validation_data=valid_batches,
                              validation_steps=val_steps,
                              epochs=30, 
                              verbose=1,
                              callbacks=callbacks_list)
model.metrics_names
val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches, 
                                                                               steps=len(df_val))

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
train_top2_acc = history.history['top_2_accuracy']
val_top2_acc = history.history['val_top_2_accuracy']
train_top3_acc = history.history['top_3_accuracy']
val_top3_acc = history.history['val_top_3_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training cat acc')
plt.plot(epochs, val_acc, 'b', label='Validation cat acc')
plt.title('Training and validation cat accuracy')
plt.legend()
plt.figure()


plt.plot(epochs, train_top2_acc, 'bo', label='Training top2 acc')
plt.plot(epochs, val_top2_acc, 'b', label='Validation top2 acc')
plt.title('Training and validation top2 accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, train_top3_acc, 'bo', label='Training top3 acc')
plt.plot(epochs, val_top3_acc, 'b', label='Validation top3 acc')
plt.title('Training and validation top3 accuracy')
plt.legend()

plt.show()
test_labels = test_batches.classes
test_labels
test_batches.class_indices
predictions = model.predict_generator(test_batches, steps=len(df_val), verbose=1)
predictions.shape
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
        plt.text(j, 
                 i, 
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
test_labels.shape
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
test_batches.class_indices
cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
y_pred = np.argmax(predictions, axis=1)
y_true = test_batches.classes
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)
input_shape = (224, 224, 3)
num_classes = 7

model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'Same', input_shape=input_shape))
model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'Same'))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))

model1.add(Conv2D(64, (3, 3), activation='relu', padding = 'Same'))
model1.add(Conv2D(64, (3, 3), activation='relu', padding = 'Same'))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))

model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(num_classes, activation='softmax'))
model1.summary()
train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

test_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model1.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
batch_size = 10
history1 = model1.fit_generator(train_batches,
                                epochs=30, 
                                validation_data=valid_batches,
                                validation_steps=val_steps,
                                verbose=1, 
                                class_weight=class_weights,
                                steps_per_epoch=train_steps)
acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.show()
predictions = model1.predict_generator(test_batches, steps=len(df_val), verbose=1)
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
y_pred = np.argmax(predictions, axis=1)
y_true = test_batches.classes
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)
model1.save('custom.h5')
resnet50 = tensorflow.keras.applications.resnet50.ResNet50(weights='imagenet')
resnet50.summary()
x = resnet50.layers[-2].output
x = Dropout(0.2)(x)
predictions = Dense(7, activation='softmax')(x)
model2 = Model(inputs=resnet50.input, outputs=predictions)
model2.summary()
datagen = ImageDataGenerator(preprocessing_function=tensorflow.keras.applications.resnet50.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

test_batches = datagen.flow_from_directory(valid_path,
                                           target_size=(image_size,image_size),
                                           batch_size=1,
                                           shuffle=False)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model2.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])

history2 = model2.fit_generator(train_batches,
                                epochs=30, 
                                validation_data=valid_batches,
                                validation_steps=val_steps,
                                verbose=1, 
                                class_weight=class_weights,
                                steps_per_epoch=train_steps)
acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.show()
predictions = model2.predict_generator(test_batches, steps=len(df_val), verbose=1)
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
y_pred = np.argmax(predictions, axis=1)
y_true = test_batches.classes
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)
model2.save('resnet50.h5')
inception = tensorflow.keras.applications.inception_v3.InceptionV3(weights='imagenet')
inception.summary()
x = inception.layers[-2].output
x = Dropout(0.2)(x)
predictions = Dense(7, activation='softmax')(x)
model3 = Model(inputs=inception.input, outputs=predictions)
model3.summary()
datagen = ImageDataGenerator(preprocessing_function=tensorflow.keras.applications.inception_v3.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(229,229),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(229,229),
                                            batch_size=val_batch_size)

test_batches = datagen.flow_from_directory(valid_path,
                                           target_size=(229,229),
                                           batch_size=1,
                                           shuffle=False)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model3.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])

history3 = model3.fit_generator(train_batches,
                                epochs=30, 
                                validation_data=valid_batches,
                                validation_steps=val_steps,
                                verbose=1, 
                                class_weight=class_weights,
                                steps_per_epoch=train_steps)
acc = history3.history['accuracy']
val_acc = history3.history['val_accuracy']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.show()
predictions = model3.predict_generator(test_batches, steps=len(df_val), verbose=1)
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
y_pred = np.argmax(predictions, axis=1)
y_true = test_batches.classes
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)
model3.save('inceptionV3.h5')
xception = tensorflow.keras.applications.xception.Xception(weights='imagenet')
xception.summary()
x = xception.layers[-2].output
x = Dropout(0.2)(x)
predictions = Dense(7, activation='softmax')(x)
model4 = Model(inputs=xception.input, outputs=predictions)
model4.summary()
datagen = ImageDataGenerator(preprocessing_function=tensorflow.keras.applications.xception.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(229,229),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(229,229),
                                            batch_size=val_batch_size)

test_batches = datagen.flow_from_directory(valid_path,
                                           target_size=(229,229),
                                           batch_size=1,
                                           shuffle=False)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model4.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])

history4 = model4.fit_generator(train_batches,
                                epochs=30, 
                                validation_data=valid_batches,
                                validation_steps=val_steps,
                                verbose=1, 
                                class_weight=class_weights,
                                steps_per_epoch=train_steps)
acc = history4.history['accuracy']
val_acc = history4.history['val_accuracy']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.show()
predictions = model4.predict_generator(test_batches, steps=len(df_val), verbose=1)
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
y_pred = np.argmax(predictions, axis=1)
y_true = test_batches.classes
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)
model4.save('xception.h5')
densenet121 = tensorflow.keras.applications.densenet.DenseNet121(weights='imagenet')
densenet121.summary()
x = densenet121.layers[-2].output
x = Dropout(0.2)(x)
predictions = Dense(7, activation='softmax')(x)
model5 = Model(inputs=densenet121.input, outputs=predictions)
model5.summary()
datagen = ImageDataGenerator(preprocessing_function=tensorflow.keras.applications.densenet.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

test_batches = datagen.flow_from_directory(valid_path,
                                           target_size=(image_size,image_size),
                                           batch_size=1,
                                           shuffle=False)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model5.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])

history5 = model5.fit_generator(train_batches,
                                epochs=30, 
                                validation_data=valid_batches,
                                validation_steps=val_steps,
                                verbose=1, 
                                class_weight=class_weights,
                                steps_per_epoch=train_steps)
acc = history5.history['accuracy']
val_acc = history5.history['val_accuracy']
loss = history5.history['loss']
val_loss = history5.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.show()
predictions = model5.predict_generator(test_batches, steps=len(df_val), verbose=1)
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
y_pred = np.argmax(predictions, axis=1)
y_true = test_batches.classes
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)

print(report)
model5.save('densenet121.h5')