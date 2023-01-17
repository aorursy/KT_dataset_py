import pandas as pd

import numpy as np

import os

import shutil

import random

import time

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from skimage.transform import resize



from keras import models

from keras import layers

from keras import optimizers

from keras.preprocessing import image

from keras.callbacks import EarlyStopping



random.seed(99)  # for reproducibility
dataset_dir = '/kaggle/input/dog-breed-identification/train'

test_dataset_dir = '/kaggle/input/50-corgi-pictures/'

train_labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')

test_labels = pd.DataFrame({'id': os.listdir(test_dataset_dir), 'breed': 'pembroke'})



# helper function to create directory for the script not throwing an error if the 

def make_dir(x):

    if not os.path.exists(x):

        os.makedirs(x)



# directory where we’ll store our dataset subset for keras generators to read from

base_dir = '/kaggle/working/dog-breed-identification/subsets/'

make_dir(base_dir)



breeds = list(train_labels.breed.unique())
print(train_labels.head())
train_frac = 0.8
train_img_fnames = []

validation_img_fnames = []

test_img_fnames = []



# directories for the training, validation, and test images

train_dir = os.path.join(base_dir, 'train')

make_dir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

make_dir(validation_dir)

test_dir = os.path.join(base_dir, 'test')

make_dir(test_dir)



breed_counts = train_labels.breed.value_counts()



# loop through breeds for training and validation

for breed in breeds:

    # make a directory for each breed

    train_breed_dir = os.path.join(train_dir, breed)

    validation_breed_dir = os.path.join(validation_dir, breed)

    

    make_dir(train_breed_dir)

    make_dir(validation_breed_dir)

    

    # get training count

    n_train = int(breed_counts[breed] * train_frac)

    i = 0

    

    # get ids for training and validation by breeds

    breed_ids = train_labels[train_labels.breed == breed].id

    breed_train = breed_ids.sample(n=n_train, random_state=57)

    breed_validation = breed_ids[~breed_ids.isin(breed_train)]



    # transfer doggo images to these folders accordingly

    for dog in breed_train:

        i+=1

        src = os.path.join(dataset_dir, dog + '.jpg')

        dst = os.path.join(train_breed_dir, dog + '.jpg')

        shutil.copyfile(src, dst)

        train_img_fnames.append(dog)



    for dog in breed_validation:

        i+=1

        src = os.path.join(dataset_dir, dog + '.jpg')

        dst = os.path.join(validation_breed_dir, dog + '.jpg')

        shutil.copyfile(src, dst)

        validation_img_fnames.append(dog)
print(f'Source: {src}\nDestination: {dst}')
train_labels.loc[train_labels.id == os.path.splitext(src)[0].split('/')[-1]]
# loop through 'breeds' for testing

test_breed_dir = os.path.join(test_dir, 'pembroke')

make_dir(test_breed_dir)

for corgi_img in test_labels.id:

    src = os.path.join(test_dataset_dir, corgi_img)

    dst = os.path.join(test_breed_dir, corgi_img)

    shutil.copyfile(src, dst)

    test_img_fnames.append(os.path.splitext(corgi_img)[0])
print(f'Source: {src}\nDestination: {dst}')
breed_color = [['red' if (x == 'pembroke') else 'lightgrey' for x in breed_counts.index]] # attention to double []

pd.DataFrame(breed_counts).plot.bar(color=breed_color, width=0.8, figsize=(21, 5))
train_labels.loc[train_labels.breed == 'pembroke', 'id']
# pick 9 random pembroke images from training data

pembrokes = train_labels.loc[train_labels.breed == 'pembroke', 'id']



train_img_sample = random.sample(pembrokes.tolist(), 9)

read_train_imgs = [mpimg.imread(os.path.join(dataset_dir, x + '.jpg')) for x in train_img_sample]



fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

for i, v in enumerate(fig.axes):

    v.imshow(read_train_imgs[i])

    v.text(x=read_train_imgs[i].shape[1]/2, y=read_train_imgs[i].shape[1]/40, s=train_img_sample[i], bbox=dict(facecolor='white', alpha=0.9), ha='center', va='top', size=11)

plt.tight_layout()

plt.show()
# take a look at the pictures of Ace

def resize_pic(x):

    return resize(x, (x.shape[0] // 8, x.shape[1] // 8), anti_aliasing=True)



test_img_sample = random.sample(test_img_fnames, 9)

read_test_imgs = [resize_pic(mpimg.imread(os.path.join(test_dataset_dir, x + '.jpg'))) for x in test_img_sample]



fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

for i, v in enumerate(fig.axes):

    v.imshow(read_test_imgs[i])

    v.text(x=read_test_imgs[i].shape[1]/2, y=read_test_imgs[i].shape[1]/40, s=test_img_sample[i], bbox=dict(facecolor='white', alpha=0.9), ha='center', va='top', size=14)

plt.tight_layout()

plt.show()
# variables used by all models

batch_size = 20

# early stopping to stop training after validation loss stops improving

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)



# helper function with the visualization



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc,  color='#008080', mec='k', label='Training accuracy')

    plt.plot(epochs, val_acc, color='#FFA500', label='Validation accuracy')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, color='#008080', label='Training loss')

    plt.plot(epochs, val_loss, color='#FFA500', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    plt.show()
results = []



def test_model(model_name, model, train_time, history, features=None, generator=None):

    if generator is None:

        predictions = model.predict(features)

    else:

        predictions = model.predict_generator(generator, steps = generator.n)

    pred = []

    truth = [86 for i in range(50)]

    for i in range(0, len(predictions)):

        pred.append(np.argmax(predictions[i]))

    acc_bool = [x == y for x, y in zip(pred, truth)]

    acc = round(sum(acc_bool) / len(acc_bool) * 100, 1)

    valid_loss = min(history.history['val_loss'])

    print(f'TEST ACCURACY\n\t{acc}% ({sum(acc_bool)}/{len(acc_bool)} correct)')

    train_time = round(train_time / 60, 1)

    return {'Model Name': model_name, 'Test Accuracy %': acc, 'Validation Loss': valid_loss, 'Training Time (minutes)': train_time}
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

input_shape=(224, 224, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(120, activation='softmax')) # 120 different breeds



model.summary()



optimizer = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=optimizer,

    loss='sparse_categorical_crossentropy',

    metrics=['acc'])



datagen = image.ImageDataGenerator(rescale=1./255) # rescale pixel values to [0, 1] interval



train_generator = datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224),

    batch_size=batch_size,

    class_mode='sparse')

        

validation_generator = datagen.flow_from_directory(

    validation_dir,

    target_size=(224, 224),

    batch_size=batch_size,

    class_mode='sparse',

    classes=train_generator.class_indices)



test_generator = datagen.flow_from_directory(

    test_dir,

    target_size=(224, 224),

    batch_size=1, # predict one picture at a time

    class_mode='sparse',

    classes=train_generator.class_indices)



# dummy check if all class indices are the same in all generators

print(f'\nAll classes same: {train_generator.class_indices == test_generator.class_indices == validation_generator.class_indices}')
start_clock = time.clock()

history = model.fit_generator(

        train_generator,

        steps_per_epoch=int(train_generator.n / train_generator.batch_size), # matching the number of samples, 8127/20  

        epochs=100,

        validation_data=validation_generator,

        validation_steps=int(validation_generator.n / validation_generator.batch_size),

        callbacks=[es],

        verbose=0)

end_clock = time.clock()

train_time = end_clock - start_clock
plot_history(history)

results.append(test_model('CNN', model, train_time, history, generator=test_generator))
aug_datagen = image.ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=False,

    brightness_range=[0.2, 1.0],

    fill_mode='nearest')
my_img = os.path.join(dataset_dir, random.sample(train_labels[train_labels['breed'] == 'pembroke'].id.tolist(), 1)[0] + '.jpg')

my_img = image.load_img(my_img, target_size=(224, 224))

my_img = image.img_to_array(my_img)



my_img = my_img.reshape((1,) + my_img.shape)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15,5))

for batch, (i, v) in zip(aug_datagen.flow(my_img, batch_size=1), enumerate(fig.axes)):

    v.imshow(image.array_to_img(batch[0]))

    if i == 3:

        break

plt.tight_layout()

plt.show()
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

input_shape=(224, 224, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))

model.add(layers.Dropout(0.2)) # it's now drop-rate instead of keep-rate

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(120, activation='softmax'))



model.summary()



optimizer = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=optimizer,

    loss='sparse_categorical_crossentropy',

    metrics=['acc'])
train_generator_aug = aug_datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224),

    batch_size=batch_size,

    class_mode='sparse')
start_clock = time.clock()

history = model.fit_generator(

        train_generator_aug,

        steps_per_epoch=int(train_generator_aug.n / train_generator_aug.batch_size) * 3, # every image augmented three times

        epochs=100,

        validation_data=validation_generator,

        validation_steps=int(validation_generator.n / validation_generator.batch_size),

        callbacks=[es],

        verbose=0)

end_clock = time.clock()

train_time = end_clock - start_clock
plot_history(history)

results.append(test_model('CNN+reg+aug', model, train_time, history, generator=test_generator))
def extract_features(directory, sample_count, x, y, z, target_size):

    global class_dictionary

    global filenames

    features = np.zeros(shape=(sample_count, x, y, z))

    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(

        directory,

        target_size=target_size,

        batch_size=batch_size,

        class_mode='sparse',

        shuffle=False,

        classes=train_generator.class_indices)

    i = 0

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        #print(i)

        if i * batch_size >= sample_count:

            break

    class_dictionary = generator.class_indices

    filenames = generator.filenames

    return features, labels



def count_files(input_dir):  # counts files from subdirs

    path = input_dir

    n = 0

    folders = ([name for name in os.listdir(path) 

                if os.path.isdir(os.path.join(path, name))])  # get all directories 

    for folder in folders:

        contents = os.listdir(os.path.join(path,folder))  # get list of contents

        

        n += len(contents)

    return(n)



train_n = count_files(train_dir)

validation_n = count_files(validation_dir)

test_n = count_files(test_dir)
from keras.applications import VGG19

conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base.layers[-1].output_shape
hw = conv_base.layers[-1].output_shape[1]  # height/width

de = conv_base.layers[-1].output_shape[-1]  # depth

ts = 224  # size
print(f'NUMBER OF IMAGE FILES\nTrain: {train_n}\nValidation: {validation_n}\nTest: {test_n}')



train_features, train_labels_arr = extract_features(train_dir, train_n, hw, hw, de, target_size=(ts, ts))

validation_features, validation_labels_arr = extract_features(validation_dir, validation_n, hw, hw, de, target_size=(ts, ts))

test_features, test_labels_arr = extract_features(test_dir, test_n, hw, hw, de, target_size=(ts, ts))



train_features = np.reshape(train_features, (train_n, hw * hw * de))

validation_features = np.reshape(validation_features, (validation_n, hw * hw * de))

test_features = np.reshape(test_features, (test_n, hw * hw * de))
model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_dim=hw * hw * de))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(120, activation='softmax'))

optimizer = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=optimizer,

    loss='sparse_categorical_crossentropy',

    metrics=['acc'])



start_clock = time.clock()

history = model.fit(train_features, train_labels_arr,

    epochs=100,

    batch_size=batch_size,

    validation_data=(validation_features, validation_labels_arr),

    callbacks=[es],

    verbose=0)

end_clock = time.clock()

train_time = end_clock - start_clock
plot_history(history)

results.append(test_model('VGG19', model, train_time, history, features=test_features))
from keras.applications import InceptionResNetV2

conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

conv_base.layers[-1].output_shape
hw = conv_base.layers[-1].output_shape[1]  # height/width

de = conv_base.layers[-1].output_shape[-1]  # depth

ts = 299  # size
print(f'NUMBER OF IMAGE FILES\nTrain: {train_n}\nValidation: {validation_n}\nTest: {test_n}')



train_features, train_labels_arr = extract_features(train_dir, train_n, hw, hw, de, target_size=(ts, ts))

validation_features, validation_labels_arr = extract_features(validation_dir, validation_n, hw, hw, de, target_size=(ts, ts))

test_features, test_labels_arr = extract_features(test_dir, test_n, hw, hw, de, target_size=(ts, ts))



train_features = np.reshape(train_features, (train_n, hw * hw * de))

validation_features = np.reshape(validation_features, (validation_n, hw * hw * de))

test_features = np.reshape(test_features, (test_n, hw * hw * de))
model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_dim=hw * hw * de))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(120, activation='softmax'))

optimizer = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=optimizer,

              loss='sparse_categorical_crossentropy',

              metrics=['acc'])



start_clock = time.clock()

history = model.fit(train_features, train_labels_arr,

    epochs=100,

    batch_size=batch_size,

    validation_data=(validation_features, validation_labels_arr),

    callbacks=[es],

    verbose=0)

end_clock = time.clock()

train_time = end_clock - start_clock
plot_history(history)

results.append(test_model('Inception-ResNet-V2', model, train_time, history, features=test_features))
test_predictions = model.predict(test_features)
all_labels = dict((v,k) for k,v in test_generator.class_indices.items())



test_ind_labels = np.argsort(-test_predictions, axis=1)[:, :5]

pred_labels = []

for i, v in enumerate(test_ind_labels):

    pred_labels.append([all_labels[x] for x in v])

pred_probs = [test_predictions[i][test_ind_labels[i]] for i in range(0, test_n)]



read_test_imgs = []

for i, v in enumerate(test_img_fnames[0:20]):

    read_test_imgs.append(resize_pic(mpimg.imread(os.path.join(test_dataset_dir, test_img_fnames[i] + '.jpg'))))
pics_per_row = 2

pred_n = len(read_test_imgs)

fig, axes = plt.subplots(nrows=pred_n // pics_per_row, ncols=pics_per_row * 2, figsize=(pics_per_row * 9, pred_n * 1.5))

r = -1

c = 0

for i, v in enumerate(read_test_imgs):

    if i % pics_per_row*2 == 0:

        r += 1

        c = 0

    axes[r,c].barh([str(x) for x in pred_labels[i]], pred_probs[i],

    color=['royalblue' if x == 'pembroke' else 'darkgrey' for x in pred_labels[i]], edgecolor='black')

    axes[r,c].set(xlim=(0,1))

    axes[r,c].set_aspect(0.2, anchor='W')

    axes[r,c].invert_yaxis()

    for a in range(len(pred_probs[i])):

        axes[r,c].text(x=pred_probs[i][a], y=a,

            s=str(round(pred_probs[i][a] * 100, 1)) + '%',

            verticalalignment='center',

            horizontalalignment='right' if pred_probs[i][a] > 0.25 else 'left', size = 16)

    axes[r,c].set_title('Predicted')

    axes[r,c].set_anchor('E')

    axes[r,c].tick_params(axis='y', labelsize=16)

    c += 1

    axes[r,c].imshow(v)

    axes[r,c].set_title(test_img_fnames[i])

    axes[r,c].set_anchor('W')

    c += 1

plt.tight_layout()



plt.show()
result_table = pd.DataFrame(results)

print(result_table)
# delete all subdirs (needed for not throwing an error for too many files)

shutil.rmtree(base_dir)