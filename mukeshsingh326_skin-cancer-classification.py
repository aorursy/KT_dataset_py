import shutil

import numpy as np

import pandas as pd

from random import random



# Image operations and plotting

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

%matplotlib inline



# File, path and directory operations

import os

import os.path

import shutil





# Model building

from fastai.vision import *

from fastai.callbacks.hooks import *

import torchvision

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

import seaborn as sns

from sklearn.metrics import classification_report

from pathlib import PurePath



# For reproducability

from numpy.random import seed

seed(108)

print(os.listdir("../input/skin-cancer-mnist-ham10000"))
# Create a new directory

base = "base"

os.mkdir(base)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]



# now we create 7 folders inside 'base':



# train

    # nv

    # mel

    # bkl

    # bcc

    # akiec

    # vasc

    # df

 

# valid

    # nv

    # mel

    # bkl

    # bcc

    # akiec

    # vasc

    # df



# create a path to 'base' to which we will join the names of the new folders

# train

train = os.path.join(base, 'train')

os.mkdir(train)



# valid

valid = os.path.join(base, 'valid')

os.mkdir(valid)





# [CREATE FOLDERS INSIDE THE TRAIN, VALIDATION AND TEST FOLDERS]

# Inside each folder we create seperate folders for each class



# create new folders inside train

nv = os.path.join(train, 'nv')

os.mkdir(nv)

mel = os.path.join(train, 'mel')

os.mkdir(mel)

bkl = os.path.join(train, 'bkl')

os.mkdir(bkl)

bcc = os.path.join(train, 'bcc')

os.mkdir(bcc)

akiec = os.path.join(train, 'akiec')

os.mkdir(akiec)

vasc = os.path.join(train, 'vasc')

os.mkdir(vasc)

df = os.path.join(train, 'df')

os.mkdir(df)











# test

test = os.path.join(base, 'test')

os.mkdir(test)



nv = os.path.join(test, 'nv')

os.mkdir(nv)

mel = os.path.join(test, 'mel')

os.mkdir(mel)

bkl = os.path.join(test, 'bkl')

os.mkdir(bkl)

bcc = os.path.join(test, 'bcc')

os.mkdir(bcc)

akiec = os.path.join(test, 'akiec')

os.mkdir(akiec)

vasc = os.path.join(test, 'vasc')

os.mkdir(vasc)

df = os.path.join(test, 'df')

os.mkdir(df)

# create new folders inside valid

nv = os.path.join(valid, 'nv')

os.mkdir(nv)

mel = os.path.join(valid, 'mel')

os.mkdir(mel)

bkl = os.path.join(valid, 'bkl')

os.mkdir(bkl)

bcc = os.path.join(valid, 'bcc')

os.mkdir(bcc)

akiec = os.path.join(valid, 'akiec')

os.mkdir(akiec)

vasc = os.path.join(valid, 'vasc')

os.mkdir(vasc)

df = os.path.join(valid, 'df')

os.mkdir(df)
import pandas as pd

df = pd.read_csv("../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")

from numpy.random import seed

seed(101)

df2=df.iloc[:,1:3]

msk = np.random.rand(len(df2)) < 0.85

train1_df2 = df2[msk]

test_df2 = df2[~msk]

msk1 = np.random.rand(len(train1_df2)) < 0.85

train_df2 = train1_df2[msk1]

validation_df2 = train1_df2[~msk1]
train_df2['dx'].value_counts()
validation_df2['dx'].value_counts()
test_df2['dx'].value_counts()
# Set the image_id as the index in df_data

df.set_index('image_id', inplace=True)
# Get a list of images in each of the two folders

folder_1 = os.listdir('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1')

folder_2 = os.listdir('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2')
# Get a list of train , val and test images 

train_df2_list = list(train_df2['image_id'])

validation_df2_list = list(validation_df2['image_id'])

test_df2_list = list(test_df2['image_id'])
# Transfer the train images



for image in train_df2_list:

    

    fname = image + '.jpg'

    label = df.loc[image,'dx']

    

    if fname in folder_1:

        # source path to image

        src = os.path.join('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1', fname)

        # destination path to image

        dst = os.path.join(train, label, fname)

        # copy the image from the source to the destination

        shutil.copyfile(src, dst)

    if fname in folder_2:

        # source path to image

        src = os.path.join('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2', fname)

        # destination path to image

        dst = os.path.join(train, label, fname)

        # copy the image from the source to the destination

        shutil.copyfile(src, dst)
# Transfer the val images



for image in validation_df2_list:

    

    fname = image + '.jpg'

    label = df.loc[image,'dx']

    

    if fname in folder_1:

        # source path to image

        src = os.path.join('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1', fname)

        # destination path to image

        dst = os.path.join(valid, label, fname)

        # copy the image from the source to the destination

        shutil.copyfile(src, dst)

    if fname in folder_2:

        # source path to image

        src = os.path.join('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2', fname)

        # destination path to image

        dst = os.path.join(valid, label, fname)

        # copy the image from the source to the destination

        shutil.copyfile(src, dst)

   

        
for image in test_df2_list:

    

    fname = image + '.jpg'

    label = df.loc[image,'dx']

    

    if fname in folder_1:

        # source path to image

        src = os.path.join('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1', fname)

        # destination path to image

        dst = os.path.join(test, label, fname)

        # copy the image from the source to the destination

        shutil.copyfile(src, dst)

    if fname in folder_2:

        # source path to image

        src = os.path.join('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2', fname)

        # destination path to image

        dst = os.path.join(test, label, fname)

        # copy the image from the source to the destination

        shutil.copyfile(src, dst)
# check how many train images we have in each folder

print("..............................")

print("Train folder")

print("..............................")

print(len(os.listdir('base/train/nv')))

print(len(os.listdir('base/train/mel')))

print(len(os.listdir('base/train/bkl')))

print(len(os.listdir('base/train/bcc')))

print(len(os.listdir('base/train/akiec')))

print(len(os.listdir('base/train/vasc')))

print(len(os.listdir('base/train/df')))

print("..............................")

# check how many train images we have in each folder

print("validation folder")

print("..............................")

print(len(os.listdir('base/valid/nv')))

print(len(os.listdir('base/valid/mel')))

print(len(os.listdir('base/valid/bkl')))

print(len(os.listdir('base/valid/bcc')))

print(len(os.listdir('base/valid/akiec')))

print(len(os.listdir('base/valid/vasc')))

print(len(os.listdir('base/valid/df')))

print("..............................")

# check how many train images we have in each folder

print("Test folder")

print("..............................")

print(len(os.listdir('base/test/nv')))

print(len(os.listdir('base/test/mel')))

print(len(os.listdir('base/test/bkl')))

print(len(os.listdir('base/test/bcc')))

print(len(os.listdir('base/test/akiec')))

print(len(os.listdir('base/test/vasc')))

print(len(os.listdir('base/test/df')))
%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        rotation_range=180,

        width_shift_range=0.1,

        height_shift_range=0.1,

        zoom_range=0.1,

        horizontal_flip=True,

        vertical_flip=True,

        fill_mode='nearest')

img_path = load_img('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2/ISIC_0029316.jpg',target_size=(224, 224))

 # this is a PIL image

x = img_to_array(img_path)  # Numpy array with shape (224, 224, 3)

x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 224, 224, 3)



# The .flow() command below generates batches of randomly transformed images

# It will loop indefinitely, so we need to `break` the loop at some point!

i = 0

for batch in datagen.flow(x, batch_size=1):

  plt.figure(i)

  imgplot = plt.imshow(array_to_img(batch[0]))

  i += 1

  if i % 5 == 0:

    break
# note that we are not augmenting class 'nv'

class_list = ['mel','bkl','bcc','akiec','vasc','df']



for item in class_list:

    

    # We are creating temporary directories here because we delete these directories later

    # create a base

    aug = 'aug'

    os.mkdir(aug)

    # create a dir within the base to store images of the same class

    img_dir = os.path.join(aug, 'img_dir')

    os.mkdir(img_dir)



    # Choose a class

    img_class = item



    # list all images in that directory

    img_list = os.listdir('base/train/' + img_class)



    # Copy images from the class train dir to the img_dir e.g. class 'mel'

    for fname in img_list:

            # source path to image

            src = os.path.join('base/train/' + img_class, fname)

            # destination path to image

            dst = os.path.join(img_dir, fname)

            # copy the image from the source to the destination

            shutil.copyfile(src, dst)





    # point to a dir containing the images and not to the images themselves

    path = aug

    save_path = 'base/train/' + img_class



    # Create a data generator

    datagen = ImageDataGenerator(

        rotation_range=180,

        width_shift_range=0.1,

        height_shift_range=0.1,

        zoom_range=0.1,

        horizontal_flip=True,

        vertical_flip=True,

        #brightness_range=(0.9,1.1),

        fill_mode='nearest')



    batch_size = 50



    aug_datagen = datagen.flow_from_directory(path,

                                           save_to_dir=save_path,

                                           save_format='jpg',

                                                    target_size=(224,224),

                                                    batch_size=batch_size)







    # Generate the augmented images and add them to the training folders

    

    ###########

    

    num_aug_images_wanted = 5000 # total number of images we want to have in each class

    

    ###########

    

    num_files = len(os.listdir(img_dir))

    num_batches = int(np.ceil((num_aug_images_wanted-num_files)/batch_size))



    # run the generator and create about 6000 augmented images

    for i in range(0,num_batches):



        imgs, labels = next(aug_datagen)

        

    # delete temporary directory with the raw image files

    shutil.rmtree('aug')
# check how many train images we have in each folder

print("..............................")

print("Train folder")

print("..............................")

print(len(os.listdir('base/train/nv')))

print(len(os.listdir('base/train/mel')))

print(len(os.listdir('base/train/bkl')))

print(len(os.listdir('base/train/bcc')))

print(len(os.listdir('base/train/akiec')))

print(len(os.listdir('base/train/vasc')))

print(len(os.listdir('base/train/df')))

print("..............................")

# check how many train images we have in each folder

print("validation folder")

print("..............................")

print(len(os.listdir('base/valid/nv')))

print(len(os.listdir('base/valid/mel')))

print(len(os.listdir('base/valid/bkl')))

print(len(os.listdir('base/valid/bcc')))

print(len(os.listdir('base/valid/akiec')))

print(len(os.listdir('base/valid/vasc')))

print(len(os.listdir('base/valid/df')))

print("..............................")

# check how many train images we have in each folder

print("Test folder")

print("..............................")

print(len(os.listdir('base/test/nv')))

print(len(os.listdir('base/test/mel')))

print(len(os.listdir('base/test/bkl')))

print(len(os.listdir('base/test/bcc')))

print(len(os.listdir('base/test/akiec')))

print(len(os.listdir('base/test/vasc')))

print(len(os.listdir('base/test/df')))
# Define transformations for data augmentation

tfms = get_transforms(do_flip=True,  

                      max_rotate=10,

                      max_zoom=1.1,

                      max_warp=0.2)



# Build dataset by applying transforms to the data from our directory

data = (ImageList.from_folder(base)

        .split_by_folder()          

        .label_from_folder()

        .add_test_folder('test')

        .transform(tfms, size=224)

        .databunch()

        .normalize(imagenet_stats))
wd=1e-2



mobilenet_split = lambda m: (m[0][0][10], m[1])

arch  = torchvision.models.mobilenet_v2

learn = cnn_learner(data, arch, cut=-1, split_on=mobilenet_split, wd=wd, metrics=[accuracy])
learn.lr_find();

learn.recorder.plot();
# Set our learning rate to the value where learning is fastest and loss 

# is still decreasing.



# This function uses our input lr as an anchor and sweeps through a range 

# in order to search out the best local minima.

learn.fit_one_cycle(5, max_lr=slice(3e-03), pct_start=0.9)
data.show_batch(rows=3, figsize=(4,4))
learn.recorder.plot_losses()
# Exctract predictions and losses to evaluate model

preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)
def top_k_spread(preds, y, spread):

  for i in range(spread):

    print(f"Top {i+1} accuracy: {top_k_accuracy(preds, y, i+1)}")
# Top-1 accuracy of 86% is quite near the best models from the open competition

top_k_spread(preds, y, 5)
interp.plot_confusion_matrix()
# probs from log preds

probs = np.exp(preds[:,1])

# Compute ROC curve

fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)



# Compute ROC area

roc_auc = auc(fpr, tpr)

print('ROC area is {0}'.format(roc_auc))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
learn.export()
learn.path
learn = load_learner(base,test=ImageList.from_folder('/kaggle/working/base/test'))
preds,y = learn.get_preds(ds_type=DatasetType.Test)

preds = np.argmax(preds, 1).tolist()
for i in range(0,7):

    print('The count of element:', i ,'is ', preds.count(i))
y_true=[]

for i in list(data.test_ds.items):

    if PurePath(i).parts[2]=="akiec":

        y_true.append(int(str(0)))

    elif PurePath(i).parts[2]=="bcc":

        y_true.append(int(str(1)))

    elif PurePath(i).parts[2]=="bkl":

        y_true.append(int(str(2))) 

    elif PurePath(i).parts[2]=="df":

        y_true.append(int(str(3)))  

    elif PurePath(i).parts[2]=="mel":

        y_true.append(int(str(4))) 

    elif PurePath(i).parts[2]=="nv":

        y_true.append(int(str(5)))

    else:

        y_true.append(int(str(6)))
target_names = ['akiec', 'bcc','bkl','df','mel','nv','vasc']

print(classification_report(y_true, preds, target_names=target_names))
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    import itertools

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
cnf_matrix = confusion_matrix(y_true, preds,labels=[0,1,2,3,4,5,6])

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['akiec', 'bcc','bkl','df','mel','nv','vasc'],

                      title='Confusion matrix, without normalization')
def plot_prediction(learn, index):

  data = learn.data.test_ds[index][0]

  pred = learn.predict(data)

  classes = learn.data.classes



  prediction = pd.DataFrame(to_np(pred[2]*100), columns=['Confidence'])

  prediction['Classes'] = classes

  prediction = prediction.sort_values(by='Confidence', ascending=False)



  fig = plt.figure(figsize=(12, 5))

  ax1 = fig.add_subplot(121)

  show_image(data, figsize=(5, 5), ax=ax1)

  ax2 = fig.add_subplot(122)

  sns.set_color_codes("pastel")

  sns.barplot(x='Confidence', y='Classes', data=prediction,

              label="Total", color="b")

  ax2.set_title(f'Actual: {PurePath(learn.data.test_ds.items[index]).parts[5]}')

plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))
plot_prediction(learn, np.random.choice(len(learn.data.test_ds)))