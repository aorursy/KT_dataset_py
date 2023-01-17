# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd  



# data from https://github.com/lixin4ever/Conference-Acceptance-Rate

Accepted = [540, 602, 643, 783, 979, 1300]  

Submitted = [1807, 2123, 2145, 2620, 3303, 5160]  

Year= [2014,2015,2016,2017,2018,2019]

list_of_tuples = list(zip(Year,Accepted, Submitted))  

df = pd.DataFrame(list_of_tuples, columns = ['Year', 'Accepted', 'Submitted'])   

df
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(20,10))



df.plot.bar(x='Year',stacked=False)

ax = plt.gca()

ax.grid(which='major', axis='y', linestyle='--')

plt.xticks(rotation=0)

plt.ylabel('# of papers')

plt.savefig('submissions.png')
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

train.AdoptionSpeed.value_counts(normalize=True)
from sklearn.metrics import cohen_kappa_score

from itertools import repeat

import random



distribution = list(reversed(list(train.AdoptionSpeed.value_counts(normalize=True))))



y_true = train['AdoptionSpeed'].tolist()

y_pred = list(np.random.choice([0,1,2,3,4], p=distribution, size=(len(y_true))))

cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_pred = list(np.random.choice([0,1,2,3,4], p=[0.2, 0.2, 0.2, 0.2, 0.2], size=(len(y_true))))

cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_pred = list(np.random.choice([0,1,2,3,4], p=[0.0, 0.0, 0.0, 0.01, 0.99], size=(len(y_true))))

cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_pred = list(np.random.choice([0,1,2,3,4], p=[0.99, 0.0, 0.0, 0.0, 0.01], size=(len(y_true))))

cohen_kappa_score(y_true, y_pred, weights='quadratic')
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')



train['has_photo'] = train['PhotoAmt'].apply(lambda x: True if x > 0 else False)

test['has_photo'] = test['PhotoAmt'].apply(lambda x: True if x > 0 else False)
train.PhotoAmt.describe()
test.PhotoAmt.describe()
print("photos in train set: %d" % train.has_photo.value_counts()[1])

print("photos in test set: %d" % test.has_photo.value_counts()[1])

print("Missing photos in train set: %d" % train.has_photo.value_counts()[0])

print("Missing photos in test set: %d" % test.has_photo.value_counts()[0])

print("Percent missing in train set: %.2f" % (train.has_photo.value_counts()[0]/train.shape[0]*100))

print("Percent missing in test set: %.2f" % (test.has_photo.value_counts()[0]/test.shape[0]*100))
#when no photo is available, which AdoptionSpeed is likey?

train[train.has_photo == False].AdoptionSpeed.value_counts(normalize=True)
!pip install imagesize



import imagesize



image_sizes=pd.DataFrame()

for file in os.listdir('../input/petfinder-adoption-prediction/train_images'):

    width, height = imagesize.get('../input/petfinder-adoption-prediction/train_images/'+file)

    image_sizes = image_sizes.append({'width' : width , 'height' : height} , ignore_index=True)

    

image_sizes.describe()
image_sizes.shape
test_image_sizes=pd.DataFrame()

for file in os.listdir('../input/petfinder-adoption-prediction/test_images'):

    width, height = imagesize.get('../input/petfinder-adoption-prediction/test_images/'+file)

    test_image_sizes = test_image_sizes.append({'width' : width , 'height' : height} , ignore_index=True)

    

test_image_sizes.describe()
image_sizes.describe()
image_sizes.sort_values(by='height').head(1)
image_sizes.sort_values(by='width').head(1)
image_sizes.sort_values(by='height', ascending=False).head(1)
image_sizes.sort_values(by='width', ascending=False).head(1)
test_image_sizes.sort_values(by='height').head(1)
test_image_sizes.sort_values(by='width', ascending=False).head(1)
plt.figure(figsize=(20,10))



ax = image_sizes.hist(bins=25, grid=False, figsize=(12,8), zorder=2, rwidth=0.9)



ax = ax[0]

for x in ax:



    # Despine

    x.spines['right'].set_visible(False)

    x.spines['top'].set_visible(False)

    x.spines['left'].set_visible(False)



    # Switch off ticks

    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")



    # Draw horizontal axis lines

    vals = x.get_yticks()

    for tick in vals:

        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)



    # Remove title

    x.set_title("")



    # Set x-axis label

    x.set_xlabel("Pixel", labelpad=20)



    # Set y-axis label

    x.set_ylabel("Anzahl Bilder", labelpad=20)

    

plt.savefig('image_sizes.png')
!pip install talos
param = {'lr': (0.1, 10, 10),

         'batch_size': [32, 64, 128, 256, 512],

         'epochs': [10,20,50],

         'validation_split' : [0.1, 0.2, 0.5],

         'dropout': [0.1, 0.25, 0.5, 0.8],

         'optimizer' : [Adam, Nadam],

         'loss': ['categorical_crossentropy'],

         'last_activation' : ['softmax'],

         'weight_regulizer': [None]}
from talos import Reporting

r = Reporting('../input/resnet50-talos-score/resnet50_talos_score.csv')



# returns the results dataframe

r.data.sort_values(['val_acc'], ascending =False)
r.best_params()[0]
# get correlation for hyperparameters against a metric

r.correlate('val_loss')
from keras.models import Sequential

from keras.layers import Dense



# Modell initialisieren

model = Sequential()



# Layer hinzufügen (input, hidden, output layer)

model.add(Dense(units=64, activation='relu', input_dim=100))

model.add(Dropout())

model.add(Dense(units=10, activation='softmax'))



# Modell kompilieren mit Loss-Funktion, Optimizer und Metrik

model.compile(loss='categorical_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])



# Modell trainieren 

model.fit(x_train, y_train, epochs=5, batch_size=32)
# class weight für manueller 20% split

from sklearn.utils import class_weight



result = []

for x in range(781):

    result.append(0)

for x in range(6768):

    result.append(1)    

for x in range(9949):

    result.append(2)        

for x in range(9467):

    result.append(3)     

for x in range(7960):

    result.append(4)

result =np.asarray(result)   



class_weight.compute_class_weight('balanced',np.unique(result),result)
class_weight = {0: 8.94366197,

                1: 1.03206265,

                2: 0.70208061,

                3: 0.73782613,

                4: 0.87751256}
# berechnen der clas weights, 



# print(train_df[train_df.AdoptionSpeed == 0].PhotoAmt.sum())

# print(train_df[train_df.AdoptionSpeed == 1].PhotoAmt.sum())

# print(train_df[train_df.AdoptionSpeed == 2].PhotoAmt.sum())

# print(train_df[train_df.AdoptionSpeed == 3].PhotoAmt.sum())

# print(train_df[train_df.AdoptionSpeed == 4].PhotoAmt.sum())
# class weight für 60%, 20%, 20% split, sollte ähnlich sein.

# pro klasse

#4    2501

#2    2422

#3    1995

#1    1834

#0     243



# array([7.40329218, 0.98091603, 0.74277457, 0.90175439, 0.71931228])



# pro Bilder pro Klasse

# 806.0

# 6986.0

# 9788.0

# 9509.0

# 8399.0



# array([8.80595533, 1.01597481, 0.72513282, 0.74640867, 0.84505298])



from sklearn.utils import class_weight



result = []

for x in range(806):

    result.append(0)

for x in range(6986):

    result.append(1)    

for x in range(9788):

    result.append(2)        

for x in range(9509):

    result.append(3)     

for x in range(8399):

    result.append(4)

result =np.asarray(result)   



class_weight.compute_class_weight('balanced',np.unique(result),result)
class_weight = {0: 8.80595533,

                1: 1.01597481,

                2: 0.72513282,

                3: 0.74640867,

                4: 0.84505298}
IMAGE_FOLDER_PATH="../input/petfinder-adoption-prediction/train_images"

FILE_NAMES=os.listdir(IMAGE_FOLDER_PATH)

WIDTH=331

HEIGHT=331
targets=list()

full_paths=list()

for file_name in FILE_NAMES:

    target=file_name.split(".")[0]

    full_path=os.path.join(IMAGE_FOLDER_PATH, file_name)

    full_paths.append(full_path)

    petID = target[: target.find("-")]

    target = train.loc[train.PetID == petID].AdoptionSpeed.item()

    targets.append(str(target))



dataset=pd.DataFrame()

dataset['image_path']=full_paths

dataset['target']=targets
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator



def get_side(img, side_type, side_size=5):

    height, width, channel=img.shape

    if side_type=="horizontal":

        return np.ones((height,side_size,  channel), dtype=np.float32)*255

        

    return np.ones((side_size, width,  channel), dtype=np.float32)*255



def show_gallery(show="all"):

    n=50

    counter=0

    images=list()

    vertical_images=[]

    rng_state = np.random.get_state()

    np.random.shuffle(full_paths)

    np.random.set_state(rng_state)

    np.random.shuffle(targets)

    for path, target in zip(full_paths, targets):

        if target!=show and show!="all":

            continue

        counter=counter+1

        if counter%50==0:

            break

        #Image loading from disk as JpegImageFile file format

        img=load_img(path, target_size=(WIDTH,HEIGHT))

        #Converting JpegImageFile to numpy array

        img=img_to_array(img)

        

        hside=get_side(img, side_type="horizontal")

        images.append(img)

        images.append(hside)



        if counter%10==0:

            himage=np.hstack((images))

            vside=get_side(himage, side_type="vertical")

            vertical_images.append(himage)

            vertical_images.append(vside)

            

            images=list()



    gallery=np.vstack((vertical_images)) 

    plt.figure(figsize=(20,20))

    plt.xticks([])

    plt.yticks([])

    title={"all":"all AdoptionSpeed",

          "0": "AdoptionSpeed 0",

          "1": "AdoptionSpeed 1",

          "2": "AdoptionSpeed 2",

          "3": "AdoptionSpeed 3",

          "4": "Adoptionspeed 4"}

    plt.title("50 samples of {} of the dataset".format(title[show]))

    plt.imshow(gallery.astype(np.uint8))
show_gallery(show="all")
show_gallery(show="0")

show_gallery(show="1")

show_gallery(show="2")

show_gallery(show="3")

show_gallery(show="4")

show_gallery(show="all")
from talos import Reporting

r = Reporting('../input/nasnet-talos-score/nasnet_talos_score.csv')



# returns the results dataframe

r.data.sort_values(['val_acc'], ascending =False)
%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



columns = 4

rows = 1

fig=plt.figure(figsize=(20, 20))

for i in range(1, columns*rows +1):

    img=mpimg.imread('../working/images/' + random.choice(os.listdir("../working/images")))

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

plt.show()
#!pip install pandas-profiling



import pandas_profiling



train.profile_report()