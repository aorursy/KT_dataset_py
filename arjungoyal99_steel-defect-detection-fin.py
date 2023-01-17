DATASET_DIR = '../input/severstal-steel-defect-detection/'

TEST_SIZE = 0.3

RANDOM_STATE = 123



NUM_TRAIN_SAMPLES = 20 # The number of train samples used for visualization

NUM_VAL_SAMPLES = 20 # The number of val samples used for visualization

COLORS = ['b', 'g', 'r', 'm'] # Color of each class
import pandas as pd

import os

import cv2

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from shutil import copyfile

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from plotly import subplots

import plotly.express as px

import plotly.figure_factory as ff

from plotly.graph_objs import *

from plotly.graph_objs.layout import Margin, YAxis, XAxis
df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
df.head()
legacy_df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])



for img_id, img_df in tqdm_notebook(df.groupby('ImageId')):

    for i in range(1, 5):

        avail_classes = list(img_df.ClassId)



        row = dict()

        row['ImageId_ClassId'] = img_id + '_' + str(i)



        if i in avail_classes:

            row['EncodedPixels'] = img_df.loc[img_df.ClassId == i].EncodedPixels.iloc[0]

        else:

            row['EncodedPixels'] = np.nan

        

        legacy_df = legacy_df.append(row, ignore_index=True)
legacy_df.head()
df = legacy_df
df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])

df['HavingDefection'] = df['EncodedPixels'].map(lambda x: 0 if x is np.nan else 1)



image_col = np.array(df['Image'])

image_files = image_col[::4]

all_labels = np.array(df['HavingDefection']).reshape(-1, 4)
num_img_class_1 = np.sum(all_labels[:, 0])

num_img_class_2 = np.sum(all_labels[:, 1])

num_img_class_3 = np.sum(all_labels[:, 2])

num_img_class_4 = np.sum(all_labels[:, 3])

print('Class 1: {} images'.format(num_img_class_1))

print('Class 2: {} images'.format(num_img_class_2))

print('Class 3: {} images'.format(num_img_class_3))

print('Class 4: {} images'.format(num_img_class_4))
def plot_figures(

    sizes,

    pie_title,

    start_angle,

    bar_title,

    bar_ylabel,

    labels=('Class 1', 'Class 2', 'Class 3', 'Class 4'),

    colors=None,

    explode=(0, 0, 0, 0.1),

):

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))



    y_pos = np.arange(len(labels))

    barlist = axes[0].bar(y_pos, sizes, align='center')

    axes[0].set_xticks(y_pos, labels)

    axes[0].set_ylabel(bar_ylabel)

    axes[0].set_title(bar_title)

    if colors is not None:

        for idx, item in enumerate(barlist):

            item.set_color(colors[idx])



    def autolabel(rects):

        """

        Attach a text label above each bar displaying its height

        """

        for rect in rects:

            height = rect.get_height()

            axes[0].text(

                rect.get_x() + rect.get_width()/2., height,

                '%d' % int(height),

                ha='center', va='bottom', fontweight='bold'

            )



    autolabel(barlist)

    

    pielist = axes[1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=start_angle, counterclock=False)

    axes[1].axis('equal')

    axes[1].set_title(pie_title)

    if colors is not None:

        for idx, item in enumerate(pielist[0]):

            item.set_color(colors[idx])



    plt.show()
print('[THE WHOLE DATASET]')



sum_each_class = np.sum(all_labels, axis=0)

plot_figures(

    sum_each_class,

    pie_title='The percentage of each class',

    start_angle=90,

    bar_title='The number of images for each class',

    bar_ylabel='Images',

    colors=COLORS,

    explode=(0, 0, 0, 0.1)

)



sum_each_sample = np.sum(all_labels, axis=1)

unique, counts = np.unique(sum_each_sample, return_counts=True)



plot_figures(

    counts,

    pie_title='The percentage of the number of classes appears in an image',

    start_angle=120,

    bar_title='The number of classes appears in an image',

    bar_ylabel='Images',

    labels=[' '.join((str(label), 'class(es)')) for label in unique],

    explode=np.zeros(len(unique))

)
X_train, X_val, y_train, y_val = train_test_split(image_files, all_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print('X_train:', X_train.shape)

print('y_train:', y_train.shape)

print('X_val:', X_val.shape)

print('y_val:', y_val.shape)
print('[TRAINING SET]')



sum_each_class = np.sum(y_train, axis=0)

plot_figures(

    sum_each_class,

    pie_title='The percentage of each class',

    start_angle=90,

    bar_title='The number of images for each class',

    bar_ylabel='Images',

    colors=COLORS,

    explode=(0, 0, 0, 0.1)

)





sum_each_sample = np.sum(y_train, axis=1)

unique, counts = np.unique(sum_each_sample, return_counts=True)



plot_figures(

    counts,

    pie_title='The percentage of the number of classes appears in an image',

    start_angle=120,

    bar_title='The number of classes appears in an image',

    bar_ylabel='Images',

    labels=[' '.join((str(label), 'class(es)')) for label in unique],

    explode=np.zeros(len(unique))

)
print('[VALIDATION SET]')



sum_each_class = np.sum(y_val, axis=0)

plot_figures(

    sum_each_class,

    pie_title='The percentage of each class',

    start_angle=90,

    bar_title='The number of images for each class',

    bar_ylabel='Images',

    colors=COLORS,

    explode=(0, 0, 0, 0.1)

)





sum_each_sample = np.sum(y_val, axis=1)

unique, counts = np.unique(sum_each_sample, return_counts=True)



plot_figures(

    counts,

    pie_title='The percentage of the number of classes appears in an image',

    start_angle=120,

    bar_title='The number of classes appears in an image',

    bar_ylabel='Images',

    labels=[' '.join((str(label), 'class(es)')) for label in unique],

    explode=np.zeros(len(unique))

)
def rle2mask(mask_rle, shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
def show_samples(samples):

    for sample in samples:

        fig, ax = plt.subplots(figsize=(15, 10))

        img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])

        img = cv2.imread(img_path)



        # Get annotations

        labels = df[df['ImageId_ClassId'].str.contains(sample[0])]['EncodedPixels']



        patches = []

        for idx, rle in enumerate(labels.values):

            if rle is not np.nan:

                mask = rle2mask(rle)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:

                    poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=1, edgecolor=COLORS[idx], fill=False)

                    patches.append(poly_patch)

        p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet)



        ax.imshow(img/255)

        ax.set_title('{} - ({})'.format(sample[0], ', '.join(sample[1].astype(np.str))))

        ax.add_collection(p)

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        plt.show()
train_pairs = np.array(list(zip(X_train, y_train)))

train_samples = train_pairs[np.random.choice(train_pairs.shape[0], NUM_TRAIN_SAMPLES, replace=False), :]



show_samples(train_samples)
val_pairs = np.array(list(zip(X_val, y_val)))

val_samples = val_pairs[np.random.choice(val_pairs.shape[0], NUM_VAL_SAMPLES, replace=False), :]



show_samples(val_samples)
df_train=legacy_df

del df_train['Image']

del df_train['HavingDefection']

train_df = df.fillna(-1)

train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis = 1)

grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)

def rle_to_mask(rle_string, height, width):  

    rows, cols = height, width

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]

        rle_pairs = np.array(rle_numbers).reshape(-1,2)

        img = np.zeros(rows*cols, dtype=np.uint8)

        for index, length in rle_pairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img
# calculate sum of the pixels for the mask per class id

train_df['mask_pixel_sum'] = train_df.apply(lambda x: rle_to_mask(x['EncodedPixels'], width=1600, height=256).sum(), axis=1)
class_ids = ['1','2','3','4']

mask_count_per_class = [train_df[(train_df['ClassId']==class_id)&(train_df['mask_pixel_sum']!=0)]['mask_pixel_sum'].count() for class_id in class_ids]

pixel_sum_per_class = [train_df[(train_df['ClassId']==class_id)&(train_df['mask_pixel_sum']!=0)]['mask_pixel_sum'].sum() for class_id in class_ids]
# Create subplots: use 'domain' type for Pie subplot

fig = subplots.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(Pie(labels=class_ids, values=mask_count_per_class, name="Mask Count"), 1, 1)

fig.add_trace(Pie(labels=class_ids, values=pixel_sum_per_class, name="Pixel Count"), 1, 2)

# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Steel Defect Mask & Pixel Count",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Mask', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Pixel', x=0.80, y=0.5, font_size=20, showarrow=False)])

fig.show()
# plot a histogram and boxplot combined of the mask pixel sum per class Id

fig = px.histogram(train_df[train_df['mask_pixel_sum']!=0][['ClassId','mask_pixel_sum']], 

                   x="mask_pixel_sum", y="ClassId", color="ClassId", marginal="box")



fig['layout'].update(title='Histogram and Boxplot of Sum of Mask Pixels Per Class')



fig.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook
df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")
df.head()
legacy_df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])



for img_id, img_df in tqdm_notebook(df.groupby('ImageId')):

    for i in range(1, 5):

        avail_classes = list(img_df.ClassId)



        row = dict()

        row['ImageId_ClassId'] = img_id + '_' + str(i)



        if i in avail_classes:

            row['EncodedPixels'] = img_df.loc[img_df.ClassId == i].EncodedPixels.iloc[0]

        else:

            row['EncodedPixels'] = np.nan

        

        legacy_df = legacy_df.append(row, ignore_index=True)
legacy_df.head()
data = legacy_df



data.info()
defects = data[pd.notna(data.EncodedPixels)]

defects.EncodedPixels = 1

defects.info()

print(defects)
print((data.EncodedPixels).isnull())

NoDefects = data[(data.EncodedPixels).isnull()]

NoDefects.EncodedPixels = 0

NoDefects.info()

print(NoDefects)
dataset= NoDefects.sample(defects.shape[0])

dataset = dataset.append(defects,ignore_index=True)

dataset = dataset.sample(frac=1, replace=True, random_state=1)

dataset
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

filename = str(dataset.sample(1).ImageId_ClassId.values)[2:]

filename = filename[:-4]

filename = "../input/severstal-steel-defect-detection/train_images/"+filename

print(filename)



img=mpimg.imread(filename)

imgplot = plt.imshow(img)

plt.show()
val = dataset[0:1000]

test = dataset[1000:2000]

train = dataset[2000:]

train.info
from skimage.feature import hog

import cv2



def my_extractHOG(filename):

    filename = str(filename)

    filename = filename[:-2]

    filename = "../input/severstal-steel-defect-detection/train_images/" + filename

    img = mpimg.imread(filename)

    img = cv2.resize(img, dsize=(600, 70), interpolation=cv2.INTER_CUBIC)

    print(str(i)+"/"+str(train.ImageId_ClassId.shape[0]))

    img = img / 256

    fd,hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)

    return fd,hog_image



ppc = 16

hog_images = []

hog_features = []



for i, filename in enumerate(train.ImageId_ClassId):

    fd,hog_image = my_extractHOG(filename)

    if i<6 : hog_images.append(hog_image) # save some of images for example purpose only

    hog_features.append(fd)

print (hog_features)
plt.imshow(hog_images[3])

print(hog_features[3].shape)
from sklearn.svm import SVC

clf = SVC(gamma='auto')

print(train.EncodedPixels.values.shape)

y = train.EncodedPixels.values

X = np.array(hog_features)

print(X.shape)

clf.fit(X,y)
from sklearn.metrics import roc_auc_score

y_scores = [] # init array

hog_features2 = []

for i, filename in enumerate(test.ImageId_ClassId):

    fd,hog_image = my_extractHOG(filename)

    out = clf.predict([np.array(fd)])

    y_scores.append(out)

    print(len(y_scores))

    hog_features2.append(fd)

y_true = test.EncodedPixels.values

y_scores = np.array(y_scores)

roc_auc_score(y_true, y_scores)
from catboost import CatBoostClassifier, Pool

cat_features = [0]

X = 10000 * X

X = X.astype(int)

print(X)

y.astype(int)

print(y)

Xval = 10000*np.array(hog_features2)

print(Xval)
train_dataset = Pool(data=X,

                     label=y,

                     cat_features=cat_features)



eval_dataset = Pool(data=Xval.astype(int),

                    label=y_true,

                    cat_features=cat_features)



# Initialize CatBoostClassifier

model = CatBoostClassifier(iterations=300,

                           learning_rate=1,

                           depth=2,

                           custom_metric='AUC')

# Fit model

model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True)

# Get predicted classes

preds_class = model.predict(eval_dataset)

print(preds_class)

# Get predicted probabilities for each class

preds_proba = model.predict_proba(eval_dataset)

# Get predicted RawFormulaVal

preds_raw = model.predict(eval_dataset,

                          prediction_type='RawFormulaVal')

print(model.get_best_score())

model.save_model('1layer_catboost')
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

from tqdm import tqdm_notebook

import cv2



import keras

from keras.layers.convolutional import Conv2DTranspose

from keras.layers.merge import concatenate

from keras.layers import UpSampling2D, Conv2D, Activation, Input, Dropout, MaxPooling2D

from keras import Model

from keras import backend as K

from keras.layers.core import Lambda
DATASET_DIR = '../input/severstal-steel-defect-detection'

df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
df.head()
legacy_df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])



for img_id, img_df in tqdm_notebook(df.groupby('ImageId')):

    for i in range(1, 5):

        avail_classes = list(img_df.ClassId)



        row = dict()

        row['ImageId_ClassId'] = img_id + '_' + str(i)



        if i in avail_classes:

            row['EncodedPixels'] = img_df.loc[img_df.ClassId == i].EncodedPixels.iloc[0]

        else:

            row['EncodedPixels'] = np.nan

        

        legacy_df = legacy_df.append(row, ignore_index=True)
legacy_df.head()
tr = legacy_df



tr.head()
df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)

print(len(df_train))

df_train.head()
def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )
img_size = 256
def keras_generator(batch_size):

    while True:

        x_batch = []

        y_batch = []

        

        for i in range(batch_size):            

            fn = df_train['ImageId_ClassId'].iloc[i].split('_')[0]

            img = cv2.imread( '../input/severstal-steel-defect-detection/train_images/'+fn )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            

            

            mask = rle2mask(df_train['EncodedPixels'].iloc[i], img.shape)

            

            img = cv2.resize(img, (img_size, img_size))

            mask = cv2.resize(mask, (img_size, img_size))

            

            x_batch += [img]

            y_batch += [mask]

                                    

        x_batch = np.array(x_batch)

        y_batch = np.array(y_batch)



        yield x_batch, np.expand_dims(y_batch, -1)
for x, y in keras_generator(4):

    break

    

print(x.shape, y.shape)
plt.imshow(x[3])
plt.imshow(np.squeeze(y[3]))
#Model



inputs = Input((256, 256, 3))

s = Lambda(lambda x: x / 255) (inputs)



c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)

c1 = Dropout(0.1) (c1)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)

c2 = Dropout(0.1) (c2)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)

c3 = Dropout(0.2) (c3)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)

c4 = Dropout(0.2) (c4)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)

c5 = Dropout(0.3) (c5)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)

c6 = Dropout(0.2) (c6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)

c7 = Dropout(0.2) (c7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)

c8 = Dropout(0.1) (c8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1], axis=3)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)

c9 = Dropout(0.1) (c9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)



outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)



model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy')
%%time

# Fit model

batch_size = 16

results = model.fit_generator(keras_generator(batch_size), 

                              steps_per_epoch=100,

                              epochs=1)
pred = model.predict(x)

plt.imshow(np.squeeze(pred[3]))
testfiles=os.listdir("../input/severstal-steel-defect-detection/test_images/")

len(testfiles)
%%time

test_img = []

for fn in tqdm_notebook(testfiles):

        img = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+fn )

        img = cv2.resize(img,(img_size,img_size))       

        test_img.append(img)
%%time

predict = model.predict(np.asarray(test_img))

print(len(predict))
def mask2rle(img):

    tmp = np.rot90( np.flipud( img ), k=3 )

    rle = []

    lastColor = 0;

    startpos = 0

    endpos = 0



    tmp = tmp.reshape(-1,1)   

    for i in range( len(tmp) ):

        if (lastColor==0) and tmp[i]>0:

            startpos = i

            lastColor = 1

        elif (lastColor==1)and(tmp[i]==0):

            endpos = i-1

            lastColor = 0

            rle.append( str(stnartpos)+' '+str(endpos-startpos+1) )

    return " ".join(rle)
%%time

pred_rle = []

for img in predict:      

    img = cv2.resize(img, (1600, 256))

    tmp = np.copy(img)

    tmp[tmp<np.mean(img)] = 0

    tmp[tmp>0] = 1

    pred_rle.append(mask2rle(tmp))
img_t = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ testfiles[4])

plt.imshow(img_t)
mask_t = rle2mask(pred_rle[4], img.shape)

plt.imshow(mask_t)
sub = pd.read_csv( '../input/submission/submission_tta.csv' )

sub.head()
%%time

for fn, rle in zip(testfiles, pred_rle):

    sub['EncodedPixels'][sub['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == fn] = rle
sub.head(8)
img_s = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ sub['ImageId_ClassId'][16].split('_')[0])

plt.imshow(img_s)
mask_s = rle2mask(sub['EncodedPixels'][16], (256, 1600))

plt.imshow(mask_s)
sub.to_csv('submission_tta.csv', index=False)
sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)