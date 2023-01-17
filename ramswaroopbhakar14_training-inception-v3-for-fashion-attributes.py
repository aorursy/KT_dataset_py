import os
import gc
import sys
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from tqdm import tqdm

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold
DATA_DIR = Path('/kaggle/input')
ROOT_DIR = Path('/kaggle/working')
IMAGE_DIR = Path('/kaggle/input/imaterialist-fashion-2019-FGVC6/train')

# For demonstration purpose, the classification ignores attributes (only categories),
# and the image size is set to 512, which is the same as the size of submission masks

NUM_CATS = 46
IMAGE_SIZE = 512
!git clone https://www.github.com/matterport/Mask_RCNN.git
os.chdir('Mask_RCNN')

!rm -rf .git # to prevent an error when the kernel is committed
!rm -rf images assets # to prevent displaying images at the bottom of a kernel
!pwd
sys.path.append(ROOT_DIR/'Mask_RCNN')
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
!ls -lh mask_rcnn_coco.h5

COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
#     STEPS_PER_EPOCH = 1000
#     VALIDATION_STEPS = 200
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 2
    
config = FashionConfig()
config.display()
with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]
attribute_names = [x['name'] for x in label_descriptions['attributes']]
print(len(label_names))
print(len(attribute_names))
label_names[:10]
attribute_names[:30]
segment_df = pd.read_csv(DATA_DIR/"train.csv")

print('le_segment_df',len(segment_df))
print(segment_df.head())
multilabel_percent = len(segment_df[segment_df['ClassId'].str.contains('_')])/len(segment_df)*100
print(f"Segments that have attributes: {multilabel_percent:.2f}%")
segment_df.columns
# segment_df['CategoryId'] = segment_df['Class']
segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]
segment_df['AttributeId'] = segment_df['ClassId'].str.split('_').str[1:]

print("Total segments: ", len(segment_df))

print('max_id:',max(list(map(lambda x:int(x),segment_df['CategoryId'] ))))
segment_df.head()
def show_img(IMG_FILE):
    I = cv2.imread("/kaggle/input/train/" + IMG_FILE, cv2.IMREAD_COLOR)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    I = cv2.resize(I, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    plt.imshow(I) 
    
def complete_make_mask(data,IMG_FILE):
    mask_list, cat_list = [], []
    df = data[data.ImageId == IMG_FILE].reset_index(drop = True)
    H = df.iloc[0,2]
    W = df.iloc[0,3]
    
    print("Correct Category :", sorted(set((list(df.CategoryId)))))
    # 1d mask 
    
    for line in df[['EncodedPixels','CategoryId']].iterrows():
        # 1d mask 
        mask = np.full(H*W,dtype='int',fill_value = -1)
        
        EncodedPixels = line[1][0]
        Category = line[1][1]
        
        pixel_loc = list(map(int,EncodedPixels.split(' ')[0::2]))
        iter_num =  list(map(int,EncodedPixels.split(' ')[1::2]))
        for p,i in zip(pixel_loc,iter_num):
            mask[p:(p+i)] = Category
        mask = mask.reshape(W,H).T
#         print(Category, mask.shape)
        mask_list+=[mask]
        cat_list+=[Category]
    
#     print("Output :",sorted(set(list(mask))))
#     print('mask:\n',set(list(mask)))
#     mask = mask.reshape(W,H).T
    #rle
#     return mask
    return cat_list, mask_list
img_list = os.listdir('/kaggle/input/train/')
for k in img_list[:3]:
    cat_list1, mask_list1 = complete_make_mask(segment_df, k)
    plt.figure(figsize=[15,15])
    plt.subplot(3,5,1)
    show_img(k)
    plt.title('Input Image')
    i=1
    for mask, cat in zip(mask_list1, cat_list1):
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        plt.subplot(3,5,i+1)
        i+=1
        plt.imshow(mask,cmap='jet')
        plt.title(label_names[int(cat)])
    plt.subplots_adjust(wspace=0.4, hspace=-0.65)
seg_att_df = segment_df[[len(x)>0 for x in segment_df['AttributeId']]].reset_index(drop=['index'])
seg_att_df.iloc[30:50]
image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
image_df = image_df.join(size_df, on='ImageId')

# image_df = image_df.iloc[:10]
print("Total images: ", len(image_df))
image_df.head()

def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img
class FashionDataset(utils.Dataset):

    def __init__(self, df):
        super().__init__(self)
        
        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("fashion", i+1, name)
        
        # Add images 
        for i, row in df.iterrows():
            self.add_image("fashion", 
                           image_id=row.name, 
                           path=str(DATA_DIR/'train'/row.name), 
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [label_names[int(x)] for x in info['labels']]
    
    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            
        return mask, np.array(labels)
dataset = FashionDataset(image_df)
dataset.prepare()

for i in range(8,10):
#     image_id = random.choice(dataset.image_ids)
    image_id = dataset.image_ids[i]
    print(dataset.image_reference(image_id))
    
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    print('mask_shape:',mask.shape)
    print('img_shape:',image.shape)
    print(class_ids)
    print(dataset.class_names)
    print(len(dataset.class_names))
#     plt.figure()
#     plt.imshow(image)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)
# This code partially supports k-fold training, 
# you can specify the fold to train and the total number of folds here
FOLD = 0
N_FOLDS = 5

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df) # ideally, this should be multilabel stratification

def get_fold():    
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]
        
train_df, valid_df = get_fold()

train_dataset = FashionDataset(train_df)
train_dataset.prepare()

valid_dataset = FashionDataset(valid_df)
valid_dataset.prepare()
train_segments = np.concatenate(train_df['CategoryId'].values).astype(int)
print("Total train images: ", len(train_df))
print("Total train segments: ", len(train_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(train_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, label_names, rotation='vertical')
plt.show()

valid_segments = np.concatenate(valid_df['CategoryId'].values).astype(int)
print("Total validation images: ", len(valid_df))
print("Total validation segments: ", len(valid_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(valid_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, label_names, rotation='vertical')
plt.show()
# Note that any hyperparameters here, such as LR, may still not be optimal
LR = 1e-4
# EPOCHS = [2, 6, 8]
EPOCHS = [1, 2, 3]

import warnings 
warnings.filterwarnings("ignore")
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5) # only horizontal flip here
])
%%time
model.train(train_dataset, valid_dataset,
            learning_rate=LR*2, # train heads with higher lr to speedup learning
            epochs=EPOCHS[0],
            layers='heads',
            augmentation=None)

history = model.keras_model.history.history
%%time
model.train(train_dataset, valid_dataset,
            learning_rate=LR,
            epochs=EPOCHS[1],
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]
%%time
model.train(train_dataset, valid_dataset,
            learning_rate=LR/5,
            epochs=EPOCHS[2],
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]
epochs = range(EPOCHS[-1])

plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()

plt.show()
best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])
glob_list = glob.glob(f'/kaggle/working/fashion*/mask_rcnn_fashion_{best_epoch:04d}.h5')
model_path = glob_list[0] if glob_list else ''
class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

assert model_path != '', "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
sample_df = pd.read_csv(DATA_DIR/"sample_submission.csv")
sample_df.head()
sample_df['EncodedPixels'][0]
# Convert data to run-length encoding
def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle
# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois
%%time
sub_list = []
missing_count = 0
for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    image = resize_image(str(DATA_DIR/'test'/row['ImageId']))
    result = model.detect([image])[0]
    if result['masks'].size > 0:
        masks, _ = refine_masks(result['masks'], result['rois'])
        for m in range(masks.shape[-1]):
            mask = masks[:, :, m].ravel(order='F')
            rle = to_rle(mask)
            label = result['class_ids'][m] - 1
            sub_list.append([row['ImageId'], ' '.join(list(map(str, rle))), label])
    else:
        # The system does not allow missing ids, this is an easy way to fill them 
        sub_list.append([row['ImageId'], '1 1', 23])
        missing_count += 1
submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
print("Total image results: ", submission_df['ImageId'].nunique())
print("Missing Images: ", missing_count)
submission_df.head()
submission_df.to_csv("submission.csv", index=False)
for i in range(9):
    image_id = sample_df.sample()['ImageId'].values[0]
    image_path = str(DATA_DIR/'test'/image_id)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = model.detect([resize_image(image_path)])
    r = result[0]
    
    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y_scale = img.shape[0]/IMAGE_SIZE
        x_scale = img.shape[1]/IMAGE_SIZE
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        
        masks, rois = refine_masks(masks, rois)
    else:
        masks, rois = r['masks'], r['rois']
        
    visualize.display_instances(img, rois, masks, r['class_ids'], 
                                ['bg']+label_names, r['scores'],
                                title=image_id, figsize=(12, 12))
def make_mask(data):
    df = data.reset_index(drop = True)
    H = df.iloc[0,2]
    W = df.iloc[0,3]
    
#     print("Correct Category :", int(df.CategoryId))
    # 1d mask 
    mask = np.full(H*W,dtype='int',fill_value = -1)
    
    for line in df[['EncodedPixels','CategoryId']].iterrows():
        # 1d mask 
#         mask = np.full(H*W,dtype='int',fill_value = -1)
        
        EncodedPixels = line[1][0]
        Category = line[1][1]
        
        pixel_loc = list(map(int,EncodedPixels.split(' ')[0::2]))
        iter_num =  list(map(int,EncodedPixels.split(' ')[1::2]))
        for p,i in zip(pixel_loc,iter_num):
            mask[p:(p+i)] = Category
        mask = mask.reshape(W,H).T
    return mask
IMAGE_SIZE2 = 299
img_id_list, apparel_img_list, cat_list, att_list = [], [], [],[]
apparel_id_list, att_id_list = [], []
# for i in range(seg_att_df.shape[0]):
for i in range(100):
#     if i%100==0:
    print(i)
    img_id_list+=[seg_att_df['ImageId'][i]]
    mask1 = make_mask(seg_att_df.iloc[i:i+1])
    mask1 = cv2.resize(mask1, (IMAGE_SIZE2, IMAGE_SIZE2), interpolation=cv2.INTER_NEAREST)  
    apparel_img_list+=[mask1]
    apparel_id_list+=[int(seg_att_df['CategoryId'][i])]
    cat_list+=[label_names[int(seg_att_df['CategoryId'][i])]]
    att_id_list+=[seg_att_df['AttributeId'][i]]
    att_list+=[[attribute_names[int(x)] for x in seg_att_df['AttributeId'][i]]]
image_att = pd.DataFrame({'ImageId':img_id_list,'ApparelImage':apparel_img_list,'ApparelId': apparel_id_list, 
                          'ApparelClass':cat_list,'AttributeId':att_id_list,'AttributeType':att_list})
image_att.head()
# for i in range(len(image_att)):
for i in range(4):
    plt.figure(figsize=[10,10])
    plt.imshow(image_att['ApparelImage'][i])
    plt.title(image_att['ApparelClass'][i]+'\n'+'; '.join(image_att['AttributeType'][i]))
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Dense,BatchNormalization,Dropout,Embedding,RepeatVector
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.models import Model
from pickle import dump, load
from keras.models import load_model
import numpy as np
inception = InceptionV3(weights='imagenet')

# pop the last softmax layer and freezing the remaining layers (re-structure the model)
inception.layers.pop()
#
for layer in inception.layers:
    layer.trainable = False

# building the final model
pre_trained_incept_v3 = Model(input = inception.input,output = inception.layers[-1].output)
pre_trained_incept_v3.summary()
msk = np.random.rand(len(image_att)) <= 0.8
train_att = image_att[msk].reset_index(drop=True)
val_att = image_att[~msk].reset_index(drop=True)
print(train_att.shape)
print(val_att.shape)
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Dense,BatchNormalization,Dropout,Embedding,RepeatVector
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

TARGET_SIZE = (299,299) # needed to convert the image as per pre-trained inceptionv3 requirements

img_feat_list = []
for i in range(len(train_att)):
    img = image_att['ApparelImage'][i]
    img = np.stack((img,)*3, axis=-1) # creating gray scale to 3-channel image
    # Converting image to array
    img_array = img_to_array(img)
    nimage = preprocess_input(img_array)
    # Adding one more dimesion
    nimage = np.expand_dims(nimage, axis=0)    
    fea_vec = pre_trained_incept_v3.predict(nimage)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    img_feat_list+=[fea_vec]
train_att['img_feat'] = img_feat_list
train_att['AttributeType'][0]
from keras.preprocessing.text import Tokenizer
# as we'll be building it as image captioning model, we need to add some fixed start and end attributes"
train_att['AttributeId'] = [[92]+x+[93] for x in train_att['AttributeId']]
train_att['AttributeType'] = [['att_start']+x+['att_end'] for x in train_att['AttributeType']]

total_train_att = np.concatenate(train_att['AttributeId'].values).astype(int)
print("Total Apparel images: ", len(train_att))
print("All atributes throughout apparel images: ", len(total_train_att))

attribute_names+=['att_start','att_end']

plt.figure(figsize=(12, 3))
values, counts = np.unique(total_train_att, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, [attribute_names[x] for x in values], rotation='vertical')
plt.show()

#currently, dropping apparel attributes with freq. < 10
final_val = values[counts>=2]
final_att = [attribute_names[x] for x in final_val]

train_att['Final_att'] = [[x for x in z if x in final_att] for z in train_att['AttributeType']]

max_no = max([len(x) for x in train_att['Final_att']])
print('Max. number of attributes:', max_no)
vocab_size = len(final_att) + 1
print('Feature vocab size:', vocab_size)

ixtoword = {}
wordtoix = {}

ix = 1
for w in final_att:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
# token = Tokenizer(num_words=vocab_size)
# token.fit_on_texts(final_att)
ixtoword = {}
wordtoix = {}
ix = 1
for w in final_att:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
wordtoix
train_att.head()
from keras.models import Model,Input
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Embedding,Dense,BatchNormalization,Dropout,LSTM,add
from keras.utils import plot_model

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np

def combined_model(MAX_LENGTH,VOCAB_SIZE):
    "model parameters"
#    NPIX = 299 # required image shape for pre-trained inceptionnv3 model 
#    TARGET_SIZE = (NPIX,NPIX,3)
    EMBEDDING_SIZE = 256 #
    
    # partial caption sequence model    
    inputs2 = Input(shape=(MAX_LENGTH,))
    se1 = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(EMBEDDING_SIZE)(se2) 
    
    
    # image feature extractor model
    inputs1 = Input(shape=(2048,)) # iceptionnv3
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(EMBEDDING_SIZE, activation='relu')(fe1)
    
    
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(EMBEDDING_SIZE, activation='relu')(decoder1) 
    #decoder2 = Dense(50, activation='relu')(decoder1) 
    outputs = Dense(VOCAB_SIZE, activation='softmax')(decoder2)
    
    
    # merge the two input models
    # image_feature + partial caption ===> output
    model = Model(inputs=[inputs1, inputs2], outputs=outputs) 
    
    # setting wight of embedded matrix that we saved earlier for words
#     with open("embedding_matrix.pkl","rb") as f:
#         embedding_matrix = load(f)   
#     model.layers[2].set_weights([embedding_matrix])
#     model.layers[2].trainable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def data_generator(train_att, MAX_LENGTH,VOCAB_SIZE, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    for i in range(len(train_att)):
        n+=1
        photo = train_att['img_feat'][i]
        att_list = list(train_att['Final_att'][i])
        
        seq = [wordtoix[x] for x in att_list]
        for i in range(1,len(seq)):
            in_seq , op_seq = seq[:i],seq[i]
            #converting input sequence to fix length
            in_seq = pad_sequences([in_seq],maxlen=MAX_LENGTH,padding="post")[0]
            # converting op_seq to vocabulary size
#                    print(op_seq)
            op_seq = to_categorical([op_seq],num_classes=VOCAB_SIZE)[0]
#                    try:
#                        op_seq = to_categorical([op_seq],num_classes=VOCAB_SIZE)[0]
#                    except:
#                        op_seq = np.array([0]*VOCAB_SIZE)
            X1.append(photo)
            X2.append(in_seq)
            y.append(op_seq)
        # yield the batch data
        if n==num_photos_per_batch:
            yield [[np.array(X1), np.array(X2)], np.array(y)]
            X1, X2, y = list(), list(), list()
            n=0

                
# image feature extracted file
train_image_extracted = train_att['img_feat']

#"load train attributes
train_descriptions = train_att['Final_att']


model = combined_model(max_length, vocab_size) #

epochs = 10


len(train_descriptions)


for i in range(epochs):
    batch_size = number_pics_per_batch = 5
    steps = len(train_descriptions)//number_pics_per_batch
    generator = data_generator(train_att,max_length, vocab_size,number_pics_per_batch)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

att_prediction_model = model
att_prediction_model = model

# extract features from each photo in the directory
def extract_features(img):
    img = np.stack((img,)*3, axis=-1) # creating gray scale to 3-channel image
    # Converting image to array
    img_array = img_to_array(img)
    nimage = preprocess_input(img_array)
    # Adding one more dimesion
    nimage = np.expand_dims(nimage, axis=0)    
    fea_vec = pre_trained_incept_v3.predict(nimage)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


# generate a description for an image
def generate_desc(model, photo, max_length):
    # seed the generation process
    sequence = ['att_start']
    photo = photo.reshape(1,2048)

    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        seq = [wordtoix[x] for x in sequence]
        # pad input
        seq1 = pad_sequences([seq], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,seq1], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = ixtoword[yhat]
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        sequence+=[word]
        # stop if we predict the end of the sequence
        if word == 'att_end':
            break
    return sequence
        
 

"prediction on new images"
val_att = train_att.copy()



for i in range(3):
    print(i)
    img = val_att['ApparelImage'][i]
    plt.figure()
    plt.imshow(img)
    photo = extract_features(img)
    description = generate_desc(att_prediction_model, photo, max_length)
    plt.title('pred. apparel attributes:\n'+'; '.join([x for x in description if x not in ['att_start', 'att_end']]))
