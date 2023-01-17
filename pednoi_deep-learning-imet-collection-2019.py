import numpy as np

import pandas as pd



import random

from tqdm import tqdm

from pathlib import Path



import cv2

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from skmultilearn.model_selection import iterative_train_test_split



import itertools

from scipy.sparse import lil_matrix, coo_matrix

from collections import defaultdict, Counter



from sklearn.metrics import fbeta_score

from albumentations import (OneOf, Compose, HorizontalFlip, RandomCrop, 

                            RandomBrightness, RandomContrast, 

                            ShiftScaleRotate, IAAAdditiveGaussianNoise)



import keras

import keras.backend as K

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint

from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate, Lambda, Layer
DATA_ROOT = Path('../input/imet-2019-fgvc6/')



label_df = pd.read_csv(DATA_ROOT/'labels.csv')

print(f"Number of attributes = {len(label_df)}")



culture_df = label_df[label_df['attribute_name'].str.startswith('culture')]

print(f"Number of cultures = {len(culture_df)}")



tag_df = label_df[label_df['attribute_name'].str.startswith('tag')]

print(f"Number of tags = {len(tag_df)}")
label_df.sample(n=25).sort_values('attribute_id')
data_df = pd.read_csv(DATA_ROOT/'train.csv')

data_df['attribute_ids'] = data_df['attribute_ids'].str.split().map(lambda x_list: [int(x) for x in x_list])
sample_df = data_df.sample(n=10)



for _, row in sample_df.iterrows():

    img = cv2.imread(str(DATA_ROOT / 'train' / (row['id']+'.png')))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    labels = '\n'.join([label_df.loc[label]['attribute_name'] for label in row['attribute_ids']])



    _, axs = plt.subplots(1, 2, figsize=(12, 4))

        

    axs[0].imshow(img)

    axs[0].title.set_text(row['id'])

    

    axs[1].text(0, 0.5, labels, fontsize=12, ha='left', va='center')

    axs[1].axis('off')

    

    plt.show()
# train_df, valid_df = train_test_split(data_df, test_size=0.2, random_state=42)
# attributes = data_df['attribute_ids'].tolist()



# row = list(itertools.chain(*[[i]*len(attributes[i]) for i in range(len(data_df))]))

# col = list(itertools.chain(*attributes))



# y = coo_matrix(([1]*len(row), (row, col)), shape=(len(data_df), len(label_df))).tolil()



# train_df, _, valid_df, _ = iterative_train_test_split(data_df.values, y, test_size=0.2)

# train_df = pd.DataFrame(train_df, columns=['id', 'attribute_ids'])

# valid_df = pd.DataFrame(valid_df, columns=['id', 'attribute_ids'])
def make_folds(n_folds: int) -> pd.DataFrame:

    df = pd.read_csv(DATA_ROOT / 'train.csv')

    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split() for cls in classes)

    fold_cls_counts = defaultdict(int)

    folds = [-1] * len(df)

    for item in tqdm(df.sample(frac=1, random_state=42).itertuples(), total=len(df)):

        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])

        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]

        min_count = min([count for _, count in fold_counts])

        random.seed(item.Index)

        fold = random.choice([f for f, count in fold_counts if count == min_count])

        folds[item.Index] = fold

        for cls in item.attribute_ids.split():

            fold_cls_counts[fold, cls] += 1

    df['fold'] = folds

    return df



fold_df = make_folds(5)

fold_df['attribute_ids'] = fold_df['attribute_ids'].str.split().map(lambda x_list: [int(x) for x in x_list])



train_df = fold_df[fold_df['fold'] != 0].reset_index(drop=True)

valid_df = fold_df[fold_df['fold'] == 0].reset_index(drop=True)
train_attributes = list(itertools.chain(*train_df['attribute_ids'].tolist()))

print("Total train images: ", len(train_df))

print("Total train attributes: ", len(train_attributes))



plt.figure(figsize=(12, 3))

values, counts = np.unique(train_attributes, return_counts=True)

plt.bar(values, counts)

plt.show()



valid_attributes = list(itertools.chain(*valid_df['attribute_ids'].tolist()))

print("Total validation images: ", len(valid_df))

print("Total validation attributes: ", len(valid_attributes))



plt.figure(figsize=(12, 3))

values, counts = np.unique(valid_attributes, return_counts=True)

plt.bar(values, counts)

plt.show()
!pip install -U git+https://github.com/qubvel/efficientnet
from efficientnet.keras import EfficientNetB3, preprocess_input
EPOCHS = 8

BATCH_SIZE = 32



INPUT_SHAPE = (288, 288, 3)

NUM_CLASS = len(label_df)
def augment(p=1.0):

    return Compose([

        HorizontalFlip(p=0.5),

#         OneOf([

#             RandomBrightness(0.1, p=1.0),

#             RandomContrast(0.1, p=1.0),

#         ], p=0.3),

        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),

#         IAAAdditiveGaussianNoise(p=0.3),      

        RandomCrop(INPUT_SHAPE[0], INPUT_SHAPE[1])

    ], p=p)
class DataGenerator(keras.utils.Sequence):

    def __init__(self, df, batch_size, shuffle=True):

        self.df = df        

        self.indices = np.arange(len(self.df))

        

        self.batch_size = batch_size        

        self.shuffle = shuffle

        

        if self.shuffle:

            np.random.shuffle(self.indices)



        self.path = DATA_ROOT / 'train'



    def __len__(self):

        return int(np.ceil(len(self.df)/self.batch_size))



    def __getitem__(self, idx):

        batch_indices = self.indices[idx*self.batch_size: (idx+1)*self.batch_size]        

        

        batch_images = np.zeros((len(batch_indices), *INPUT_SHAPE))

        batch_labels = np.zeros((len(batch_indices), NUM_CLASS))

        

        for i in range(len(batch_indices)):

            row = self.df.iloc[batch_indices[i]]

            

            path = self.path / (row['id']+'.png')

            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        

            img = augment()(image=img)['image']

            batch_images[i] = preprocess_input(img)

            

            for label in row['attribute_ids']:

                batch_labels[i][label] = 1

                

        batch_images = np.array(batch_images, np.float32)

        return batch_images, batch_labels

    

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(self.indices)
train_generator = DataGenerator(train_df, BATCH_SIZE)

valid_generator = DataGenerator(valid_df, BATCH_SIZE, shuffle=False)
def create_model(input_shape):

    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)

    input_tensor = base_model.input

    

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dense(512, activation='relu',name='final_features')(x)

    x = Dropout(0.5)(x)

    output = Dense(NUM_CLASS, activation='sigmoid')(x)



    model = Model(input_tensor, output)

    return model



model = create_model(INPUT_SHAPE)

model.summary()
train_f2_hist = []

valid_f2_hist = []



def generate_labels(df):

    labels = np.zeros((len(df), NUM_CLASS))



    for i, row in df.iterrows():

        for label in row['attribute_ids']:

            labels[i][label] = 1

            

    return labels



def _make_mask(argsorted, top_n: int):

    mask = np.zeros_like(argsorted, dtype=np.uint8)

    col_indices = argsorted[:, -top_n:].reshape(-1)

    row_indices = [i // top_n for i in range(len(col_indices))]

    mask[row_indices, col_indices] = 1

    return mask



def binarize_prediction(predictions, threshold: float, min_labels=1, max_labels=10):

    assert predictions.shape[1] == NUM_CLASS

    argsorted = predictions.argsort(axis=1)

    max_mask = _make_mask(argsorted, max_labels)

    min_mask = _make_mask(argsorted, min_labels)

    prob_mask = predictions > threshold

    return (max_mask & prob_mask) | min_mask



class F2Evaluation(Callback):

    def __init__(self, interval=1):

        super(Callback, self).__init__()



        self.interval = interval        

        self.train_generator = DataGenerator(train_df, BATCH_SIZE, shuffle=False)

        

        self.train_y = generate_labels(train_df)

        self.valid_y = generate_labels(valid_df)        

        

    def predict(self, generator, y_true):

        predictions = self.model.predict_generator(generator, verbose=1)

        

        best_threshold = 0.0

        best_score = 0.0

        

        for threshold in np.arange(0.05, 0.55, 0.05):

            #y_pred = np.where(predictions > threshold, 1, 0)

            y_pred = binarize_prediction(predictions, threshold)

            

            f2_score = fbeta_score(y_true, y_pred, beta=2, average='samples')

            

            if f2_score > best_score:

                best_score = f2_score

                best_threshold = threshold

            

        return best_score, best_threshold        



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval != 0:

            return

        

        train_f2_score, train_threshold = self.predict(self.train_generator, self.train_y)

        valid_f2_score, valid_threshold = self.predict(valid_generator, self.valid_y)

        

        train_f2_hist.append(train_f2_score)             

        print("train f2 = %.4f (threshold = %.2f)" % (train_f2_score, train_threshold))        



        valid_f2_hist.append(valid_f2_score)             

        print("valid f2 = %.4f (threshold = %.2f)" % (valid_f2_score, valid_threshold))



        if valid_f2_score >= max(valid_f2_hist):

            print('save checkpoint: ', valid_f2_score)

            self.model.save_weights('model_bestf2.h5')



f2_metric = F2Evaluation(interval=1)
model.compile(loss='binary_crossentropy', optimizer=Adam(3e-4))



hist = model.fit_generator(train_generator, 

                           validation_data=valid_generator, 

                           epochs=EPOCHS, verbose=1,

                           callbacks=[f2_metric], 

                           use_multiprocessing=True, workers=2)
plt.plot(range(1, EPOCHS+1), hist.history['loss'], label='train_loss')

plt.plot(range(1, EPOCHS+1), hist.history['val_loss'], label='valid_loss')

plt.legend()

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()



plt.plot(range(1, EPOCHS+1), train_f2_hist, label='train_f2')

plt.plot(range(1, EPOCHS+1), valid_f2_hist, label='valid_f2')

plt.legend()

plt.ylabel('f2')

plt.xlabel('epoch')

plt.show()
model.load_weights('model_bestf2.h5')

predictions = np.zeros((len(valid_df), NUM_CLASS))



for _ in range(4):

    predictions += model.predict_generator(valid_generator, verbose=1)

predictions /= 4



y_true = generate_labels(valid_df)

y_pred = binarize_prediction(predictions, 0.1)



valid_f2_score = fbeta_score(y_true, y_pred, beta=2, average='samples')

print("valid tta f2 = %.4f (threshold = 0.10)" % valid_f2_score)
sample_df = valid_df.sample(n=10)



for i, row in sample_df.iterrows():

    img = cv2.imread(str(DATA_ROOT / 'train' / (row['id']+'.png')))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    plt.title(row['id'])

    plt.imshow(img)

    plt.show()



    true_labels = ', '.join([label_df.loc[label]['attribute_name'] 

                             for label in row['attribute_ids']])

    

    pred_labels = ', '.join([label_df.loc[label]['attribute_name'] 

                             for label in np.where(y_pred[i]==1)[0]])

    

    print("True labels = " + true_labels)

    print("Predicted labels = " + pred_labels)