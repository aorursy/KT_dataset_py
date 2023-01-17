import numpy as np # linear algebra
import pandas as pd
import cv2, gc
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten
from keras.applications.vgg19 import VGG19
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
input_size = 128
epoch = 20
batch_size = 128
input_shape = (input_size, input_size, 3)
path = '/kaggle/input/planets-dataset/planet/planet/'
gc.collect()
train_classes = pd.read_csv(f'{path}train_classes.csv')
train_classes = shuffle(train_classes, random_state=0)
sample_submission = pd.read_csv(f'{path}sample_submission.csv')
trad_sample_df = sample_submission[sample_submission.image_name.str.contains('file_')].copy()
sample_submission = sample_submission[sample_submission.image_name.str.contains('test_')]
s = train_classes.tags.str.split(' ').explode()
lb = MultiLabelBinarizer()
encoded = lb.fit_transform(s.values[:, None])
one_hot_df = pd.DataFrame(encoded.tolist(), columns=np.ravel(lb.classes_), dtype='int') \
                .groupby(s.index) \
                .sum()
one_hot_df['image_name'] = train_classes["image_name"].apply(lambda fn: fn+".jpg")
cols = ['image_name'] + list(np.ravel(lb.classes_))
train_classes = one_hot_df[cols].copy()
del one_hot_df, s, encoded, lb
trad_sample_df['image_name'] = trad_sample_df["image_name"].apply(lambda fn: fn+".jpg")
sample_submission['image_name'] = sample_submission["image_name"].apply(lambda fn: fn+".jpg")
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                             zoom_range=0.5, rotation_range=90,
                             rescale=1./255.)
def VGG19_Amazon_Model(input_shape=input_shape):
    gc.collect()
    base_model = VGG19(include_top=False, weights='imagenet',
                       input_shape=input_shape)
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(17, activation='sigmoid'))
   
    return model


def return_model_name(k):
    return '/kaggle/working/model_'+str(k)+'.h5'


def train_model(df, k=5):
    gc.collect()
    model = VGG19_Amazon_Model()
    kf = KFold(n_splits=k, random_state=1, shuffle=True)
    fold = 1

    for train_index, val_index in kf.split(df.image_name):
        
        training_data = df.iloc[train_index]
        validation_data = df.iloc[val_index]
        
        train_generator=datagen.flow_from_dataframe(
                                            dataframe=training_data, directory=f'{path}/train-jpg/',
                                            x_col="image_name", y_col=cols[1:], batch_size=batch_size,
                                            seed=42, shuffle=True, class_mode="raw",
                                            target_size=(input_size, input_size))
        
        val_generator=datagen.flow_from_dataframe(
                                            dataframe=validation_data, directory=f'{path}/train-jpg/',
                                            x_col="image_name", y_col=cols[1:], batch_size=batch_size,
                                            seed=42, shuffle=True, class_mode="raw",
                                            target_size=(input_size, input_size))
        
        STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n//val_generator.batch_size
        
        opt = Adam(lr=0.0001)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        callback = [EarlyStopping(monitor='val_accuracy', patience=4, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                   cooldown=2, verbose=1),
                    ModelCheckpoint(return_model_name(fold), monitor='val_accuracy', 
                                    verbose=1, save_best_only=True, mode='max')]
        history = model.fit_generator(train_generator, 
                                      validation_data=val_generator,
                                      callbacks=callback, verbose=1, epochs=epoch) 
        
        #pred_val = model.predict_generator(val_generator, verbose=1)
        fold += 1
        
    return val_generator


def predict_model(test_gen, k=5, batch_size=batch_size):
    model = VGG19_Amazon_Model()
    full_test = []

    for nfold in range(1,k+1):
        model.load_weights(filepath=return_model_name(nfold))
        p_test = model.predict_generator(test_gen, verbose=1)
        full_test.append(p_test)
    
    result = np.array(full_test[0])
    for i in range(1, k):
        result += np.array(full_test[i])
    result = result / k
    
    result_bool = (result > 0.18)
    
    return result_bool.astype(int)


def generate_original_format(df):
    preds = []
    for i in tqdm(range(df.shape[0]), miniters=1000):
        a = df.iloc[[i]]
        pred_tag=[]
        for k in cols[1:]:
            if(a[k][i] == 1):
                pred_tag.append(k)
        preds.append(' '.join(pred_tag))

    df['tags'] = preds
    df['image_name'] = df['image_name'].apply(lambda x: x.split('.')[0])
    return df[['image_name', 'tags']]
val_generator = train_model(train_classes)
gc.collect()
pred_val = predict_model(val_generator, 5)
preds = np.argmax(pred_val, axis=1)
vals = np.argmax(val_generator.labels, axis=1)

print('F2 = {}'.format(fbeta_score(vals, preds, beta=2, average='micro')))

test_datagen=ImageDataGenerator(rescale=1./255.)

test1_generator=test_datagen.flow_from_dataframe(
                                            dataframe=sample_submission, directory=f'{path}/test-jpg/',
                                            x_col="image_name", y_col=None, batch_size=8,
                                            seed=42, shuffle=False, class_mode=None, 
                                            target_size=(input_size, input_size))

result1 = predict_model(test1_generator, 5)
result1 = pd.DataFrame(result1, columns=cols[1:])
result1["image_name"]=test1_generator.filenames
result1 = generate_original_format(result1.copy())
test2_generator=test_datagen.flow_from_dataframe(
                                            dataframe=trad_sample_df, 
                                            directory='../input/planets-dataset/test-jpg-additional/test-jpg-additional/',
                                            x_col="image_name", y_col=None, batch_size=8,
                                            seed=42, shuffle=False, class_mode=None, 
                                            target_size=(input_size, input_size))

result2 = predict_model(test2_generator, 5)
result2 = pd.DataFrame(result2, columns=cols[1:])
result2["image_name"]=test2_generator.filenames
result2 = generate_original_format(result2.copy())
results = result1.append(result2, ignore_index=True)
results.to_csv("submission.csv",index=False)
