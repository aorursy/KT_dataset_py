import pandas as pd

class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")

sample_submission = pd.read_csv("../input/bengaliai-cv19/sample_submission.csv")

test = pd.read_csv("../input/bengaliai-cv19/test.csv")

train = pd.read_csv("../input/bengaliai-cv19/train.csv")
print("train : ",train.shape)

print("test : ",test.shape)

print("class map : ",class_map.shape)
train.head()
test.head()
class_map.head()
train = train.drop(['grapheme'], axis=1, inplace=False)

train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
def resize(df, size=64):

    

    resized = {}

    resize_size=64

    

    for i in range(df.shape[0]):

        #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

        image=df.loc[df.index[i]].values.reshape(137,236)

        _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



        idx = 0 

        ls_xmin = []

        ls_ymin = []

        ls_xmax = []

        ls_ymax = []

        for cnt in contours:

            idx += 1

            x,y,w,h = cv2.boundingRect(cnt)

            ls_xmin.append(x)

            ls_ymin.append(y)

            ls_xmax.append(x + w)

            ls_ymax.append(y + h)

        xmin = min(ls_xmin)

        ymin = min(ls_ymin)

        xmax = max(ls_xmax)

        ymax = max(ls_ymax)



        roi = image[ymin:ymax,xmin:xmax]

        resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

        resized[df.index[i]] = resized_roi.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
def get_dummies(df):

    cols = []

    for col in df:

        cols.append(pd.get_dummies(df[col].astype(str)))

    return pd.concat(cols, axis=1)
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):



    def flow(self,

             x,

             y=None,

             batch_size=32,

             shuffle=True,

             sample_weight=None,

             seed=None,

             save_to_dir=None,

             save_prefix='',

             save_format='png',

             subset=None):



        targets = None

        target_lengths = {}

        ordered_outputs = []

        for output, target in y.items():

            if targets is None:

                targets = target

            else:

                targets = np.concatenate((targets, target), axis=1)

            target_lengths[output] = target.shape[1]

            ordered_outputs.append(output)





        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,

                                         shuffle=shuffle):

            target_dict = {}

            i = 0

            for output in ordered_outputs:

                target_length = target_lengths[output]

                target_dict[output] = flowy[:, i: i + target_length]

                i += target_length



            yield flowx, target_dict
IMG_SIZE=64

N_CHANNELS=1
inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))



model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.3)(model)



model = Flatten()(model)

model = Dense(1024, activation = "relu")(model)

model = Dropout(rate=0.3)(model)

dense = Dense(512, activation = "relu")(model)



head_root = Dense(168, activation = 'softmax')(dense)

head_vowel = Dense(11, activation = 'softmax')(dense)

head_consonant = Dense(7, activation = 'softmax')(dense)



model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_3_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_4_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_5_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)
X_train = train.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

X_train.shape
batch_size = 256

epochs = 30
histories=[]

for i in range(4):

    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train, on='image_id').drop(['image_id'], axis=1)

    

    print("merge done")

    

    X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

    X_train = resize(X_train)/255



    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)



    Y_train_root = pd.get_dummies(train_df['grapheme_root']).values

    Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values

    Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values



    print(f'Training images: {X_train.shape}')

    print(f'Training labels root: {Y_train_root.shape}')

    print(f'Training labels vowel: {Y_train_vowel.shape}')

    print(f'Training labels consonants: {Y_train_consonant.shape}')

    

    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)

    del train_df

    del X_train

    del Y_train_root, Y_train_vowel, Y_train_consonant

    

    datagen = MultiOutputDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=True,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=True,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.15, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)

    

    datagen.fit(x_train)

    

    

    history = model.fit_generator(datagen.flow(x_train, {'dense_3': y_train_root, 'dense_4': y_train_vowel, 'dense_5': y_train_consonant},batch_size=batch_size),

                                  epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]),

                                  steps_per_epoch=x_train.shape[0] // batch_size,

                                  callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant])

    

    del x_train

    del x_test

    del y_train_root

    del y_test_root

    del y_train_vowel

    del y_test_vowel

    del y_train_consonant

    del y_test_consonant

    gc.collect()
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img)/255

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    preds = model.predict(X_test)



    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

    del df_test_img

    del X_test

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()