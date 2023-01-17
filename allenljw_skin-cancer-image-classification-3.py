import glob

import numpy as np

import pandas as pd

import os

import PIL

import matplotlib.pyplot as plt

import math



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder



from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, concatenate

from keras.layers import Activation, Dropout, Flatten, Dense, Input

from tensorflow.keras import optimizers

from tensorflow.keras.utils import Sequence

from tensorflow.keras.callbacks import Callback
cat_dtype = {'sex':'category', 'anatom_site_general_challenge':'category'}



train_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv", dtype=cat_dtype)

test_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv", dtype=cat_dtype)

train_df.head()
# define the function to change the category column from string to int16

def chg_cat_int(df, cat_col):

    

    for col, col_dtype in cat_col.items():

        if col_dtype == 'category':

            df[col] = df[col].cat.codes.astype('int16')

            df[col] -= df[col].min()

            

    return df
# fill the NA value and change the category string to int

train_df['sex'] = train_df['sex'].cat.add_categories('unknown').fillna('unknown')

train_df['anatom_site_general_challenge'] = train_df['anatom_site_general_challenge'].cat.add_categories('unknown').fillna('unknown')

train_df['age_approx'] = train_df['age_approx'].fillna(train_df['age_approx'].mean())



train_df = chg_cat_int(train_df, cat_dtype)
test_df['sex'] = test_df['sex'].cat.add_categories('unknown').fillna('unknown')

test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].cat.add_categories('unknown').fillna('unknown')

test_df['age_approx'] = test_df['age_approx'].fillna(test_df['age_approx'].mean())



test_df = chg_cat_int(test_df, cat_dtype)
print('Training dataset: Number of data with target=1: ', len(train_df[train_df['target'] == 1]))

print('Training dataset: Number of data with target=0: ', len(train_df[train_df['target'] == 0]))
train_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'



train_df['file_path'] = train_path + train_df['image_name'] + '.jpg'

test_df['file_path'] = test_path + test_df['image_name'] + '.jpg'



cols = ['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant', 'file_path', 'target']

train_df = train_df[cols]



train_df.head()
enc = OneHotEncoder(categories=[[0, 1, 2], [0, 1, 2, 3, 4, 5, 6]], handle_unknown='ignore')

enc_train_df = pd.DataFrame(enc.fit_transform(train_df[['sex', 'anatom_site_general_challenge']]).toarray())

enc_test_df = pd.DataFrame(enc.fit_transform(test_df[['sex', 'anatom_site_general_challenge']]).toarray())



train_df = train_df.join(enc_train_df)

test_df = test_df.join(enc_test_df)



train_df.head()
# define the function to crop the center part of the image

def crop_center(img, cropx, cropy):

    y,x = img.shape[1:3]

    startx = x//2-(cropx//2)

    starty = y//2-(cropy//2)    

    return img[:,starty:starty+cropy,startx:startx+cropx:,]
image_1 = np.array([img_to_array(load_img(test_df['file_path'][0], target_size=(400,400)))])

image_1 /= 255.0

crop_image_1 = crop_center(image_1, 300, 300)



f, axarr = plt.subplots(1,2, figsize=(8,15)) 

axarr[0].imshow(image_1[0])

axarr[0].title.set_text("Original image")

axarr[1].imshow(crop_image_1[0])

axarr[1].title.set_text("Cropped image")
train_file = train_df['file_path'][train_df['target']==1].iloc[8]

images = np.array([img_to_array(load_img(train_file, target_size=(400,400)))])

images /= 255.0

new_img = np.rot90(images[0])



f, axarr = plt.subplots(1,2, figsize=(8,15)) 

axarr[0].imshow(images[0])

axarr[0].title.set_text("Original image")

axarr[1].imshow(new_img)

axarr[1].title.set_text("rotate 90 degree")
# Split the original training dataset as the training data and validation data



X_train, X_validate, y_train, y_validate = train_test_split(train_df, train_df['target'], test_size=0.2, random_state=9)
X_train_cancer_len = len(X_train[X_train['target'] == 1])

X_validate_cancer_len = len(X_validate[X_validate['target'] == 1])

print("Number of Melanoma in training set:", X_train_cancer_len)

print("Number of Melanoma in validation set:", X_validate_cancer_len)
class Train_Generator(Sequence):



    def __init__(self, train_df, num_samples=5000, batch_size=50, target_dim=(400, 400)):

        

        # separate the data in normal and cancer status

        normal_data = train_df[train_df['target'] == 0]

        cancer_data = train_df[train_df['target'] == 1]



        # shuffle the normal data and take num_samples records only to save training time

        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        normal_data = normal_data[:num_samples] 

        

        # append the cancer data and product X and y for model training data and shuffle the sequence

        normal_data = normal_data.append(cancer_data, ignore_index=True)

        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        x_image_set = normal_data['file_path']

        y_set = normal_data['target']

        cols = [i for i in range(10)]

        cols.append('age_approx')

        x_meta_set = normal_data[cols]

            

        self.df = train_df

        self.x_image, self.x_meta, self.y = x_image_set, x_meta_set, y_set

        self.samples = num_samples

        self.batch_size = batch_size

        self.target_size = target_dim

        #print('Init function, self.x=', self.x_image[0], ', lengh=', len(self.x_image))



    def __len__(self):

        return math.ceil(len(self.x_image) / self.batch_size)



    def __getitem__(self, idx):

        #print('index:', idx)

        batch_x = self.x_image[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = np.array(self.y[idx * self.batch_size:(idx + 1) * self.batch_size])

        image_x = np.array([img_to_array(load_img(img, target_size=self.target_size)) for img in batch_x])

        image_x /= 255.0

        

        crop_image_x = crop_center(image_x, self.target_size[0]-100, self.target_size[1]-100)

        meta_x = self.x_meta[idx * self.batch_size:(idx + 1) * self.batch_size]

        

        for i in range(len(batch_y)):

            if batch_y[i] == 1:

                new_img = np.rot90(crop_image_x[i])

                crop_image_x = np.concatenate((crop_image_x, [new_img]))

                batch_y = np.concatenate((batch_y, [1]))

                new_meta = meta_x.iloc[i]

                meta_x = meta_x.append(new_meta)      

        #print('target size:', batch_y)



        return [crop_image_x, meta_x], batch_y

    

    def on_epoch_end(self):

        normal_data = self.df[self.df['target'] == 0]

        cancer_data = self.df[self.df['target'] == 1]

        

        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        normal_data = normal_data[:self.samples] 

        

        normal_data = normal_data.append(cancer_data, ignore_index=True)

        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        x_image_set = normal_data['file_path']

        y_set = normal_data['target']

        cols = [i for i in range(10)]

        cols.append('age_approx')

        x_meta_set = normal_data[cols]

        

        self.x_image, self.x_meta, self.y = x_image_set, x_meta_set, y_set
# create the CNN model

def create_cnn(input_dim):



    model = Sequential()



    # Convolutional layer and max pooling layer

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_dim))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))



    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(2048, activation='relu', name='deep_1'))

    model.add(Dense(1024, activation='relu', name='deep_2'))

    model.add(Dense(256, activation='relu'))

    #model.add(Dropout(0.5))

    #model.add(Dense(1, activation='sigmoid'))





    return model
# create the MLP model

def create_mlp(input_dim):

    

    model = Sequential()

    model.add(Dense(16, input_dim=input_dim, activation="relu"))

    model.add(Dense(8, activation="relu"))

   

    return model
# Combine the CNN and MLP model

mlp = create_mlp(11)

cnn = create_cnn((300, 300, 3))



combinedInput = concatenate([mlp.output, cnn.output])



# Adding the classification layer

x = Dense(64, activation="relu")(combinedInput)

x = Dense(1, activation="sigmoid")(x)



model = Model(inputs=[cnn.input, mlp.input], outputs=x)



model.compile(loss='binary_crossentropy',

                  optimizer=optimizers.RMSprop(lr=1e-4),

                  metrics=['accuracy'])

model.summary()
# define the batch size and the image loading resolution

batch_size = 50

target_dim = (400, 400)



# train the model with the customized image generator. Enable multi-thread with workers=6 and run for 8 epochs

history = model.fit_generator(

    Train_Generator(X_train, 5000, batch_size, target_dim),

    steps_per_epoch = (5000+X_train_cancer_len) // batch_size,

    validation_data = Train_Generator(X_validate, 800, batch_size, target_dim),

    validation_steps = (800+X_validate_cancer_len) // batch_size,

    epochs = 6,

    workers=6,

    use_multiprocessing=True,

    verbose = 1)
# The image generator for the test dataset

class Test_Generator(Sequence):



    def __init__(self, x_image, x_meta, batch_size=50, target_dim=(200, 200)):

        self.x_files = x_image

        self.x_meta = x_meta

        self.batch_size = batch_size

        self.target_size = target_dim



    def __len__(self):

        return math.ceil(len(self.x_files) / self.batch_size)



    def __getitem__(self, idx):



        batch_x = self.x_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        image_x = np.array([img_to_array(load_img(img, target_size=self.target_size)) for img in batch_x])

        image_x /= 255.0

        

        crop_image_x = crop_center(image_x, self.target_size[0]-100, self.target_size[1]-100)     

        meta_x = self.x_meta[idx * self.batch_size:(idx + 1) * self.batch_size]



        return [crop_image_x, meta_x]
# Use the trained model to predict the result



test_files = test_df['file_path']

cols = [i for i in range(10)]

cols.append('age_approx')

meta_data = test_df[cols]



test_predict = model.predict(Test_Generator(test_files, meta_data, batch_size, target_dim),

                                                   steps=(len(test_files) // batch_size)+1,

                                                 workers=6, use_multiprocessing=True)
# Output the result for Kaggle submission



test_df['target'] = test_predict

test_df[['image_name', 'target']].to_csv('multitask-submission.csv', index=False)
