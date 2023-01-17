import glob

import numpy as np

import pandas as pd

import os

import PIL

import matplotlib.pyplot as plt

import math



from sklearn.model_selection import train_test_split



from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D

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
def crop_center(img,cropx,cropy):

    y,x = img.shape[1:3]

    startx = x//2-(cropx//2)

    starty = y//2-(cropy//2)    

    return img[:,starty:starty+cropy,startx:startx+cropx:,]
X_train, X_validate, y_train, y_validate = train_test_split(train_df, train_df['target'], test_size=0.2, random_state=9)

X_train.head()
X_train_cancer_len = len(X_train[X_train['target'] == 1])

X_validate_cancer_len = len(X_validate[X_validate['target'] == 1])

print(X_train_cancer_len)

print(X_validate_cancer_len)
# separate the data in normal and cancer status

#normal_data = train_df[train_df['target'] == 0]

#cancer_data = train_df[train_df['target'] == 1]



# shuffle the normal data and take 10000 records only to save training time

#normal_data = normal_data.sample(frac=1).reset_index(drop=True)

#normal_data = normal_data[:500] 



# append the cancer data and product X and y for model training data

#normal_data = normal_data.append(cancer_data, ignore_index=True)

#normal_data = normal_data.sample(frac=1).reset_index(drop=True)

#X = normal_data.iloc[:, 0:-1]

#y = normal_data['target']
class Train_Generator(Sequence):



    def __init__(self, train_df, num_samples=5000, batch_size=50, target_dim=(400, 400)):

        

        # separate the data in normal and cancer status

        normal_data = train_df[train_df['target'] == 0]

        cancer_data = train_df[train_df['target'] == 1]



        # shuffle the normal data and take num_samples records only to save training time

        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        normal_data = normal_data[:num_samples] 

        

        # append the cancer data and product X and y for model training data

        normal_data = normal_data.append(cancer_data, ignore_index=True)

        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        x_set = normal_data['file_path']

        y_set = normal_data['target']

        

        self.df = train_df

        self.x, self.y = x_set, y_set

        self.samples = num_samples

        self.batch_size = batch_size

        self.target_size = target_dim

        print('Init function, self.x=', self.x[0], ', lengh=', len(self.x))



    def __len__(self):

        return math.ceil(len(self.x) / self.batch_size)



    def __getitem__(self, idx):

        #print('index:', idx)

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        image_x = np.array([img_to_array(load_img(img, target_size=self.target_size)) for img in batch_x])

        image_x /= 255.0

        

        crop_image_x = crop_center(image_x, self.target_size[0]-100, self.target_size[1]-100)



        return crop_image_x, batch_y

    

    def on_epoch_end(self):

        normal_data = self.df[self.df['target'] == 0]

        cancer_data = self.df[self.df['target'] == 1]

        

        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        normal_data = normal_data[:self.samples] 

        

        normal_data = normal_data.append(cancer_data, ignore_index=True)

        normal_data = normal_data.sample(frac=1).reset_index(drop=True)

        x_set = normal_data['file_path']

        y_set = normal_data['target']

        

        self.x, self.y = x_set, y_set

        print('on_epoch_end, self.x=', self.x[0], ', lengh=', len(self.x))
#X_train_files = X_train['file_path']

#X_validate_files = X_validate['file_path']



#target_dim = (300, 300)

#train_images = [img_to_array(load_img(img, target_size=target_dim)) for img in X_train_files]

#train_images = np.array(train_images)



#validate_images = [img_to_array(load_img(img, target_size=target_dim)) for img in X_validate_files]

#validate_images = np.array(validate_images)



#train_images.shape
model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(2048, activation='relu', name='deep_1'))

model.add(Dense(1024, activation='relu', name='deep_2'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



#model.compile(loss='binary_crossentropy',

#              optimizer=optimizers.RMSprop(lr=1e-4),

#              metrics=['accuracy'])

model.compile(optimizer='adam', 

                  loss='binary_crossentropy',

                  metrics=['accuracy'])



model.summary()
batch_size = 50

target_dim = (400, 400)

X_train_files = X_train['file_path']

X_validate_files = X_validate['file_path']



#class printbatch(Callback):

#    def on_batch_end(self, epoch, logs={}):

#        print(logs)



#pb = printbatch()



history = model.fit_generator(

    Train_Generator(X_train, 12000, batch_size, target_dim),

    steps_per_epoch = (12000+X_train_cancer_len) // batch_size,

    validation_data = Train_Generator(X_validate, 1500, batch_size, target_dim),

    validation_steps = (1500+X_validate_cancer_len) // batch_size,

    #callbacks=[pb],

    epochs = 2,

    workers=6,

    use_multiprocessing=True,

    verbose = 1)
deep_feature_model = Model(inputs=model.input, outputs=model.get_layer('deep_2').output)

deep_feature_model.summary()
class Test_Generator(Sequence):



    def __init__(self, x_set, batch_size=50, target_dim=(200, 200)):

        self.x = x_set

        self.batch_size = batch_size

        self.target_size = target_dim



    def __len__(self):

        return math.ceil(len(self.x) / self.batch_size)



    def __getitem__(self, idx):



        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        image_x = np.array([img_to_array(load_img(img, target_size=self.target_size)) for img in batch_x])

        image_x /= 255.0

        

        crop_image_x = crop_center(image_x, self.target_size[0]-100, self.target_size[1]-100)



        return crop_image_x
# separate the data in normal and cancer status

normal_data = train_df[train_df['target'] == 0]

cancer_data = train_df[train_df['target'] == 1]



# shuffle the normal data and take 20000 records only to save training time

normal_data = normal_data.sample(frac=1).reset_index(drop=True)

normal_data = normal_data[:20000] 



# append the cancer data and product X and y for model training data

normal_data = normal_data.append(cancer_data, ignore_index=True)

normal_data = normal_data.sample(frac=1).reset_index(drop=True)

X_files = normal_data['file_path']

y_lables = normal_data['target']
X_train_files, X_validate_files, y_train, y_validate = train_test_split(X_files, y_lables, test_size=0.2, random_state=9)

X_train_files.head()
X_train_feature_predict = deep_feature_model.predict(Test_Generator(X_train_files, batch_size, target_dim), 

                                                        steps=(len(X_train_files) // batch_size)+1,

                                                        workers=6, use_multiprocessing=True)

print(len(X_train_files))

print(X_train_feature_predict.shape)
X_validate_feature_predict = deep_feature_model.predict(Test_Generator(X_validate_files, batch_size, target_dim), 

                                                        steps=(len(X_validate_files) // batch_size)+1,

                                                        workers=6, use_multiprocessing=True)

X_validate_feature_predict.shape
test_files = test_df['file_path']
X_test_feature_predict = deep_feature_model.predict(Test_Generator(test_files, batch_size, target_dim),

                                                   steps=(len(test_files) // batch_size)+1,

                                                        workers=6, use_multiprocessing=True)

X_test_feature_predict.shape
cols = [f"col_{num}" for num in range(1024)]

train_feature_df = pd.DataFrame(X_train_feature_predict, columns=cols)

validate_feature_df = pd.DataFrame(X_validate_feature_predict, columns=cols)

test_feature_df = pd.DataFrame(X_test_feature_predict, columns=cols)



train_img_name = [file.split('/')[-1].split('.')[0].strip() for file in X_train_files]

validate_img_name = [file.split('/')[-1].split('.')[0].strip() for file in X_validate_files]

test_img_name = [file.split('/')[-1].split('.')[0].strip() for file in test_files]



train_feature_df['image_name'] = train_img_name

validate_feature_df['image_name'] = validate_img_name

test_feature_df['image_name'] = test_img_name



train_feature_df.head()
train_feature_df = train_feature_df.merge(train_df, on='image_name')

validate_feature_df = validate_feature_df.merge(train_df, on='image_name')

test_feature_df = test_feature_df.merge(test_df, on='image_name')





print('length of train features dataframe: ', len(train_feature_df))

train_feature_df.head()
cols = [f"col_{num}" for num in range(1024)]

cat_cols = ['sex', 'anatom_site_general_challenge']

age_col = ['age_approx']



columns = cat_cols + age_col + cols



X_feature_train = train_feature_df[columns]

y_feature_train = train_feature_df['target']



X_feature_validate = validate_feature_df[columns]

y_feature_validate = validate_feature_df['target']



X_feature_test = test_feature_df[columns]
import lightgbm as lgb



# use the lightGBM model, the category features is not required to performe the OneHotEnconder as the lightGBM have the parameter to indicate the category features. 

def create_GBMmodel(X_train, y_train, X_validate, y_validate, cat_features):

            

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

    validate_data = lgb.Dataset(X_validate, label=y_validate, categorical_feature=cat_features)

    

    params = {

        "objective" : "xentropy",

        "metric" :"binary_logloss",

        "force_row_wise" : True,

        'verbosity': 1,

    }

    

    num_round = 200

    m_lgb = lgb.train(params, train_data, num_round, valid_sets = [validate_data], early_stopping_rounds=5, verbose_eval=25) 

        

    return m_lgb
bst = create_GBMmodel(X_feature_train, y_feature_train, X_feature_validate, y_feature_validate, cat_cols)

y_test = bst.predict(X_feature_test, num_iteration=bst.best_iteration)



test_feature_df['target'] = y_test



test_feature_df[['image_name', 'target']].to_csv('submission.csv', index=False)