import numpy as np

import pandas as pd

import os

import cv2

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint

from tqdm import tqdm

from keras import optimizers



from sklearn.cross_validation import KFold

from sklearn.metrics import fbeta_score

from sklearn.utils import shuffle
labels = ['blow_down',

 'bare_ground',

 'conventional_mine',

 'blooming',

 'cultivation',

 'artisinal_mine',

 'haze',

 'primary',

 'slash_burn',

 'habitation',

 'clear',

 'road',

 'selective_logging',

 'partly_cloudy',

 'agriculture',

 'water',

 'cloudy']



label_map = {'agriculture': 14,

 'artisinal_mine': 5,

 'bare_ground': 1,

 'blooming': 3,

 'blow_down': 0,

 'clear': 10,

 'cloudy': 16,

 'conventional_mine': 2,

 'cultivation': 4,

 'habitation': 9,

 'haze': 6,

 'partly_cloudy': 13,

 'primary': 7,

 'road': 11,

 'selective_logging': 12,

 'slash_burn': 8,

 'water': 15}



image_size=128

def Amazon_Model(input_shape=(128, 128,3),weight_path=None):

    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))

        

    model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

        

    model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

        

    model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))

    model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

        

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(17, activation='sigmoid'))

    if(weight_path!=None):

        if os.path.isfile(weight_path):

            model.load_weights(weight_path)

    return model
def Amazon_Model_VGG19(input_shape=(128, 128,3),weight_path=None):

    from keras.applications.vgg19 import VGG19

    base_model=VGG19(include_top=False,

                   weights='imagenet',

                   input_shape=input_shape)



    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))

    model.add(base_model)

    model.add(Flatten())

    # I don't use 4096 Dense layer because it might get higher score on LB.

    # Source: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36091#202629

    #model.add(Dense(4096, activation='relu'))

    #model.add(Dense(4096, activation='relu'))

    model.add(Dense(17, activation='sigmoid'))

    if(weight_path!=None):

        if os.path.isfile(weight_path):

            model.load_weights(weight_path)

    return model



def Amazon_Model_VGG16(input_shape=(128, 128,3),weight_path=None):

    from keras.applications.vgg16 import VGG16

    base_model=VGG19(include_top=False,

                   weights='imagenet',

                   input_shape=input_shape)



    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))

    model.add(base_model)

    model.add(Flatten())

    # I don't use 4096 Dense layer because it might get higher score on LB.

    # Source: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36091#202629

    #model.add(Dense(4096, activation='relu'))

    #model.add(Dense(4096, activation='relu'))

    model.add(Dense(17, activation='sigmoid'))

    if(weight_path!=None):

        if os.path.isfile(weight_path):

            model.load_weights(weight_path)

    return model
def KFold_Train(x_train,y_train,nfolds=5,batch_size=128):

    model = Amazon_Model()

    kf = KFold(len(y_train), n_folds=nfolds, shuffle=False, random_state=1)

    num_fold = 0

    for train_index, test_index in kf:

    

        X_train = x_train[train_index]

        Y_train = y_train[train_index]

        X_valid = x_train[test_index]

        Y_valid = y_train[test_index]



        num_fold += 1

        print('Start KFold number {} from {}'.format(num_fold, nfolds))

        print('Split train: ', len(X_train), len(Y_train))

        print('Split valid: ', len(X_valid), len(Y_valid))

        weight_path = os.path.join('', '../h5_128_rotate_uint8/weights_kfold_' + str(num_fold) + '.h5')

        if os.path.isfile(weight_path):

            model.load_weights(weight_path)

        

        # I forgot what's the setting here

        # Maybe like these

        epochs_arr = [60, 15, 15]

        learn_rates = [0.001, 0.0001, 0.00001]



        for learn_rate, epochs in zip(learn_rates, epochs_arr):

            opt  = optimizers.Adam(lr=learn_rate)

            model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

            callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),

            ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=0)]



            model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),

                  batch_size=batch_size,verbose=2, epochs=epochs,callbacks=callbacks,shuffle=True)

        

        p_valid = model.predict(X_valid, batch_size = batch_size, verbose=2)

        print(fbeta_score(Y_valid, np.array(p_valid) > 0.18, beta=2, average='samples'))
def KFold_Predict(x_test,nfolds=5,batch_size=128):

    model = Amazon_Model()

    yfull_test = []

    for num_fold in range(1,nfolds+1):

        weight_path = os.path.join('', '../h5_128_rotate_uint8/weights_kfold_' + str(num_fold) + '.h5')

        if os.path.isfile(weight_path):

            model.load_weights(weight_path)

            

        p_test = model.predict(x_test, batch_size = batch_size, verbose=2)

        yfull_test.append(p_test)

        

    result = np.array(yfull_test[0])

    for i in range(1, nfolds):

        result += np.array(yfull_test[i])

    result /= nfolds

    return result
def Train():

    x_train = []

    y_train = []



    df_train = pd.read_csv('../input/train_v2.csv')

    df_train = shuffle(df_train,random_state=0)

    for f, tags in tqdm(df_train.values, miniters=400):

        img = cv2.imread('C:/train-jpg/{}.jpg'.format(f))

        targets = np.zeros(17)

        for t in tags.split(' '):

            targets[label_map[t]] = 1 

        img = cv2.resize(img, (image_size, image_size))

        flipped_img=cv2.flip(img,1)

        rows,cols,channel = img.shape

        # regular

        x_train.append(img)

        y_train.append(targets)

        

        # flipped

        x_train.append(flipped_img)

        y_train.append(targets)

        # rotated

        for rotate_degree in [90,180,270]:

            M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_degree,1)

            dst = cv2.warpAffine(img,M,(cols,rows))

            x_train.append(dst)

            y_train.append(targets)

            

            dst = cv2.warpAffine(flipped_img,M,(cols,rows))

            x_train.append(dst)

            y_train.append(targets)

        

    y_train = np.array(y_train, np.uint8)

    x_train = np.array(x_train, np.uint8)

    KFold_Train(x_train,y_train)
def Predict():

    df_test = pd.read_csv('../input/sample_submission_v2.csv')

    

    x_test = []

    for f, tags in tqdm(df_test.values, miniters=400):

        img = cv2.imread('C:/test-jpg/{}.jpg'.format(f))

        x_test.append(cv2.resize(img, (image_size, image_size)))

    x_test  = np.array(x_test, np.uint8)

    

    result = KFold_Predict(x_test)

    result = pd.DataFrame(result, columns = labels)

    

    

    thres = {   'blow_down':0.2,

                'bare_ground':0.138,

                'conventional_mine':0.1,

                'blooming':0.168,

                'cultivation':0.204,

                'artisinal_mine':0.114,

                'haze':0.204,

                'primary':0.204,

                'slash_burn':0.38,

                'habitation':0.17,

                'clear':0.13,

                'road':0.156,

                'selective_logging':0.154,

                'partly_cloudy':0.112,

                'agriculture':0.164,

                'water':0.182,

                'cloudy':0.076}

    

    

    preds = []

    for i in tqdm(range(result.shape[0]), miniters=1000):

        a = result.ix[[i]]

        pred_tag=[]

        for k,v in thres.items():

            if(a[k][i]>=v):

                pred_tag.append(k)

        preds.append(' '.join(pred_tag))

        

    df_test['tags'] = preds

    df_test.to_csv('sub.csv', index=False)
def main():

    #Train()

    Predict()

if __name__ == '__main__':

    main()