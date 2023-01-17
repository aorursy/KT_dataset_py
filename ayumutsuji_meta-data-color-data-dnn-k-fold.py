import tensorflow as tf

import numpy as np

import pandas as pd

import os

from tqdm import tqdm

from matplotlib import pyplot

from tensorflow.keras.layers import Dense,Activation,Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam





from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



from sklearn.model_selection import StratifiedKFold



#おしゃれ化

pyplot.style.use('ggplot')
df_train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

df_test =  pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")



img_stats_path = '/kaggle/input/melanoma2020imgtabular'

#train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

#test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
#getting image_info

#reference notebook:https://www.kaggle.com/datafan07/starter-analysis-of-melanoma-metadata-and-images/data?

"""

from keras.preprocessing import image



for data, location in zip([df_train, df_test],[train_img_path, test_img_path]):

    images = data['image_name'].values

    reds = np.zeros(images.shape[0])

    greens = np.zeros(images.shape[0])

    blues = np.zeros(images.shape[0])

    mean = np.zeros(images.shape[0])

    x = np.zeros(images.shape[0], dtype=int)

    y = np.zeros(images.shape[0], dtype=int)

    for i, path in enumerate(tqdm(images)):

        img = np.array(image.load_img(os.path.join(location, f'{path}.jpg')))



        reds[i] = np.mean(img[:,:,0].ravel())

        greens[i] = np.mean(img[:,:,1].ravel())

        blues[i] = np.mean(img[:,:,2].ravel())

        mean[i] = np.mean(img)

        x[i] = img.shape[1]

        y[i] = img.shape[0]



    data['reds'] = reds

    data['greens'] = greens

    data['blues'] = blues

    data['mean_colors'] = mean

    data['width'] = x

    data['height'] = y



df_train['total_pixels']= df_train['width']*df_train['height']

df_test['total_pixels']= df_test['width'].astype(str)*df_test['height']

"""
# Loading color data:

train_attr = pd.read_csv(

    os.path.join(img_stats_path, 'train_mean_colorres.csv'))

test_attr = pd.read_csv(os.path.join(img_stats_path, 'test_mean_colorres.csv'))
train_attr
df_train = pd.concat([df_train, train_attr], axis=1)

df_test = pd.concat([df_test, test_attr], axis=1)
drop_list = ["image_name","patient_id","diagnosis","benign_malignant"]

drop_list_test = ["image_name","patient_id"]

df_train = df_train.drop(drop_list,axis =1)

df_test  = df_test.drop(drop_list_test,axis =1)
#get_dummies



train_dummy = pd.get_dummies(df_train['anatom_site_general_challenge'])



train_dummy2 = pd.get_dummies(df_train["sex"])



df_train = pd.concat([df_train.drop(['anatom_site_general_challenge',"sex"],axis=1),train_dummy,train_dummy2],axis=1)



test_dummy = pd.get_dummies(df_test['anatom_site_general_challenge'])



test_dummy2 = pd.get_dummies(df_test["sex"])



df_test = pd.concat([df_test.drop(['anatom_site_general_challenge',"sex"],axis=1),test_dummy,test_dummy2],axis=1)
#train_data and label

X = df_train.drop("target",axis=1)



Y = df_train["target"]
#standarization

def standarization(df,column_list):

    for column in column_list:

        df[column] = ((df[column] - df[column].mean())/df[column].std())

    return
X
column_list = ["age_approx",'reds','greens','blues','mean_colors','width','height','total_pixels']

standarization(X,column_list)

standarization(df_test,column_list)
X.isnull().any()
# fillna

X["age_approx"] = X["age_approx"].fillna(X["age_approx"].mean())
X
def build_seq_model(features):

    model = Sequential()

    model.add(Dense(16, activation='relu',input_dim=features))

    model.add(Dropout(0.25))

    model.add(Dense(1,activation="sigmoid"))

    model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["accuracy"])

    return model
features = X.shape[1]

model = build_seq_model(features)



model.summary()
#https://keras.io/ja/visualization/

import graphviz

from keras.utils import plot_model

tf.keras.utils.plot_model(model, to_file='model.png')
X = X.values

Y = Y.values



#test data

df_test = df_test.values
#stratified k-fold



# number for CV

fold_num = 5 



# fix random seed for reproducibility

seed = 7

np.random.seed(seed)



# define X-fold cross validation

kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)



#make list

AUROC = []



#keras callbacks

es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

i=1

for train, test in kfold.split(X, Y):

    print(f"fold{i} start")

    #print(str(#*50))

    model = build_seq_model(features)

    history = model.fit(

        X[train],

        Y[train],

        verbose=1,

        epochs=50,

        batch_size = 128,

        validation_data=(X[test], Y[test]),

        class_weight = {0:1,1:3},

        callbacks=[es_cb])

    

    # plot history

    pyplot.plot(history.history['loss'], label='train')

    pyplot.plot(history.history['val_loss'], label='valid')

    pyplot.legend()

    pyplot.show()

    

    pred = model.predict(X[test])

    AUROC_score = roc_auc_score(Y[test], pred)

    

    AUROC.append(AUROC_score)

    

    i += 1



ave = sum(AUROC) / len(AUROC)

print(f"AUROC_list:{AUROC}")

print("mean_AUROC:" + str(ave))
#retrain model (using all-data)

model = build_seq_model(features)
model.fit(

        X,

        Y,

        verbose=1,

        epochs=50,

        batch_size = 128,

        validation_data=(X[test], Y[test]),

        class_weight = {0:1,1:3},

        callbacks=[es_cb])
submit = model.predict(df_test)

submit
submission = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")

submission["target"] = submit

submission.to_csv('submission.csv', index=False)

submission.head()