!pip install -q efficientnet

import efficientnet.tfkeras as efn

import numpy as np

import pandas as pd

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import xgboost as xgb

from scipy.special import softmax



IMAGE_SIZE = [256, 256]

DO_PARAMETER_TUNING = False
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

image_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
columns = train_df.columns

target_columns = train_df.columns.drop("image_id")

train_target = train_df[target_columns]

display(train_target.head())



class_names = list(target_columns.values)

train_labels = train_target.idxmax(axis=1)

sns.countplot(train_labels)
train_files_labels = pd.concat([train_df["image_id"], train_labels], axis=1)

train_files_labels.columns = ['file', 'label']

train_files_labels['file'] = train_files_labels['file'].apply(lambda x : x+".jpg")



display(train_files_labels.head())
ident_gen = ImageDataGenerator(rescale=1./255)



train_gen = ident_gen.flow_from_dataframe(

    train_files_labels, 

    directory=image_dir, 

    x_col='file', 

    y_col='label', 

    target_size=IMAGE_SIZE, 

    classes=class_names,

    class_mode='categorical', 

    batch_size=25, 

    shuffle=True)



def show_batch(image_batch, label_batch, true_label_batch=[]):

    plt.figure(figsize=(15,15))

    for n in range(min(25, len(image_batch))):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n])

        title = class_names[label_batch[n].argmax()]

        if len(true_label_batch):

            title += f" ({class_names[true_label_batch[n].argmax()]})"

        plt.title(title)

        plt.axis('off')



image_batch, label_batch = next(train_gen)

show_batch(image_batch, label_batch)
def feature_extractor_model():

    pretrained_model = efn.EfficientNetB5(

        input_shape=[*IMAGE_SIZE, 3],

        weights='imagenet',

        include_top=False)

    pretrained_model.trainable = False

    return tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D()

    ])



feature_model = feature_extractor_model()

feature_model.summary()
%%time

# Load all the images in a single batch

train_batch_gen = ident_gen.flow_from_dataframe(

    train_files_labels, 

    directory=image_dir, 

    x_col='file', 

    y_col='label', 

    target_size=IMAGE_SIZE, 

    color_mode='rgb', 

    classes=class_names,

    class_mode='categorical', 

    batch_size=len(train_files_labels), 

    shuffle=True)



train_X_all, train_y_all = next(train_batch_gen)
%%time

train_y_all = train_y_all.argmax(axis=-1) # onehot -> labels

train_X, valid_X, train_y, valid_y = train_test_split(train_X_all, train_y_all, test_size=0.1, random_state=99)



train_X_feat = feature_model.predict(train_X)

valid_X_feat = feature_model.predict(valid_X)



print(train_X_feat.shape)

print(train_y.shape)



print(train_X.shape)

print(train_y.shape)
%%time

xgb_model = XGBRegressor(objective='multi:softmax', 

                         num_class=4)

xgb_model.fit(train_X_feat, train_y)
from sklearn.metrics import roc_auc_score, accuracy_score



def to_onehot(labels, n_classes=4):

    m = np.zeros(shape=(labels.size, n_classes))

    m[np.arange(labels.size), labels.astype('int')] = 1

    return m



def get_acc_scores(y_true_labels, y_pred_labels):

    y_true = to_onehot(y_true_labels)

    y_pred = to_onehot(y_pred_labels)

    return {

        cat : accuracy_score(y_true.T[i], y_pred.T[i]) for 

        (i, cat) in enumerate(class_names)

    }



def get_auc_scores(y_true_labels, y_pred_labels):

    y_true = to_onehot(y_true_labels)

    y_pred = to_onehot(y_pred_labels)

    return {

        cat : roc_auc_score(y_true.T[i], y_pred.T[i]) for 

        (i, cat) in enumerate(class_names)

    }
pred_y = xgb_model.predict(train_X_feat)



print("Accuracy on training set:", get_acc_scores(train_y, pred_y))

print("ROC AUC on training set:", get_auc_scores(train_y, pred_y))
pred_y = xgb_model.predict(valid_X_feat)



print(valid_y.shape)

print(train_y.shape)

print(pred_y[:3])



val_auc_scores = get_auc_scores(valid_y, pred_y)

print("Accuracy on validation set:", get_acc_scores(valid_y, pred_y))

print("ROC AUC on validation set:", val_auc_scores)



print("\nMean:",np.mean(list(val_auc_scores.values())))
correct_preds = valid_y == pred_y

bad_preds = valid_y != pred_y



print("#Mistakes", sum(bad_preds))



mistakes_X = valid_X[bad_preds]

mistakes_y = pred_y[bad_preds]

mistakes_y_true = valid_y[bad_preds]



print(mistakes_X.shape)



show_batch(mistakes_X, to_onehot(mistakes_y), to_onehot(mistakes_y_true))

# see: https://www.kaggle.com/agungor2/various-confusion-matrix-plots

from sklearn.metrics import confusion_matrix



data = confusion_matrix(pred_y, valid_y)



df_cm = pd.DataFrame(data, columns = class_names, index = class_names)

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'



sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})# font size
fig, axes = plt.subplots(1,2, figsize=(12,6))



mistake_pred_data = {

    'healthy': mistakes_y == 0,

    'multiple_diseases': mistakes_y == 1,

    'rust': mistakes_y == 2,

    'scab': mistakes_y == 3}

mistakes_pred_df = pd.DataFrame(data=mistake_pred_data, dtype='int')

ax = sns.countplot(mistakes_pred_df.idxmax(axis=1), ax=axes[0])

ax.set_title("False positives")



mistake_data_true = {

    'healthy': mistakes_y_true == 0,

    'multiple_diseases': mistakes_y_true == 1,

    'rust': mistakes_y_true == 2,

    'scab': mistakes_y_true == 3}

mistakes_true_df = pd.DataFrame(data=mistake_data_true, dtype='int')

ax = sns.countplot(mistakes_true_df.idxmax(axis=1), ax=axes[1])

ax.set_title("False negatives")

plt.show()
from sklearn.model_selection import GridSearchCV



def get_mean_auc(estimator, X, y_true_labels):

    y_pred_labels = estimator.predict(X)

    y_true = to_onehot(y_true_labels)

    y_pred = to_onehot(y_pred_labels)

    return np.mean([roc_auc_score(y_true.T[i], y_pred.T[i]) for i in range(len(class_names))])



if DO_PARAMETER_TUNING:

    parameters = {

        "n_estimators": [5,25,50,75],

        "max_depth": [2,4,6],

        "learning_rate":  [0.2, 0.3, 0.4]

    }



    xgb = XGBRegressor(objective='multi:softmax', num_class=4, threads=1)

    search = GridSearchCV(xgb, parameters, scoring=get_mean_auc, n_jobs=-1, cv=3, verbose=3)

    search.fit(train_X_feat, train_y)

    tuned_params = search.best_params_

else:

    tuned_params = {'learning_rate': 0.4, 'max_depth': 2, 'n_estimators': 75}



print(tuned_params)
%%time

xgb_model_rust = XGBRegressor(n_estimators=tuned_params['n_estimators'],

                              max_depth=tuned_params['max_depth'],

                              learning_rate=tuned_params['learning_rate'],

                              objective="binary:logistic")

xgb_model_scab = XGBRegressor(n_estimators=tuned_params['n_estimators'],

                              max_depth=tuned_params['max_depth'],

                              learning_rate=tuned_params['learning_rate'],

                              objective="binary:logistic")



# need to split our data into two sets:

# - one with combined 'rust' and 'multiple_diseases' as the prediction target

# - another with 'scab' and 'multiple_diseases'



healthy_idx = train_y == class_names.index("healthy")

multiple_idx = train_y == class_names.index("multiple_diseases")

rust_idx = train_y == class_names.index("rust")

scab_idx = train_y == class_names.index("scab")



has_rust_idx = rust_idx + multiple_idx

has_scab_idx = scab_idx + multiple_idx



xgb_model_rust.fit(train_X_feat, has_rust_idx)

xgb_model_scab.fit(train_X_feat, has_scab_idx)
rust_pred_y = xgb_model_rust.predict(valid_X_feat)

scab_pred_y = xgb_model_scab.predict(valid_X_feat)



val_multiple_idx = valid_y == class_names.index("multiple_diseases")

val_healthy_idx = valid_y == class_names.index("healthy")

val_rust_idx = valid_y == class_names.index("rust")

val_scab_idx = valid_y == class_names.index("scab")



rust_auc = roc_auc_score(val_rust_idx, rust_pred_y)

scab_auc = roc_auc_score(val_scab_idx, scab_pred_y)

mult_auc = roc_auc_score(val_multiple_idx, np.minimum(scab_pred_y, rust_pred_y))

heal_auc = roc_auc_score(val_healthy_idx, 1 - np.maximum(scab_pred_y, rust_pred_y))



print("Rust AUC", rust_auc)

print("Scab AUC", scab_auc)

print("Multiple AUC", mult_auc)

print("Healthy AUC", heal_auc)



print("\nMean AUC", (rust_auc+scab_auc+mult_auc+heal_auc)/4)
def double_model_predict(X):

    rust_pred_y = xgb_model_rust.predict(X)

    scab_pred_y = xgb_model_scab.predict(X)

    

    multiple_pred_y = np.minimum(scab_pred_y, rust_pred_y)

    healthy_pred_y = 1 - np.maximum(scab_pred_y, rust_pred_y)

    

    return np.stack((healthy_pred_y,multiple_pred_y,rust_pred_y,scab_pred_y)).T

    

print(double_model_predict(valid_X_feat)[:3])
%%time

test_files = pd.concat([test_df["image_id"], 

                        test_df["image_id"].apply(lambda x : x+".jpg")],

                        axis=1)

test_files.columns = ["image_id", "file"]



test_batch_gen = ident_gen.flow_from_dataframe(

    test_files, 

    directory=image_dir, 

    x_col='file', 

    target_size=IMAGE_SIZE, 

    color_mode='rgb',

    class_mode=None,

    batch_size=len(test_files),

    shuffle=False)



test_X_img = next(test_batch_gen)
%%time

test_X_feat = feature_model.predict(test_X_img)
%%time

test_y = double_model_predict(test_X_feat)
show_batch(test_X_img, to_onehot(np.argmax(test_y, axis=-1)))
print(test_y.shape)

#test_y = softmax(test_y, axis=1)

ids = test_files['image_id'].to_numpy()

print(ids.shape)



np.savetxt('submission.csv', 

           np.rec.fromarrays([ids] + [test_y[:,i] for i in range(4)]), 

           fmt=['%s', '%.2f', '%.2f', '%.2f', '%.2f'], 

           delimiter=',', 

           header='image_id,healthy,multiple_diseases,rust,scab', 

           comments='')



!head submission.csv