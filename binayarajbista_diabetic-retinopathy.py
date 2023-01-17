import numpy as np 
import pandas as pd 
import os
import numpy as np
import os
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Flatten,SpatialDropout2D
from tensorflow.keras.layers import concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
from tensorflow.keras.applications.inception_v3 import InceptionV3 as PTModel
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
%matplotlib inline
base_image_dir = os.path.join('..', 'input', 'diabetic-retinopathy-resized')
print(base_image_dir)
retina_df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels.csv'))

indexNames = retina_df[ retina_df['level'] == 3 ].index
 
# Delete these row indexes from dataFrame
retina_df.drop(indexNames , inplace=True)

indexNames = retina_df[ retina_df['level'] == 2 ].index
retina_df.drop(indexNames , inplace=True)
indexNames = retina_df[ retina_df['level'] == 1 ].index
 
# Delete these row indexes from dataFrame
retina_df.drop(indexNames , inplace=True)

retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(base_image_dir,'resized_train','resized_train',
                                                         '{}.jpeg'.format(x)))
retina_df['exists'] = retina_df['path'].map(os.path.exists)
print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')
retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0)
from keras.utils.np_utils import to_categorical
retina_df['level'].replace(4,1,inplace=True)
retina_df['level_cat'] = retina_df['level'].map(lambda x: to_categorical(x, 1+retina_df['level'].max()))

retina_df.dropna(inplace = True)
retina_df = retina_df[retina_df['exists']]


retina_df[retina_df['level']==1].shape
retina_df[['level', 'eye']].hist(figsize = (10, 5))
rr_df = retina_df[['PatientId', 'level']].drop_duplicates()
train_ids, valid_ids = train_test_split(rr_df['PatientId'], 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = rr_df['level'])
raw_train_df = retina_df[retina_df['PatientId'].isin(train_ids)]
valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)]
valid_df = valid_df.groupby(['level', 'eye']).apply(lambda x: x.sample(86, replace = False)
                                                      ).reset_index(drop = True)
valid_df.drop_duplicates(subset="image", keep='first', inplace=True)
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])


valid_df.drop_duplicates(subset="image", keep='first', inplace=True)
retina_df[(retina_df['level']==1) &( retina_df['eye']==1)]
train_df = raw_train_df.groupby(['level', 'eye']).apply(lambda x: x.sample(4000, replace = True)
                                                      ).reset_index(drop = True)
train_df.drop_duplicates(subset='image', keep='first', inplace=False)
#retina_df.drop_duplicates(subset=None, keep='first', inplace=False)
print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
train_df[['level', 'eye']].hist(figsize = (10, 5))

batch_size = 200
IMG_SIZE = (512, 512)
train_df['level'] = train_df['level'].map(lambda x: str(x))
valid_df['level'] = valid_df['level'].map(lambda x: str(x))

train_df.head()

import tensorflow
train_datagen = ImageDataGenerator(preprocessing_function=tensorflow.keras.applications.inception_v3.preprocess_input,
                                   height_shift_range=0.01,
                                   width_shift_range=0.01,
                                   brightness_range=(0.8,1.2),
                                   horizontal_flip=True,
                                   shear_range=0.01,
                                   vertical_flip=True,
                                   rotation_range=10,
                                   zoom_range=0.05,
#                                     featurewise_std_normalization=True,
#                                     featurewise_center=True
#                                    samplewise_std_normalization=True,
                                  )

test_datagen = ImageDataGenerator(preprocessing_function=tensorflow.keras.applications.inception_v3.preprocess_input)

train_gen=train_datagen.flow_from_dataframe(dataframe=train_df,
                                            x_col="path",
                                            y_col="level",
                                            batch_size=batch_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=IMG_SIZE)
valid_gen=test_datagen.flow_from_dataframe(dataframe=valid_df,
                                            x_col="path",
                                            y_col="level",
                                            batch_size=batch_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=IMG_SIZE)
t_x, t_y = next(valid_gen)
fig, m_axs = plt.subplots(3, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(np.clip(c_x*127+127, 0, 255).astype(np.uint8))
    c_ax.set_title('Retinopathy: {}'.format(np.argmax(c_y, -1)))
    c_ax.axis('off')
t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(3, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(np.clip(c_x*127+127, 0, 255).astype(np.uint8))
    c_ax.set_title('Retinopathy {}'.format(np.argmax(c_y, -1)))
    c_ax.axis('off')
in_lay = Input(t_x.shape[1:])
base_pretrained_model = PTModel(input_shape =  t_x.shape[1:], include_top = False, weights =  os.path.join('..', 'input', 'inceptionv3','inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'))
for layers in base_pretrained_model.layers[:-80]:
    layers.trainable = False
base_pretrained_model.summary()
# from keras.applications.vgg16 import VGG16 as PTModel


pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)
bn_features = BatchNormalization()(pt_features)

# here we do an attention mechanism to turn pixels in the GAP on an off

attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.6)(bn_features))
attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(1, 
                    kernel_size = (1,1), 
                    padding = 'valid', 
                    activation = 'sigmoid')(attn_layer)
# fan it out to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
               activation = 'linear', use_bias = False, weights = [up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.50)(gap)
dr_steps = Dropout(0.50)(Dense(128, activation = 'relu')(gap_dr))
out_layer = Dense(t_y.shape[-1], activation = 'softmax', name='visualized_layer')(dr_steps)
retina_model = Model(inputs = [in_lay], outputs = [out_layer])

retina_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])
retina_model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_newweights.best.hdf5".format('retina')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=7) 
callbacks_list = [checkpoint, early, reduceLROnPlat]
history=retina_model.fit(train_gen, 
                           steps_per_epoch = train_df.shape[0]//batch_size,
                           validation_data = valid_gen,
                             validation_steps = valid_df.shape[0]//batch_size,
                              epochs =100, 
                              callbacks = callbacks_list,
                             workers = 0, 
                             use_multiprocessing=True, 
                             max_queue_size = 0
                            )
cm_batch = valid_gen
test_labels = cm_batch.classes

predictions = retina_model.predict(cm_batch, steps=len(cm_batch), verbose=0)
cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
cm_batch.class_indices
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm_plot_labels = ['0','1']
import itertools
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
def plot_model_history(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['lr'])
    plt.title('learning rate')
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.show()
plot_model_history(history)
retina_model.save('/kaggle/working/final_model.h5')
##### create one fixed dataset for evaluating
from tqdm import tqdm_notebook
# fresh valid gen
# print(valid_df.head(4))
# valid_dfnew = valid_df.head(100)
# print(valid_dfnew.shape)
print(valid_df.shape[0]//batch_size-1)
vbatch_count = (valid_df.shape[0]//batch_size-1)
if(vbatch_count<0):
    vbatch_count=1
out_size = vbatch_count*batch_size
print(t_x.shape[1:])
test_X = np.zeros((out_size,)+t_x.shape[1:], dtype = np.float32)
test_Y = np.zeros((out_size,)+t_y.shape[1:], dtype = np.float32)
for i, (c_x, c_y) in zip(tqdm_notebook(range(vbatch_count)), 
                         valid_gen):
    j = i*batch_size
    test_X[j:(j+c_x.shape[0])] = c_x
    test_Y[j:(j+c_x.shape[0])] = c_y
# get the attention layer since it is the only one with a single output dim
for attn_layer in retina_model.layers:
    c_shape = attn_layer.get_output_shape_at(0)
    if len(c_shape)==4:
        if c_shape[-1]==1:
            print(attn_layer)
            break
from sklearn.metrics import accuracy_score, classification_report
pred_Y = retina_model.predict(test_X, batch_size = 32, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
test_Y_cat = np.argmax(test_Y, -1)
print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_Y_cat, pred_Y_cat)))
print(classification_report(test_Y_cat, pred_Y_cat))
import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(test_Y_cat, pred_Y_cat), 
            annot=True, fmt="d", cbar = False, cmap = plt.cm.Blues, vmax = test_X.shape[0]//16)
from sklearn.metrics import roc_curve, roc_auc_score
sick_vec = test_Y_cat>0
sick_score = np.sum(pred_Y[:,1:],1)
fpr, tpr, _ = roc_curve(sick_vec, sick_score)
fig, ax1 = plt.subplots(1,1, figsize = (6, 6), dpi = 150)
ax1.plot(fpr, tpr, 'b.-', label = 'Model Prediction (AUC: %2.2f)' % roc_auc_score(sick_vec, sick_score))
ax1.plot(fpr, fpr, 'g-', label = 'Random Guessing')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');
fig, m_axs = plt.subplots(2, 3, figsize = (32, 20))
for (idx, c_ax) in enumerate(m_axs.flatten()):
    c_ax.imshow(np.clip(test_X[idx]*127+127,0 , 255).astype(np.uint8), cmap = 'bone')
    c_ax.set_title('Actual Severity: {}\n{}'.format(test_Y_cat[idx], 
                                                           '\n'.join(['Predicted %02d (%04.1f%%): %s' % (k, 100*v, '*'*int(10*v)) for k, v in sorted(enumerate(pred_Y[idx]), key = lambda x: -1*x[1])])), loc='left')
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png', dpi = 300)

retina_model.save('16kfinal_model.h5')