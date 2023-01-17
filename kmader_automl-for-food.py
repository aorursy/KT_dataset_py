import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils.io_utils import HDF5Matrix
base_path = os.path.join('..', 'input')
train_h5_path = os.path.join(base_path, 'food_c101_n10099_r32x32x3.h5')
test_h5_path = os.path.join(base_path, 'food_test_c101_n1000_r32x32x3.h5')
%matplotlib inline
X_train = HDF5Matrix(train_h5_path, 'images')[:2000]
y_train = HDF5Matrix(train_h5_path, 'category')[:2000]
y_train_cat = np.argmax(y_train,-1)
print('In Data',X_train.shape,'=>', y_train.shape, '=>', y_train_cat.shape)
X_test = HDF5Matrix(test_h5_path, 'images')[:]
y_test = HDF5Matrix(test_h5_path, 'category')[:]
y_test_cat = np.argmax(y_test,-1)
print('In Data',X_test.shape,'=>', y_test.shape, '=>', y_test_cat.shape)
sample_imgs = 25
with h5py.File(train_h5_path, 'r') as n_file:
    total_imgs = n_file['images'].shape[0]
    read_idxs = slice(0,sample_imgs)
    im_data = n_file['images'][read_idxs]
    im_label = n_file['category'].value[read_idxs]
    label_names = [x.decode() for x in n_file['category_names'].value]
fig, m_ax = plt.subplots(5, 5, figsize = (12, 12))
for c_ax, c_label, c_img in zip(m_ax.flatten(), im_label, im_data):
    c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')
    c_ax.axis('off')
    c_ax.set_title(label_names[np.argmax(c_label)])
from tpot import TPOTClassifier
from sklearn.preprocessing import FunctionTransformer, Normalizer
from sklearn.pipeline import make_pipeline
full_pipeline = make_pipeline(
    FunctionTransformer(lambda x: x.reshape((x.shape[0],-1)), validate = False), 
    Normalizer(),
    TPOTClassifier(generations=1, population_size=3, verbosity=2, random_state = 1234,
                  max_eval_time_mins=0.1)
)
%%time
full_pipeline.fit(X_train, y_train_cat)
%%time
y_train_pred = full_pipeline.predict(X_train)
y_pred = full_pipeline.predict(X_test)
from sklearn.metrics import classification_report, classification, accuracy_score
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
ax1.matshow(classification.confusion_matrix(y_train_cat, y_train_pred))
ax1.set_title('Training Results')
ax2.matshow(classification.confusion_matrix(y_test_cat, y_pred))
ax2.set_title('Validation Results')
print('Validation Accuracy: %2.1f%%' % (100*accuracy_score(y_test_cat, y_pred)))
print(classification_report(y_test_cat, y_pred))
sample_imgs = 16
with h5py.File(test_h5_path, 'r') as n_file:
    total_imgs = n_file['images'].shape[0]
    read_idxs = slice(0,sample_imgs)
    im_data = n_file['images'][read_idxs]
    im_label = n_file['category'].value[read_idxs]
    label_names = [x.decode() for x in n_file['category_names'].value]
pred_label = full_pipeline.predict_proba(im_data)
fig, m_ax = plt.subplots(4, 4, figsize = (20, 20))
for c_ax, c_label, c_pred, c_img in zip(m_ax.flatten(), im_label, pred_label, im_data):
    c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')
    c_ax.axis('off')
    c_ax.set_title('Predicted:{}\nActual:{}'.format(label_names[np.argmax(c_pred)],
                                                  label_names[np.argmax(c_label)]))
