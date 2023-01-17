%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os, sys
DATA_ROOT_PATH = os.path.join('..', 'input')
!ls -lh {DATA_ROOT_PATH} # show the data
with np.load(os.path.join(DATA_ROOT_PATH, 'train.npz')) as npz_data:
    train_img = npz_data['img']
    train_idx = npz_data['idx'] # the id for each image so we can match the labels
    print('image shape', train_img.shape)
    print('idx shape', train_idx.shape)
train_labels = pd.read_csv(os.path.join(DATA_ROOT_PATH, 'train_labels.csv'))
train_dict = dict(zip(train_labels['idx'], train_labels['label'])) # map idx to label
train_labels.head(4)
fig, m_axs = plt.subplots(3, 3, figsize = (10, 10))
for c_ax, c_img, c_idx in zip(m_axs.flatten(), train_img, train_idx):
    c_ax.matshow(c_img)
    c_ax.set_title('{}'.format(train_dict[c_idx]))
    c_ax.axis('off')
# Preprocess data so we can fit it into the model
train_flat_vec = train_img.reshape((train_img.shape[0], -1))
train_flat_label = np.array([train_dict[idx] for idx in train_idx])
print(train_flat_vec.shape, train_flat_label.shape)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression() 
MAX_TRAIN_POINTS = 5000
lr_model.fit(train_flat_vec[:MAX_TRAIN_POINTS], train_flat_label[:MAX_TRAIN_POINTS]) # train model on first MAX_TRAIN_POINTS (can be very memory intensive)
with np.load(os.path.join(DATA_ROOT_PATH, 'valid.npz')) as npz_data:
    valid_img = npz_data['img']
    valid_idx = npz_data['idx'] # the id for each image so we can match the labels
valid_labels = pd.read_csv(os.path.join(DATA_ROOT_PATH, 'valid_labels.csv'))
valid_dict = dict(zip(valid_labels['idx'], valid_labels['label'])) # map idx to label
valid_flat_vec = valid_img.reshape((valid_img.shape[0], -1))
valid_flat_label = np.array([valid_dict[idx] for idx in valid_idx])
from sklearn.metrics import confusion_matrix, accuracy_score
valid_pred = lr_model.predict(valid_flat_vec)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(confusion_matrix(valid_flat_label, valid_pred), ax=ax1)
ax1.set_title('Accuracy: {:2.2%}'.format(accuracy_score(valid_flat_label, valid_pred)));
with np.load(os.path.join(DATA_ROOT_PATH, 'test.npz')) as npz_data:
    test_img = npz_data['img']
    test_idx = npz_data['idx'] # the id for each image so we can match the labels
test_flat_vec = test_img.reshape((test_img.shape[0], -1))
test_pred_df = pd.DataFrame([{'idx': idx, 'label': label} for idx, label in zip(test_idx, lr_model.predict(test_flat_vec))])
test_pred_df.sample(5)
test_pred_df.to_csv('predictions.csv', index=False)
