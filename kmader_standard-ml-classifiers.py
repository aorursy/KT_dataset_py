import h5py 

import numpy as np

import matplotlib.pyplot as plt
with h5py.File('../input/all_mias_scans.h5', 'r') as scan_h5:

    bg_info = scan_h5['BG'][:]

    class_info = scan_h5['CLASS'][:]

    # low res scans

    scan_lr = scan_h5['scan'][:][:, ::16, ::16]
scan_lr_flat = scan_lr.reshape((scan_lr.shape[0], -1))
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()

class_le.fit(class_info)

class_vec = class_le.transform(class_info)

class_le.classes_
from sklearn.model_selection import train_test_split

idx_vec = np.arange(scan_lr_flat.shape[0])

x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(scan_lr_flat, 

                                                    class_vec, 

                                                    idx_vec,

                                                    random_state = 2017,

                                                   test_size = 0.5,

                                                   stratify = class_vec)

print('Training', x_train.shape)

print('Testing', x_test.shape)
# useful tools

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

creport = lambda gt_vec,pred_vec: classification_report(gt_vec, pred_vec, 

                                                        target_names = [x.decode() for x in 

                                                                        class_le.classes_])
from sklearn.dummy import DummyClassifier

dc = DummyClassifier(strategy='most_frequent')

dc.fit(x_train, y_train)

y_pred = dc.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
%%time

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(8)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
%%time

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
%%time

from xgboost import XGBClassifier

xgc = XGBClassifier(silent = False, nthread=2)

xgc.fit(x_train, y_train)

y_pred = xgc.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
from tpot import TPOTClassifier

tpc = TPOTClassifier(generations = 2, population_size=5, verbosity=True)

tpc.fit(x_train, y_train)

y_pred = tpc.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
fig, ax1 = plt.subplots(1,1)

ax1.matshow(np.log10(confusion_matrix(y_test, y_pred).clip(0.5,1e9)), cmap = 'RdBu')

ax1.set_xticks(range(len(class_le.classes_)))

ax1.set_xticklabels([x.decode() for x in class_le.classes_])

ax1.set_yticks(range(len(class_le.classes_)))

ax1.set_yticklabels([x.decode() for x in class_le.classes_])

ax1.set_xlabel('Predicted Class')

ax1.set_ylabel('Actual Class')
fig, c_axs = plt.subplots(3,3, figsize = (12,12))

for c_ax, test_idx in zip(c_axs.flatten(), np.where(y_pred!=y_test)[0]):

    c_idx = idx_test[test_idx]

    c_ax.imshow(scan_lr[c_idx], cmap = 'bone')

    c_ax.set_title('Predict: %s\nActual: %s' % (class_le.classes_[y_pred[test_idx]].decode(),

                                               class_le.classes_[y_test[test_idx]].decode()))

    c_ax.axis('off')
