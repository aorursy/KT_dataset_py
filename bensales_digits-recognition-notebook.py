import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import matplotlib.pyplot as plt
train_path = '/kaggle/input/digit-recognizer/train.csv'
test_path = '/kaggle/input/digit-recognizer/test.csv'
submission_file = 'submission.csv'
# pulling training data from csv file
train_data = pd.read_csv(train_path)

# pulling labels
train_labels = train_data['label'].to_numpy(dtype='int8')
train_data = train_data.drop(columns=['label']).to_numpy(dtype='float')

# pulling test data from csv
test_data = pd.read_csv(test_path).to_numpy(dtype='float')

# performing min-max scaling on data. min-max scaling is as follow:
# feature = (feature - min(feature)) / (max(feature) - min(feature))
min_max_scal = MinMaxScaler()
min_max_scal.fit(np.vstack((train_data, test_data)))

train_data = min_max_scal.transform(train_data)
test_data = min_max_scal.transform(test_data)
model = keras.Sequential([
    keras.layers.Reshape((28,28,1)),
    keras.layers.Conv2D(filters=100, kernel_size=2, activation='relu'),
    keras.layers.Conv2D(filters=200, kernel_size=2, activation='relu'),
    keras.layers.Conv2D(filters=300, kernel_size=2, activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters=400, kernel_size=2, activation='relu'),
    keras.layers.Conv2D(filters=500, kernel_size=2, activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=10, activation='softmax')
])
# Here we hot-encode labels. You might not. In fact, whether you use
# 'categorical_crossentropy' or 'sparse_categorical_crossentropy' as loss
# you ought to use respectively hot-encoded labels or labels as integers
vec_labels = np.zeros((train_labels.shape[0],10))
vec_labels[range(train_labels.shape[0]), train_labels] = 1
# performing cross-validation to evaluate model performance on unseen data
# we use stratified kfold with 5 folds here but for sake of simplicity
# we stop after first fold. You can try 5 folds by yourself by removing
# the break keyword at the end of cell and run it again
skf = StratifiedKFold()

for t_idx, v_idx in skf.split(train_data, train_labels):
    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])
    train = train_data[t_idx]
    val = train_data[v_idx]
    t_labels = train_labels[t_idx]
    v_labels = train_labels[v_idx]
    history = model.fit(train, t_labels, validation_data=(val, v_labels), epochs=2, verbose=0)
    print(history.history['val_acc'][-1])
    break # remove this line if you want to run all folds
# model compiling and fitting
model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10)
print('Convolutional Neural Network:', history.history['acc'][-1])
skf = StratifiedKFold()
svc = SVC()
for t_idx, v_idx in skf.split(train_data, train_labels):
    train = train_data[t_idx]
    val = train_data[v_idx]
    t_labels = train_labels[t_idx]
    v_labels = train_labels[v_idx]
    svc.fit(train, t_labels)
    p_labels = svc.predict(val)
    print(accuracy_score(p_labels, v_labels))
    break
svc = SVC()
svc.fit(train_data, train_labels)
svc_pred_labels = svc.predict(train_data)
svc_acc = accuracy_score(svc_pred_labels, train_labels)
print('SVM:', svc_acc)
skf = StratifiedKFold()
knn = KNeighborsClassifier()
for t_idx, v_idx in skf.split(train_data, train_labels):
    train = train_data[t_idx]
    val = train_data[v_idx]
    t_labels = train_labels[t_idx]
    v_labels = train_labels[v_idx]
    knn.fit(train, t_labels)
    p_labels = knn.predict(val)
    print(accuracy_score(p_labels, v_labels))
    break
knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)
knn_pred_labels = knn.predict(train_data)
knn_acc = accuracy_score(knn_pred_labels, train_labels)
print('KNN:', knn_acc)
skf = StratifiedKFold()
rfc = RandomForestClassifier(100)
for t_idx, v_idx in skf.split(train_data, train_labels):
    train = train_data[t_idx]
    val = train_data[v_idx]
    t_labels = train_labels[t_idx]
    v_labels = train_labels[v_idx]
    rfc.fit(train, t_labels)
    p_labels = rfc.predict(val)
    print(accuracy_score(p_labels, v_labels))
    break
rfc = RandomForestClassifier(100)
rfc.fit(train_data, train_labels)
rfc_pred_labels = rfc.predict(train_data)
rfc_acc = accuracy_score(rfc_pred_labels, train_labels)
print('Random forest:', rfc_acc)
skf = StratifiedKFold()
gbc = xgb.XGBClassifier(n_estimators=100, objective='reg:logistic')
for t_idx, v_idx in skf.split(train_data, train_labels):
    train = train_data[t_idx]
    val = train_data[v_idx]
    t_labels = train_labels[t_idx]
    v_labels = train_labels[v_idx]
    gbc.fit(train, t_labels)
    p_labels = gbc.predict(val)
    print(accuracy_score(p_labels, v_labels))
    break
gbc = xgb.XGBClassifier(n_estimators=100, objective='reg:logistic')
gbc.fit(train_data, train_labels)
gbc_pred_labels = gbc.predict(train_data)
gbc_acc = accuracy_score(gbc_pred_labels, train_labels)
print('Gradient Boosted Trees:', gbc_acc)
pred_labels = model.predict(test_data)
pred_labels = np.argmax(pred_labels, axis=1)
submission = pd.DataFrame({'ImageId': range(1, test_data.shape[0]+1), 'Label': pred_labels})
submission.to_csv(submission_file, index=False)