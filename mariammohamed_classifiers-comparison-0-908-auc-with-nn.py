import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import Callback
import keras.backend as k
import tensorflow as tf
data = pd.read_csv('../input/creditcard.csv')
data.head()
data.shape
data.Class.value_counts()
492 / 284315
plt.plot(list(range(1, data.shape[0]+1)), data.Class, 'r+')
plt.show()
np.max(data.Amount), np.min(data.Amount) 
colors = {1: 'red', 0: 'green'}
fraud = data[data.Class == 1]
not_fraud = data[data.Class == 0]
fig, axes = plt.subplots(1, 2)
axes[0].scatter(list(range(1, fraud.shape[0]+1)), fraud.Amount, color='red')
axes[1].scatter(list(range(1, not_fraud.shape[0]+1)), not_fraud.Amount, color='green')
plt.show()
X = data.loc[:, data.columns.tolist()[1:30]]
X = X.as_matrix()
Y = data.loc[:, 'Class']
Y = Y.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
np.bincount(y_train), np.bincount(y_test) 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def train_model(model):
    model.fit(X_train, y_train)
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def predict_model(model):
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(metrics.auc(fpr, tpr))
    print(metrics.classification_report(y_test, y_pred))
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')
mlc = MLPClassifier()
train_model(mlc)
predict_model(mlc)
mlc_2 = MLPClassifier(hidden_layer_sizes=200)
train_model(mlc_2)
predict_model(mlc_2)
mlc_3 = MLPClassifier(hidden_layer_sizes=50)
train_model(mlc_3)
predict_model(mlc_3)
knn = KNeighborsClassifier()
train_model(knn)
predict_model(knn)
knn_3 = KNeighborsClassifier(n_neighbors=3, n_jobs=3)
train_model(knn_3)
predict_model(knn_3)
knn_2 = KNeighborsClassifier(n_neighbors=2, n_jobs=3)
train_model(knn_2)
predict_model(knn_2)
svc = SVC()
train_model(svc)
predict_model(svc)
dtc = DecisionTreeClassifier()
train_model(dtc)
predict_model(dtc)
rfc = RandomForestClassifier()
train_model(rfc)
predict_model(rfc)
rfc_20 = RandomForestClassifier(max_depth=20)
train_model(rfc_20)
predict_model(rfc_20)
rfc_30 = RandomForestClassifier(max_depth=30)
train_model(rfc_30)
predict_model(rfc_30)
abc = AdaBoostClassifier()
train_model(abc)
predict_model(abc)
gnb = GaussianNB()
train_model(gnb)
predict_model(gnb)
lr = LogisticRegression()
train_model(lr)
predict_model(lr)
lr_2 = LogisticRegression(C=0.1)
train_model(lr_2)
predict_model(lr_2)
lr_3 = LogisticRegression(solver='newton-cg')
train_model(lr_3)
predict_model(lr_3)
model = Sequential()
model.add(Dense(256, activation='tanh', input_dim=29))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)
y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')
fraud = X_train[y_train==1]
y_fraud = y_train[y_train==1]
X_train.shape
for _ in range(5):
    copy_fraud = np.copy(fraud)
    y_fraud_copy = np.copy(y_fraud)
    X_train = np.concatenate((X_train, copy_fraud))
    y_train = np.concatenate((y_train, y_fraud_copy))
p = np.random.permutation(X_train.shape[0])
X_train = X_train[p]
y_train = y_train[p]
mlp = MLPClassifier()
train_model(mlp)
predict_model(mlp)
dtc = DecisionTreeClassifier()
train_model(dtc)
predict_model(dtc)
model = Sequential()
model.add(Dense(256, activation='tanh', input_dim=29))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)
y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')
model.fit(X_train, y_train, epochs=5)
y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')
model.fit(X_train, y_train, epochs=5)
y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')
model.fit(X_train, y_train, epochs=15)
y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')
model.fit(X_train, y_train, epochs=10)
y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')
model.fit(X_train, y_train, epochs=20)
y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')