# Directive pour afficher les graphiques dans Jupyter

%matplotlib inline
# Pandas : librairie de manipulation de données

# NumPy : librairie de calcul scientifique

# MatPlotLib : librairie de visualisation et graphiques

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn import metrics

from sklearn import preprocessing

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score



from sklearn.preprocessing import StandardScaler, MinMaxScaler



from sklearn.model_selection import train_test_split



from IPython.core.display import HTML # permet d'afficher du code html dans jupyter
from sklearn.model_selection import learning_curve

def plot_learning_curve(est, X_train, y_train) :

    train_sizes, train_scores, test_scores = learning_curve(estimator=est, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10),

                                                        cv=5,

                                                        n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8,10))

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')

    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')

    plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')

    plt.grid(b='on')

    plt.xlabel('Number of training samples')

    plt.ylabel('Accuracy')

    plt.legend(loc='lower right')

    plt.ylim([0.6, 1.0])

    plt.show()
def plot_roc_curve(est,X_test,y_test) :

    probas = est.predict_proba(X_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(8,8))

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe

    plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe

    plt.xlim([-0.05,1.2])

    plt.ylim([-0.05,1.2])

    plt.ylabel('Taux de vrais positifs')

    plt.xlabel('Taux de faux positifs')

    plt.show
dftrain = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
dftrain.head()
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",

          "Sneaker","Bag","Ankle boot"]
print(labels[dftrain.label[0]])
X_train=dftrain.drop(['label'],axis=1)

y_train=dftrain['label']
images=np.array(X_train).reshape(len(X_train),28,28)
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(images[i], cmap="gray_r")

    plt.title(labels[y_train[i]])
dftest = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
X_test=dftest.drop(['label'], axis=1)

y_test=dftest['label']
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
print(confusion_matrix(y_test, y_rf))

print(classification_report(y_test, y_rf))
import xgboost as XGB

xgb  = XGB.XGBClassifier(nthread=5)

xgb.fit(X_train, y_train)

y_xgb = xgb.predict(X_test)
print(confusion_matrix(y_test, y_xgb))
from keras.models import Sequential

from keras.layers import Dense

from keras.utils.np_utils import to_categorical

y_cat = to_categorical(y_train)

num_classes = y_cat.shape[1]

X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)

model = Sequential()

model.add(Dense(200, activation='relu'))

model.add(Dense(60, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)
model.evaluate(X_test,y_test)
def plot_scores(train) :

    accuracy = train.history['accuracy']

    val_accuracy = train.history['val_accuracy']

    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')

    plt.plot(epochs, val_accuracy, 'r', label='Score validation')

    plt.title('Scores')

    plt.legend()

    plt.show()

plot_scores(train)
model2 = Sequential()

model2.add(Dense(40, activation='relu'))

model2.add(Dense(30, activation='relu'))

model2.add(Dense(20, activation='relu'))

model2.add(Dense(num_classes, activation='softmax'))

model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train2 = model2.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)
plot_scores(train2)
print(X_train.shape)
model3 = Sequential()

model3.add(Dense(200, input_dim=784, activation='relu'))

model3.add(Dense(num_classes, activation='softmax'))

model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train3 = model3.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)
plot_scores(train3)
model4 = Sequential()

model4.add(Dense(num_classes,input_dim=784, activation='softmax'))

model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train4 = model4.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)
plot_scores(train4)
model5 = Sequential()

model5.add(Dense(400,input_dim=784, activation='relu'))

model5.add(Dense(150, input_dim=400,activation='relu'))

model5.add(Dense(num_classes, input_dim=150, activation='softmax'))

model5.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train5 = model5.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)
plot_scores(train5)