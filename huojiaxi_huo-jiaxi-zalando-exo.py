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
df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

df_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
df_test.head()
df.head()
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",

          "Sneaker","Bag","Ankle boot"]
print(labels[df.label[0]])

plt.imshow(df.values[0][1:].reshape(28,28), cmap="gray_r")
plt.figure(figsize=(10,20))

for i in range(50) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(df.values[i][1:].reshape(28,28), cmap="gray_r")

    plt.title(labels[df.label[i]])
y_train = df['label']
X_train = df.drop(['label'], axis=1)

scaler = MinMaxScaler()

X_train=scaler.fit_transform(X_train) # on normalise des données
y_test=df_test['label']

X_test=df_test.drop(['label'],axis=1)

X_test=scaler.fit_transform(X_test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # On sélectionne des données pour tester et entrainer, mais on a le set train et set test dans le Data donnée
X1 = np.array(X_train)
print(X1[0])
image = X1[0].reshape(28,28) # convertir le tableau à 28*28

print(image)
plt.imshow(image)
plt.axis('off')

plt.imshow(image, cmap="gray_r")
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
# plot_learning_curve(rf, X, y)
print(classification_report(y_test, y_rf))
cm = confusion_matrix(y_test, y_rf)

print(cm)
for i in range(0,10):

    print(y_rf[i])

y_test.head(10)
plt.figure(figsize=(10,20))

for i in range(30):

    plt.subplot(6, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    X1_test = np.array(X_test)

    image = X1_test[i].reshape(28,28)

    plt.axis('off')

    plt.imshow(image, cmap="gray_r")

    plt.title(labels[y_rf[i]])
import xgboost as XGB

xgb  = XGB.XGBClassifier()

xgb.fit(X_train, y_train)

y_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_rf))
cm = confusion_matrix(y_test, y_rf)

print(cm)
for i in range(0,10):

    print(y_rf[i])
y_test.head(10)
plt.figure(figsize=(10,20))

for i in range(10):

    plt.subplot(2, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    X1_test = np.array(X_test)

    image = X1_test[i].reshape(28,28)

    plt.axis('off')

    plt.imshow(image, cmap="gray_r")

    plt.title(labels[y_rf[i]])
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(200,60)) # couche

mlp.fit(X_train,y_train)

y_mlp = mlp.predict(X_test)
mlp_score = accuracy_score(y_test, y_mlp)

print(mlp_score)
from keras.utils.np_utils import to_categorical
print(y_train[0])

y_cat_train = to_categorical(y_train)

print(y_cat_train[0])



print(y_test[0])

y_cat_test = to_categorical(y_test)

print(y_cat_test[0])
num_classes = y_cat_train.shape[1]

print(num_classes)
X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(200, activation='relu')) # premiere couche

model.add(Dense(60, activation='relu')) # deuxieme couche

model.add(Dense(num_classes, activation='softmax')) # troisieme couche
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train , y_cat_train , validation_data=(X_test,y_cat_test), epochs=30, verbose=1)
model.evaluate(X_test,y_cat_test)
print(train.history['accuracy'])
print(train.history['val_accuracy'])
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