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
df.head()
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",

          "Sneaker","Bag","Ankle boot"]
print(labels[df.label[0]])
#On crée la cible y (colonne 'label')

y = df['label']

#On créeles caractéristiques X

X = df.drop(['label'], axis=1)
#On sépare les ensembles d'apprentissage et de test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# transformer le dataframe X en un tableau pour visualiser les images

X1 = np.array(X)
#pour toutes les lignes

n_samples = len(df.index)

#convertir cette ligne de 784 éléments en une matrice 28x28

images = X1.reshape(n_samples,28,28)



plt.figure(figsize=(10,20))

for i in range(0,49) : #pour 50 éléments

    plt.subplot(10,5,i+1) #Grille de 10 par 5 

    plt.axis('off') #Ne pas afficher les axes

    plt.imshow(images[i], cmap="gray_r") #Afficahge en nuance de gris

    plt.title(y[i]) #Cible d'indice o
#Importation de la méthode

from sklearn import ensemble

#Entrainement

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

#Prédiction

y_rf = rf.predict(X_test)

#Calcul du score

rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
#Matrice de confusion 

cm = confusion_matrix(y_test, y_rf)

print(cm)

#Classification report

print('\n\n',classification_report(y_test, y_rf))
# Sous Jupyter, si xgboost n'est pas déjà installé

!pip install xgboost
#Importation de la méthode

import xgboost as XGB

#Entrainement

xgb  = XGB.XGBClassifier()

xgb.fit(X_train, y_train)

#Prédiction

y_xgb = xgb.predict(X_test)

#Calcul du score

rf_score = accuracy_score(y_test, y_xgb)

print(rf_score)
#Matrice de confusion 

cm = confusion_matrix(y_test, y_xgb)

print(cm)

#Classification report

print(classification_report(y_test, y_xgb))
#Importation de la méthode de régression logistique

from sklearn.linear_model import LogisticRegression

#Entrainement

lr = LogisticRegression()

lr.fit(X_train,y_train)

#Prédiction

y_lr = lr.predict(X_test)

#Calcul du score

rf_score = accuracy_score(y_test, y_lr)

print(rf_score)
#Matrice de confusion 

cm = confusion_matrix(y_test, y_lr)

print(cm)

#Classification report

print(classification_report(y_test, y_lr))
#Importation de la méthode de régression logistique

from sklearn import svm

#Entrainement

lr = svm.SVC()

lr.fit(X_train,y_train)

#Prédiction

y_lr = lr.predict(X_test)

#Calcul du score

rf_score = accuracy_score(y_test, y_lr)

print(rf_score)
#Matrice de confusion 

cm = confusion_matrix(y_test, y_lr)

print(cm)

#Classification report

print(classification_report(y_test, y_lr))
#Re-Importation du dataset pour éviter de faire un "Run All" de plusieurs dizaines de minutes.

df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
df.shape
df.label.value_counts()
#On crée la cible y (colonne 'label')

y = df['label']

#On créeles caractéristiques X

X = df.drop(['label'], axis=1)
X = X/255
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Importation de la méthode

from sklearn.neural_network import MLPClassifier

# Entrainement avec :

#    1ère couche : 200 neuronnes

#    2ème couche : 60  neuronnes

mlp = MLPClassifier(hidden_layer_sizes=(200,60))

mlp.fit(X_train,y_train)

# Prédiction

y_mlp = mlp.predict(X_test)
mlp_score = accuracy_score(y_test, y_mlp)

print(mlp_score)
pd.crosstab(y_test, y_mlp, rownames=['Reel'], colnames=['Prediction'], margins=True)
# Importation de la méthode

from sklearn.neural_network import MLPClassifier

# Entrainement avec :

#    1ère couche : 400 neuronnes

#    2ème couche : 120  neuronnes

#    3ème couche : 40  neuronnes

mlp = MLPClassifier(hidden_layer_sizes=(400,120,40))

mlp.fit(X_train,y_train)

# Prédiction

y_mlp = mlp.predict(X_test)
mlp_score = accuracy_score(y_test, y_mlp)

print(mlp_score)

pd.crosstab(y_test, y_mlp, rownames=['Reel'], colnames=['Prediction'], margins=True)
from keras.datasets import mnist



from keras.models import Sequential, load_model



from keras.layers import Dense, Dropout, Flatten



from keras.layers.convolutional import Conv2D, MaxPooling2D



from keras.utils.np_utils import to_categorical
print("Avant:",y[0])

y_cat = to_categorical(y)

print("Après: ",y_cat[0])
# On affecte le nombre de classe à la variable num_classes qui pourra être ré-utilisé plus tard

num_classes = y_cat.shape[1]

print("Nombre de classe(s): ",num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=1)
# Conversion des ensembles en tabeaux :

X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
model = Sequential()

model.add(Dense(200, activation='relu'))

model.add(Dense(60, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
# loss: la distance entre la prédiction et le résultat attendu

# optimizer: La descente de gradian, adam est assez utilisé et permet d'accélérer la descente de gradian

# metrics : on veut la précision

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Entrainement

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
# On affiche le graphique

plot_scores(train)