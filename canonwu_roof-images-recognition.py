# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import glob

import scipy.io as sio



from sklearn.cluster import KMeans

from sklearn.svm import LinearSVC, SVC

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import normalize, Normalizer, scale



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# img_folder = '../input/roof-images/roof_images/roof_images/'

train_ids_labels = pd.read_csv('../input/id-train-csv/id_train.csv')

train_idx_raw = train_ids_labels.Id.tolist()



train_img_names_raw = [str(img_idx) + '.jpg' for img_idx in train_idx_raw]

y_raw = train_ids_labels.label.tolist()
name_nb_features = pd.read_csv('../input/surf-features/features/nb_features.csv')

train_img_names = name_nb_features.train_img_names.values.tolist()

nb_features = name_nb_features.nb_features.values.tolist()



train_idx = list(map(lambda name: name.split('.')[0], train_img_names))
y = []



for name, cat in zip(train_img_names_raw, y_raw):

    if name in train_img_names:

        y.append(cat)
loaded_train_descs = np.genfromtxt(fname='../input/surf-features/features/SURF_features.csv',

                                   delimiter=',')

# Charger le fichier .mat avec l'aide de la librairie scipy.io

extra_features = sio.loadmat('../input/surf-features/features/features.mat')['F']

extra_features_df = pd.DataFrame(extra_features[0])

# Extraire l'objet(string) depuis le numpy.ndarray pour la colonne des noms des images

extra_features_df.name = [name[0] for name in extra_features_df.name]

# Filtrer les exemples étiquetés

extra_train_features_df = extra_features_df[extra_features_df.name.isin(train_idx)]

extra_train_features_df = extra_features_df[extra_features_df.name.isin(train_idx)]



labHisto = extra_train_features_df.LabHisto.values.tolist()

labHisto = np.array(list(map(lambda x: np.reshape(x, newshape=(4*7*8)), labHisto)))



haralick = extra_train_features_df.Haralick.values.tolist()

haralick = np.array(list(map(lambda x: np.reshape(x, newshape=(14)), haralick)))

# haralick = normalize(X=haralick)



segHisto = extra_train_features_df.segHisto.values.tolist()

segHisto = np.array(list(map(lambda x: np.reshape(x, newshape=(24)), segHisto)))



domColor = extra_train_features_df.domColor.values.tolist()

domColor = np.array(list(map(lambda x: np.reshape(x, newshape=(3)), domColor)))



# Calculer la différence entre le width et le height pour chaque image et puis le reshaper pour pouvoir

# être enchaîne avec le corpus visuel

diff_wh = np.array([((float(width[0, 0]) - float(height[0, 0]))/max(width[0, 0], height[0, 0])) for\

                    (width, height) in zip(extra_train_features_df.width, extra_train_features_df.height)])

diff_wh = np.array(list(map(lambda x: np.reshape(x, newshape=(1)), diff_wh)))
accum = 0

all_train_descs = []

for k in nb_features:

    all_train_descs.append(np.array(loaded_train_descs[accum:accum + k]))

    accum = accum + k
def cluster_features(img_descs, cluster):

    n_clusters = cluster.n_clusters

    print('Starting of preprocessing!')

    # Ici la variable img_descs est une liste contenant des descripteurs de toutes les images. Chaque élément de la liste

    # est un objet de type numpy.ndarray avec un shape de(?, 64). Par exemple, pour l'image "-3935637.jpg", elle a

    # totalement 284 descripteurs, donc le shape pour elle est de (284, 64). Alors pour pouvoir appliquer l'algorithme

    # de clusterisation, il faut tout d'abord séparer ces descripteurs, comme ça chaque descripteur sera traité comme un

    # sample par le cluster.

    all_train_descs = [desc for desc_list in img_descs for desc in desc_list]

    all_train_descs = np.array(all_train_descs)

    

    print('Starting of training!')

    # Calculer les k=n_clusters centroïdes

    cluster.fit(all_train_descs)

    

    print('Training ends!')

    # Predire le cluster le plus proche auquel chaque sample appartient. Chaque élément de la liste est un objet

    # de type numpy.ndarray 1d de shape(?,). Par exemple, pour l'image "-3935637.jpg", le shape est de (284,). La valeur

    # dans le numpy.ndarray représente le cluster où ce descripteur appartient

    img_clustered_words = [cluster.predict(raw_words) for raw_words in img_descs]

    # Calculer l'occurrence de chaque cluster pour chaque image avec np.bincount(). Pour démonstration,

    # np.bincount(np.array([0, 1, 2, 1, 3, 2]), minlength=4) va nous retourner array([1, 2, 2, 1])

    img_bow_hist = np.array([np.bincount(clustered_words, minlength=n_clusters) for\

                             clustered_words in img_clustered_words])

    

    return img_bow_hist, cluster
# Créer un objet de cluster KMeans. Après quelques essaies de nombre de cluster entre 10 et 200,

# il semble que 120 nous donne la meilleur performance. Le paramètre n_init est le nombre

# de fois que l'algorithme KMeans va exécuter avec différentes centroïdes initiales. On spécifie

# ainsi le paramètre random_state pour que le résultat puisse être reproduit. Pour les autres paramètre,

# on les laisse par défaut

km = KMeans(n_clusters=120, random_state=7, n_init=7)

X, km = cluster_features(all_train_descs, km)
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MaxAbsScaler, QuantileTransformer



# diff_wh2 = np.array(list(map(lambda x: x*X.max(), diff_wh)))

X_normed = normalize(X)

X_normed = QuantileTransformer().fit_transform(X)

X_final = np.concatenate((X, haralick), axis=1)

X_final = scale(X, axis=1)

# X_final = normalize(X_final)

normalizer = Normalizer()

# X_final = normalizer.transform(X_final)

# X_final = StandardScaler().fit_transform(X_final)



# Séparer le jeu de données en deux parties: données d'apprentissage et données de test,

# pour les features et les catégories respectivement. Le paramètre test_size spécifie le

# pourcentage des données de test (ici donc 25%). Le random_state, ainsi est pour la 

# reproductibilité des résultats

X_train, X_val, y_train, y_val = train_test_split(X_final, y, test_size=0.25, random_state=7)
class Bag_of_visual_words():

    

    def cluster_features(self, img_descs, cluster):

        n_clusters = cluster.n_clusters

        print('Starting of preprocessing!')

        # train_descs = [img_descs[i] for i in train_index]

        # train_descs = train_test_split(train_descs, test_size=0.05, random_state=77)[1]

        all_train_descs = [desc for desc_list in img_descs for desc in desc_list]

        all_train_descs = np.array(all_train_descs)



        if all_train_descs.shape[1] != 64:

            raise ValueError('Expected SURF descriptors to have 64 features, got', all_train_descs.shape[1])



        print('Starting of training!')

        cluster.fit(all_train_descs)

        print('Training ends!')

        img_clustered_words = [cluster.predict(raw_words) for raw_words in img_descs]

        img_bow_hist = np.array([np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])



        return img_bow_hist, cluster



    def classify_and_evaluate(self, X_train, y_train, X_val, y_val, estimator):

        estimator.fit(X_train, y_train)

        try:

            print('Best estimator: ', estimator.best_estimator_)

            print('Train score: ', estimator.score(X_train, y_train))

            print('Test score: ', estimator.score(X_val, y_val))

        except:

            preds = estimator.predict(X_val)

            print(classification_report(y_true=y_val, y_pred=preds))

            print('Test accuracy: ', sum(preds==y_val)/len(preds))

            print(confusion_matrix(y_pred=preds, y_true=y_val))

        
bovw = Bag_of_visual_words()

X1, km = bovw.cluster_features(cluster=km, img_descs=all_train_descs)

bovw.classify_and_evaluate(X_train, y_train, X_val, y_val, grid_svc)
def classify_and_evaluate(X_train, y_train, X_val, y_val, estimator):

    estimator.fit(X_train, y_train)

    # Si le estimator est un objet de GridSearchCV, alors on affiche les informations ci-dessous

    try:

        print('Best estimator: ', estimator.best_estimator_)

        print('Train score: ', estimator.score(X_train, y_train))

        print('Test score: ', estimator.score(X_val, y_val))

    # Sinon on affiche la précision et la matrice de confusion

    except:

        preds = estimator.predict(X_val)

        print(classification_report(y_true=y_val, y_pred=preds))

        print('Test accuracy: ', sum(preds==y_val)/len(preds))

        print(confusion_matrix(y_pred=preds, y_true=y_val))
clf = MultinomialNB()

classify_and_evaluate(X_train, y_train, X_val, y_val, clf)
# Grille des paramètres

c_vals = [3.5, 3.9, 4.5, 4.9, 5.5]

gamma_vals = [0.005, 0.006, 0.007, 0.008, 0.009]

param_grid = {'C': c_vals, 'gamma': gamma_vals}



grid_svc = GridSearchCV(estimator=SVC(random_state=7), param_grid=param_grid, n_jobs=17)



classify_and_evaluate(X_train, y_train, X_val, y_val, grid_svc)
clf = SVC(C=3.9, gamma=0.006, random_state=7)

classify_and_evaluate(X_train, y_train, X_val, y_val, clf)
from sklearn.ensemble import RandomForestClassifier



param_grid = {'bootstrap': [True], 'max_depth': [20, 30, 40, 50]}

rfc = RandomForestClassifier(n_estimators=300, random_state=77)

grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)

classify_and_evaluate(X_train, y_train, X_val, y_val, grid_rfc)
eta = [0.37]

n_estimators = [100]

param_grid = {'n_estimators': n_estimators, 

              'learning_rate': eta, 

              'base_estimator': [clf]}

grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid)

# ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.43)

classify_and_evaluate(X_train, y_train, X_val, y_val, grid_ada)



# 0.43