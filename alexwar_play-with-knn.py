import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
from  scipy import stats
from matplotlib import cm
%matplotlib inline
def rand_gauss(n=100, mu=[1, 1], sigma=[0.1, 0.1]):
    """ Sample n points from a Gaussian variable with center mu, 
    and std deviation sigma
    """
    d = len(mu)
    res = np.random.randn(n, d)
    return np.array(res * sigma + mu)


def rand_bi_gauss(n1=100, n2=100, mu1=[1, 1], mu2=[-1, -1], sigma1=[0.1, 0.1],
                  sigma2=[0.1, 0.1]):
    """ Sample n1 and n2 points from two Gaussian variables centered in mu1,
    mu2, with std deviation sigma1, sigma2
    """
    ex1 = rand_gauss(n1, mu1, sigma1)
    ex2 = rand_gauss(n2, mu2, sigma2)
    res = np.vstack([np.hstack([ex1, 1. * np.ones((n1, 1))]),
                     np.hstack([ex2, 2. * np.ones((n2, 1))])])
    ind = np.arange(res.shape[0])
    np.random.shuffle(ind)
    return np.array(res[ind, :-1]), np.array(res[ind, -1])


def rand_tri_gauss(n1=100, n2=100, n3=100, mu1=[1, 1],
                   mu2=[-1, -1], mu3=[1, -1], sigma1=[0.1, 0.1],
                   sigma2=[0.1, 0.1], sigma3=[0.1, 0.1]):
    """ Sample n1, n2 and n3 points from three Gaussian variables centered in mu1,
    mu2 and mu3 with std deviation sigma1, sigma2 and sigma3
    """
    ex1 = rand_gauss(n1, mu1, sigma1)
    ex2 = rand_gauss(n2, mu2, sigma2)
    ex3 = rand_gauss(n3, mu3, sigma3)
    res = np.vstack([np.hstack([ex1, 1. * np.ones((n1, 1))]),
                     np.hstack([ex2, 2. * np.ones((n2, 1))]),
                     np.hstack([ex3, 3. * np.ones((n3, 1))])])
    ind = np.arange(res.shape[0])
    np.random.shuffle(ind)
    return np.array(res[ind, :-1]), np.array(res[ind, -1])


def rand_clown(n1=100, n2=100, sigma1=1, sigma2=2):
    """ Sample a dataset clown  with
    n1 points and noise std deviation sigma1 for the first class, and
    n2 points and noise std deviation sigma2 for the second one
    """
    x0 = np.random.randn(n1)
    x1 = x0 * x0 + sigma1 * np.random.randn(n1)
    x2 = np.vstack([sigma2 * np.random.randn(n2),
                    sigma2 * np.random.randn(n2) + 2.])
    res = np.hstack([np.vstack([[x0, x1], 1. * np.ones([1, n1])]),
                     np.vstack([x2, 2. * np.ones([1, n2])])]).T
    ind = np.arange(res.shape[0])
    np.random.shuffle(ind)
    return np.array(res[ind, :-1]), np.array(res[ind, -1])


def rand_checkers(n1=100, n2=100, sigma=0.1):
    """ Sample n1 and n2 (multiples of 8) points from a noisy checker
    """
    nbp = int(np.floor(n1 / 8))
    nbn = int(np.floor(n2 / 8))
    xapp = np.reshape(np.random.rand((nbp + nbn) * 16), [(nbp + nbn) * 8, 2])
    yapp = np.ones((nbp + nbn) * 8)
    idx = 0
    for i in range(-2, 2):
        for j in range(-2, 2):
            if (((i + j) % 2) == 0):
                nb = nbp
            else:
                nb = nbn
                yapp[idx:(idx + nb)] = [(i + j) % 3 + 1] * nb

            xapp[idx:(idx + nb), 0] = np.random.rand(nb)
            xapp[idx:(idx + nb), 0] += i + sigma * np.random.randn(nb)
            xapp[idx:(idx + nb), 1] = np.random.rand(nb)
            xapp[idx:(idx + nb), 1] += j + sigma * np.random.randn(nb)
            idx += nb

    ind = np.arange((nbp + nbn) * 8)
    np.random.shuffle(ind)
    res = np.hstack([xapp, yapp[:, np.newaxis]])
    return np.array(res[ind, :-1]), np.array(res[ind, -1])
symlist = ['o', 's', 'D', '+', 'x', '*', 'p', 'v', '-', '^']
collist = ['blue', 'grey', 'red', 'purple', 'orange', 'salmon',
           'black', 'fuchsia']


def plot_2d(data, y=None, w=None, alpha_choice=1):
    """ Plot in 2D the dataset data, colors and symbols according to the
    class given by the vector y (if given); the separating hyperplan w can
    also be displayed if asked"""
    if y is None:
        labs = [""]
        idxbyclass = [range(data.shape[0])]
    else:
        labs = np.unique(y)
        idxbyclass = [np.where(y == labs[i])[0] for i in range(len(labs))]

    for i in range(len(labs)):
        plt.plot(data[idxbyclass[i], 0], data[idxbyclass[i], 1], '+',
                 color=collist[i % len(collist)], ls='None',
                 marker=symlist[i % len(symlist)])
    plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])
    plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
    mx = np.min(data[:, 0])
    maxx = np.max(data[:, 0])
    if w is not None:
        plt.plot([mx, maxx], [mx * -w[1] / w[2] - w[0] / w[2],
                              maxx * -w[1] / w[2] - w[0] / w[2]],
                 "g", alpha=alpha_choice)
def frontiere(f, data, step=50, cmap_choice=cm.coolwarm, tiny=False):
    """ En : Boundary for decision function f """
    xmin, xmax = data[:, 0].min() - 1., data[:, 0].max() + 1.
    ymin, ymax = data[:, 1].min() - 1., data[:, 1].max() + 1.
    xx, yy = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step),
                         np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
    z = f(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.imshow(z, origin='lower', interpolation="nearest",
               extent=[xmin, xmax, ymin, ymax], cmap=cmap_choice)
    if tiny:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.colorbar()
from sklearn.neighbors import KNeighborsClassifier

class ErrorCurve(object):
    def __init__(self, k_range=None, weights='uniform'):
        if k_range is None:
            k_range = [i + 1 for i in range(5)]
        self.k_range = k_range
        self.weights = weights

    def fit_curve(self, X, y, Xtest, ytest):
        def error_func(k):
            knn = KNeighborsClassifier(n_neighbors=k,
                                                 weights=self.weights)
            knn.fit(X, y)
            error = np.mean(knn.predict(Xtest) != ytest)
            return error

        errors = [error_func(el) for el in self.k_range]
        self.errors = np.array(errors)
        self.y = y

    def plot(self, marker='o', maketitle=True, **kwargs):
        plt.plot(self.k_range, self.errors, marker=marker, **kwargs)
        plt.xlabel("K")
        plt.ylabel("error rate")
        if maketitle:
            plt.title("number of training points : %d" % len(self.y))

from sklearn.model_selection import ShuffleSplit 

class LOOCurve(object):
    """Leave-One-Out (LOO) curve"""
    def __init__(self, k_range=None, weights='uniform'):
        if k_range is None:
            k_range = [i + 1 for i in range(5)]
        self.k_range = k_range
        self.weights = weights

    def fit_curve(self, X, y, n_iter=200, random_state=1):
        def score_func(k):
            n_samples = len(X)
            # Selon la verson de scikit-learn : shuffleSplit prend en argument
            # 'niter ' ou niterations'. De plus, largument test_size peut ne
            # pas etre reconnu. Il est recommande de consulter
            # help(cross_validation.ShuffleSplit) pour connaitre la liste
            # des arguments reconnus par votre version de sickitlearn.
            loo = ShuffleSplit(n_samples, n_iter,
                                                test_size=1,
                                                random_state=random_state)
            knn = KNeighborsClassifier(n_neighbors=k,
                                                 weights=self.weights)

            scores = cross_validation.cross_val_score(estimator=knn,
                                                      X=X, y=y,
                                                      cv=loo)
            return np.mean(scores)

        scores = [score_func(el) for el in self.k_range]
        self.cv_scores = np.array(scores)
        self.y = y

    def plot(self, marker='o', maketitle=True, **kwargs):
        plt.plot(self.k_range, self.cv_scores, marker=marker, **kwargs)
        plt.xlabel("K")
        plt.ylabel("Leave One Out Score (1-error rate)")
        if maketitle:
            plt.title("number of training points : %d" % (len(self.y) - 1))

np.random.seed(42)
figure=plt.figure()
figure,axes=plt.subplots(figsize=(15,5))
figure.suptitle("Exemples de rand_bi_gauss")
# Affichage du jeu de données 1
plt.subplot(1,3,1)
plt.title("Jeu de données 1 par defaut")
randbigauss=rand_bi_gauss(n1=100, n2=100, mu1=[1, 1], mu2=[-1, -1], sigma1=[0.1, 0.1],sigma2=[0.1, 0.1])
plot_2d(randbigauss[0],randbigauss[1])
# Affichage du jeu de données 2
plt.subplot(1,3,2)
plt.title("Jeu de données 2")
randbigauss=rand_bi_gauss(n1=500, n2=250,mu1=[0, 0],mu2=[-1, -1],sigma1=[1,1],sigma2=[0.4, 0.4])
plot_2d(randbigauss[0],randbigauss[1])
# Affichage du jeu de données 3
plt.subplot(1,3,3)
plt.title("Jeu de données 3")
randbigauss=rand_bi_gauss(n1=100, n2=100,mu1=[0, 0],mu2=[-1, -1],sigma1=[0.4, 0.4],sigma2=[0.4, 0.4])
plot_2d(randbigauss[0],randbigauss[1])
plt.show()
np.random.seed(42)

figure=plt.figure()
figure,axes=plt.subplots(figsize=(15,5))
figure.suptitle("Exemple de rand_tri_gauss")
# Affichage du jeu de données 1
plt.subplot(1,3,1)
plt.title("Jeu de données 1")
randtrigauss=rand_tri_gauss(n1=100, n2=100, n3=100,
                            mu1=[1, 1],mu2=[-1, -1], mu3=[1, -1], 
                            sigma1=[0.1, 0.1],sigma2=[0.1, 0.1],sigma3=[0.1, 0.1])
plot_2d(randtrigauss[0],randtrigauss[1])
# Affichage du jeu de données 2
plt.subplot(1,3,2)
plt.title("Jeu de données 2")
randtrigauss=rand_tri_gauss(n1=100, n2=100, n3=100,
                            mu1=[0, 0],mu2=[-1, -1], mu3=[1, -1], 
                            sigma1=[0.4, 0.4],sigma2=[0.4, 0.4],sigma3=[0.2, 0.4])
plot_2d(randtrigauss[0],randtrigauss[1])


# Affichage du jeu de données 3
plt.subplot(1,3,3)
plt.title("Jeu de données 3")
randtrigauss=rand_tri_gauss(n1=100, n2=100, n3=500,
                            mu1=[4, 2],mu2=[0.5, 0.5], mu3=[-1, -1], 
                            sigma1=[1, 1],sigma2=[0.5, 0.5],sigma3=[0.1, 0.1])
plot_2d(randtrigauss[0],randtrigauss[1])
plt.show()
np.random.seed(42)
figure=plt.figure()
figure,axes=plt.subplots(figsize=(15,5))
figure.suptitle("Exemples de rand_clow")
# Affichage du jeu de données 1
plt.subplot(1,3,1)
plt.title("Jeu de données 1 par defaut")
randclow=rand_clown(n1=100, n2=100, sigma1=1, sigma2=2)
plot_2d(randclow[0],randclow[1])
# Affichage du jeu de données 2
plt.subplot(1,3,2)
plt.title("Jeu de données 2")
randclow=rand_clown(n1=100, n2=250, sigma1=1, sigma2=10)
plot_2d(randclow[0],randclow[1])
# Affichage du jeu de données 3
plt.subplot(1,3,3)
plt.title("Jeu de données 3")
randclow=rand_clown(n1=100, n2=100, sigma1=1, sigma2=6)
plot_2d(randclow[0],randclow[1])
plt.show()
np.random.seed(42)
figure=plt.figure()
figure,axes=plt.subplots(figsize=(15,5))
figure.suptitle("Exemples de rand_checkers")
# Affichage du jeu de données 1
plt.subplot(1,3,1)
plt.title("Jeu de données 1 par defaut")
randcheckers=rand_checkers(n1=100, n2=500, sigma=1)
plot_2d(randcheckers[0],randcheckers[1])
# Affichage du jeu de données 2
plt.subplot(1,3,2)
plt.title("Jeu de données 2")
randcheckers=rand_checkers(n1=100, n2=500, sigma=3)
plot_2d(randcheckers[0],randcheckers[1])
# Affichage du jeu de données 3
plt.subplot(1,3,3)
plt.title("Jeu de données 3")
randcheckers=rand_checkers(n1=250, n2=500, sigma=0)
plot_2d(randcheckers[0],randcheckers[1])
plt.show()
class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        n_samples, n_features = X.shape
        tmp =[]
        y_prediction = []
        # Calcul des distances #--> compute distance
        distance=euclidean_distances(X, self.X_)
        # Trie des distances #-->sort distance
        tri=np.argsort(distance, axis=1)
        # Selection des n premiers voisins #-->select ordered neighbors
        nvoisins=tri[:,0:self.n_neighbors]
        # Récupére les classes des individus mes plus proches #-->get the class of closest neighbors
        classe_vois=self.y_[nvoisins]
        # Recupére la prediction #prediction
        tmp=stats.mode(classe_vois, axis=1)
        y_prediction=np.array([tmp.mode[k,0] for k in range(len(tmp.mode))], dtype=np.float64)
        return y_prediction
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd        
from timeit import timeit  


#to compare
def comparaison(data):
        # train_set, test_set
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.8,random_state=0)

        # k=10 sklearn
        neigh = KNeighborsClassifier(n_neighbors=10)
        neigh.fit(X_train,y_train) 
        predic_KNeighborsClassifier=neigh.predict(X_test)

        # k=10 our classifier
        MY_KNM=KNNClassifier(n_neighbors=10)
        MY_KNM.fit(X_train,y_train)
        predic_my_knm=MY_KNM.predict(X_test)

        # 1er comparaison : le taux de bonnes predictions
        print("Taux de bonnes predictions des deux classes :")
        print("---------------------------------------------")
        print("")
        acc_KNeighborsClassifier=accuracy_score(y_test,predic_KNeighborsClassifier)
        print (u'    Taux de bonnes predictions KNeighborsClassifier : {:.1f}%.'.format(acc_KNeighborsClassifier * 100))
        acc_my_knm=accuracy_score(y_test,predic_my_knm)
        print (u'    Taux de bonnes predictions my_knm: {:.1f}%.'.format(acc_my_knm * 100))
        print("")
        if acc_KNeighborsClassifier==acc_my_knm:
            print("    =>Le taux de bonnes prédictions est identique")
        else:
            print("    =>Le taux de bonnes prédictions est différent") 
        print("")

        # 2éme comparaison : résultats sur les différentes classes
        classes_list = np.unique(y_test).astype(int)
        print("Résultats sur les différentes classes : ")
        print("----------------------------------------")
        print("")

        precision1, recall1, _, _ = precision_recall_fscore_support(y_test,predic_my_knm)
        precision2, recall2, _, _ = precision_recall_fscore_support(y_test,predic_KNeighborsClassifier)

        for classe in range(len(classes_list)):
            print("Pour la classe {} :".format(classes_list[classe])) 
            print("    Precision KNeighborsClassifier est {:.2f}% et recall et {:.2f}% pour la classe {}." \
                  .format(precision2[classe] * 100,recall2[classe] * 100,classes_list[classe]))
            print("    Precision my_knm est {:.2f}% et recall et {:.2f}% pour la classe {}." \
                  .format(precision1[classe] * 100,recall1[classe] * 100,classes_list[classe]))
            if recall1[classe]==recall2[classe] and precision1[classe]==precision2[classe]:
                print("    =>Résultats identiques pour la classe {}".format(classes_list[classe]))
            else:
                print("    =>Résultats différents pour la classe {}".format(classes_list[classe]))
        print("")

        # 3ème comparaison : classification identique
        bonne_prediction=0.0
        for i in range(len(predic_KNeighborsClassifier)):
            if predic_KNeighborsClassifier[i]==predic_my_knm[i]:
                bonne_prediction=bonne_prediction+1
        taux=bonne_prediction/len(predic_KNeighborsClassifier) 
        print("Comparaison des classifications :")
        print("---------------------------------")
        print("")

        print("    Le taux de prédiction identique est de :{:.2f}%".format(taux*100))
        if taux==1:
            print("    =>Les prédictions sont identiques")
        else: 
            print("    =>Les 2 classes ne donnent pas les mêmes prédictions")
        print("")
        
        plt.show()
print("On RAND_TRI_GAUSS dataset")
comparaison(randtrigauss)
print("On RAND_CLOW")
comparaison(randclow)
print("On RAND_CHECKERS")
comparaison(randcheckers)
print("On RAND_BI_GAUSS")
comparaison(randbigauss)
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def affichage(data,typedonnee):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.8,random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=10,metric='euclidean')
    neigh.fit(X_train,y_train) 
    acc_1=accuracy_score(y_test,neigh.predict(X_test))
    acc_2=accuracy_score(y_train,neigh.predict(X_train))
    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("Résultat sur le jeu de données {}.".format(typedonnee))
    plt.subplot(1,2,1)
    plt.title("Apprentissage - taux d'erreurs {:.2f}%".format(1.0-acc_2*100))
    plot_2d(X_train,y_train)
    frontiere(neigh.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=False)
    plt.subplot(1,2,2)
    plt.title("Validation - taux d'erreurs {:.2f}%".format(1.0-acc_1*100))
    plot_2d(X_test,y_test)
    frontiere(neigh.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=False)
    plt.show()
affichage(randtrigauss,"RAND_TRI_GAUSS")
affichage(randclow,"RAND_CLOW")
affichage(randcheckers,"RAND_CHECKERS")

affichage(randbigauss,"RAND_BI_GAUSS")
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def variation_k(data,typedonnee,steps=10):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.8,random_state=0)
    k=1
    i=1
    
    col=3
    ligne=int(len(y_train)/(col*steps))+2
    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(10,15))
    figure.suptitle("Résultat sur le jeu de données {}.".format(typedonnee))
    while k<len(y_train):
        neigh = KNNClassifier(n_neighbors=k)
        neigh.fit(X_train,y_train) 
        t=neigh.predict(X_test)
        acc2=accuracy_score(y_test, t)
        plt.subplot(ligne,col,i)
        plt.title ("K={:.0f} -  {:.0f}%".format(k,acc2*100))
        plot_2d(X_test,y_test)
        frontiere(neigh.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=True)
        i=i+1
        k=k+steps
    if k!=len(y_train):
        neigh = KNNClassifier(n_neighbors=len(y_train))
        neigh.fit(X_train,y_train) 
        t=neigh.predict(X_test)
        acc2=accuracy_score(y_test, t)
        plt.subplot(ligne,col,i)
        plt.title ("K={:.0f} -  {:.0f}%".format(k,acc2*100))
        plot_2d(X_test,y_test)
        frontiere(neigh.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=True)
    plt.show()

variation_k(randtrigauss,"RAND_TRI_GAUSS",steps=50)
variation_k(randclow,"RAND_CLOW",steps=50)
variation_k(randcheckers,"RAND_CHECKERS",steps=100)
variation_k(randbigauss,"RAND_BI_GAUSS",steps=100)
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics.pairwise import euclidean_distances
def weights(dist):
    res=[]
    res=np.array([np.exp(-1*((dist[k]*dist[k])/0.2)) for k in range(len(dist))],dtype=np.float64)
    return res

class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=1,weights=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self
    

    
    def predict(self, X):
        n_samples, n_features = X.shape
        tmp =[]
        y_prediction = []
        # Calcul des distances
        distance=euclidean_distances(X, self.X_)
        #print(distance[:,2])
        #print(distance)
        # Trie des distances
        tri=np.argsort(distance, axis=1)
        
        # Selection des n premiers voisins
        nvoisins=tri[:,0:self.n_neighbors]
        distancekvoisins=[]
        # Récupére les classes des individus mes plus proches
        classe_vois=self.y_[nvoisins]

        if self.weights=='weights':
            for i in range(len(classe_vois)):
                tmp=[]
                for k in range(self.n_neighbors):
                    tmp.append(distance[i][nvoisins[i][k]])
                distancekvoisins.append(tmp)
            dis=np.array([distancekvoisins[i] for i in range(len(classe_vois))])
            poids_vois=weights(dis)
            tmp=weighted_mode(classe_vois,poids_vois,axis=1)
            pred=tmp[0]
            y_prediction=np.array([pred[k][0] for k in range(len(pred))], dtype=np.float64)
        else:
            tmp=stats.mode(classe_vois, axis=1)
            y_prediction=np.array([tmp.mode[k,0] for k in range(len(tmp.mode))], dtype=np.float64)
        return y_prediction

def comparaison2(data):
        # Création des échantillons d'apprentissage et de validation
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.8,random_state=0)

        # Application de la classe KNeighborsClassifier avec k=10
        neigh = KNeighborsClassifier(n_neighbors=10,weights=weights)
        neigh.fit(X_train,y_train) 
        predic_KNeighborsClassifier=neigh.predict(X_test)
        # Application de notre classe avec k=10
        MY_KNM=KNNClassifier(n_neighbors=10,weights='weights')
        MY_KNM.fit(X_train,y_train)
        predic_my_knm=MY_KNM.predict(X_test)

        # comparaison : classification identique
        bonne_prediction=0.0
        for i in range(len(predic_KNeighborsClassifier)):
            if predic_KNeighborsClassifier[i]==predic_my_knm[i]:
                bonne_prediction=bonne_prediction+1
        taux=bonne_prediction/len(predic_KNeighborsClassifier) 
        print("Comparaison des classifications :")
        print("---------------------------------")
        print("")

        print("    Le taux de prédiction identique est de :{:.2f}%".format(taux*100))
        if taux==1:
            print("    =>Les prédictions sont identiques")
        else: 
            print("    =>Les 2 classes ne donnent pas les mêmes prédictions")
        print("")
        
def affichage2(data,typedonnee):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.8,random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=20,metric='euclidean',weights=weights)
    neigh.fit(X_train,y_train) 
    acc_1=accuracy_score(y_train,neigh.predict(X_train))
    
    neigh2 = KNeighborsClassifier(n_neighbors=20,metric='euclidean')
    neigh2.fit(X_train,y_train) 
    acc_2=accuracy_score(y_train,neigh2.predict(X_train))

    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("Résultat sur le jeu de données {}.".format(typedonnee))
    plt.subplot(1,2,1)
    plt.title("Sans Fonction Poids Apprentissage")
    plot_2d(X_train,y_train)
    frontiere(neigh2.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=False)
    plt.subplot(1,2,2)
    plt.title("Avec Fonction Poids Apprentissage")
    plot_2d(X_train,y_train)
    frontiere(neigh.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=False)
    plt.show()
print("Application sur le jeu RAND_TRI_GAUSS")
comparaison2(randtrigauss)
affichage2(randtrigauss,"RAND TRI GAUSS")
print("Application sur le jeu RAND_CLOW")
comparaison2(randclow)
affichage2(randclow,"RAND CLOW")
print("Application sur le jeu RAND_CHECKERS")
comparaison2(randcheckers)
affichage2(randcheckers,"RAND CHECKERS")

print("Application sur le jeu RAND_BI_GAUSS")
comparaison2(randbigauss)
affichage2(randbigauss,"RAND BI GAUSS")
def affichage3(data,typedonnee):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.8,random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=10,metric='euclidean',weights=weights)
    neigh.fit(X_train,y_train) 
    acc_1=accuracy_score(y_test,neigh.predict(X_test))
    acc_2=accuracy_score(y_train,neigh.predict(X_train))
    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("Résultat sur le jeu de données {}.".format(typedonnee))
    plt.subplot(1,2,1)
    plt.title("Apprentissage - taux d'erreurs {:.2f}%".format(1.0-acc_2*100))
    plot_2d(X_train,y_train)
    frontiere(neigh.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=False)
    plt.subplot(1,2,2)
    plt.title("Validation - taux d'erreurs {:.2f}%".format(1.0-acc_1*100))
    plot_2d(X_test,y_test)
    frontiere(neigh.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=False)
    plt.show()

def weights(dist):
    res=[]
    res=np.array([np.exp(-1*((dist[k]*dist[k])/0.002)) for k in range(len(dist))],dtype=np.float64)
    return res

print("")
affichage3(randclow,"RAND_CLOW")
def weights(dist):
    res=[]
    res=np.array([np.exp(-1*((dist[k]*dist[k])/0.5)) for k in range(len(dist))],dtype=np.float64)
    return res
affichage3(randclow,"RAND_CLOW")

def weights(dist):
    res=[]
    res=np.array([np.exp(-1*((dist[k]*dist[k])/1)) for k in range(len(dist))],dtype=np.float64)
    return res
print("")
affichage3(randclow,"RAND_CLOW")
def weights(dist):
    res=[]
    res=np.array([np.exp(-1*((dist[k]*dist[k])/1)) for k in range(len(dist))],dtype=np.float64)
    return res
affichage3(randclow,"RAND_CLOW")
def weights(dist):
    res=[]
    res=np.array([np.exp(-1*((dist[k]*dist[k])/10)) for k in range(len(dist))],dtype=np.float64)
    return res
affichage3(randclow,"RAND_CLOW")
def weights(dist):
    res=[]
    res=np.array([np.exp(-1*((dist[k]*dist[k])/100)) for k in range(len(dist))],dtype=np.float64)
    return res
affichage3(randclow,"RAND_CLOW")

def k1(data):
        # Création des échantillons d'apprentissage et de validation
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.8,random_state=0)

        # Application de notre classe avec k=1
        MY_KNM=KNNClassifier(n_neighbors=1,weights='weights')
        MY_KNM.fit(X_train,y_train)
        predic_my_knm=MY_KNM.predict(X_train)

        # 1er comparaison : le taux de bonnes predictions
        print("Taux d'erreurs :")
        print("----------------")
        print("")
        acc_my_knm=accuracy_score(y_train,predic_my_knm)
        print ("Taux d'erreurs: {:.1f}%.".format(100.0-acc_my_knm * 100.0))
        print("")

print("")
print("Application sur le jeu RAND_TRI_GAUSS")
print("")
k1(randtrigauss)
print("")
print("Application sur le jeu RAND_CLOW")
print("")
k1(randclow)
print("")
print("Application sur le jeu RAND_CHECKERS")
print("")
k1(randcheckers)
print("")
print("Application sur le jeu RAND_BI_GAUSS")
print("")
k1(randbigauss)
def testK_randcheckers():
    taille=[100,500,1000]
    print("RandCheckers")
    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("RandCheckers")
    for i in range(len(taille)):
        donnees=rand_checkers(n1=taille[i], n2=taille[i], sigma=0)
        X_train, X_test, y_train, y_test = train_test_split(donnees[0], donnees[1], train_size=0.8,random_state=0)
        #test=ErrorCurve(k_range=[i + 1 for i in range(len(X_train))])
        test=ErrorCurve(k_range=[i + 1 for i in range(50)])
        test.fit_curve(X=X_train,y=y_train, Xtest=X_test,  ytest=y_test)
        tri=np.min(test.errors)
        print(tri)
        j=1
        while j<len(test.errors) and test.errors[j] != tri:
            j=j+1
        print ("Pour n ={:.0f} k={:.0f}.".format(taille[i],j+1))
        plt.subplot(1,3,i+1)
        test.plot()
    plt.show(figure)
    
testK_randcheckers()

def testK_randclown():
    taille=[100,500,1000]
    print("RandClown")
    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("RandClown")
    for i in range(len(taille)):
        donnees=rand_clown(n1=taille[i], n2=taille[i], sigma1=1, sigma2=6)
        X_train, X_test, y_train, y_test = train_test_split(donnees[0], donnees[1], train_size=0.8,random_state=0)
        #test=ErrorCurve(k_range=[i + 1 for i in range(len(X_train))])
        test=ErrorCurve(k_range=[i + 1 for i in range(50)])
        test.fit_curve(X=X_train,y=y_train, Xtest=X_test,  ytest=y_test)
        tri=np.min(test.errors)
        j=1
        while j<len(test.errors) and test.errors[j] != tri:
            j=j+1
        print ("Pour n ={:.0f} k={:.0f}.".format(taille[i],j+1))
        plt.subplot(1,3,i+1)
        test.plot()
    plt.show(figure)
    
testK_randclown()
from sklearn import datasets

digit = datasets.load_digits()
X, y = digit.data, digit.target
def weights(dist):
    res=[]
    res=np.array([np.exp(-1.0*((dist[k]*dist[k])/1000)) for k in range(len(dist))],dtype=np.float64)
    return res
             
echantillon=len(y)
taux=0.8
voisins=int(echantillon*taux)-1
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=taux,random_state=0)            
#test=ErrorCurve(k_range=[i + 1 for i in range(voisins)],weights=weights)
#test.fit_curve(X=X_train,y=y_train, Xtest=X_test,  ytest=y_test)
#test.plot()
#plt.show()
test=ErrorCurve(k_range=[i + 1 for i in range(50)],weights=weights)
test.fit_curve(X=X_train,y=y_train, Xtest=X_test,  ytest=y_test)
test.plot()
plt.show()
from timeit import timeit 
import seaborn as sns

def weights(dist):
    res=[]
    res=np.array([np.exp(-1.0*((dist[k]*dist[k])/1000)) for k in range(len(dist))],dtype=np.float64)
    return res

digit = datasets.load_digits()
X, y= digit.data, digit.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=0)     

from sklearn.neighbors import KNeighborsClassifier
#neigh = KNeighborsClassifier(n_neighbors=10,weights=weights)
neigh = KNeighborsClassifier(n_neighbors=10)

neigh.fit(X_train,y_train) 
predic_KNeighborsClassifier=neigh.predict(X_test)
for i in range(len(y_test)): 
    if y_test[i]!=predic_KNeighborsClassifier[i]:
        for j in (range(len(X))):
            e=0
            while e<len(digit.data[j]) and digit.data[j][e]==X_test[i][e]:
                e=e+1
            if e==len(digit.data[j]):
                print(y_test[i])
                print(predic_KNeighborsClassifier[i])
                plt.imshow(digit.images[j], cmap=plt.cm.Greys)
                plt.title('digit at index {}.'.format(i))
                plt.axis('off')
                plt.show()
perf= confusion_matrix(y_test, predic_KNeighborsClassifier)
print(perf)
sns.heatmap(perf, annot=True)  
plt.show()

