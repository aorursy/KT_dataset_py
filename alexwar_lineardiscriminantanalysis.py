""" the purpose of this book is to see the effects of *
 LDA on data with different structures and with different parameters
"""

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
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

def mediatrice(mu_1=[0,0],mu_2=[-1,-2]):
    milieu=np.array([(mu_1[0]+mu_2[0])/2,(mu_1[1]+mu_2[1])/2])
    print(milieu) #middle
    #pente de la droite mu1 mu2   #slope of the mu1 mu2 line
    a=(mu_1[1]-mu_2[1])/(mu_1[0]-mu_2[0])
    # mediatrice pente
    a1=-1/a
    b1=milieu[1]-a1*milieu[0]
    point1=np.array([10,a1*10+b1])
    point2=np.array([-10,-a1*10+b1])
    return np.array((point1,point2))
def analyse_lda(n_1=100, n_2=100, mu_1=[1, 1], mu_2=[-1, -1], sigma_1=[0.1, 0.1],sigma_2=[0.1, 0.1]):
    
    #generation du jeu de données #EN : generate data 
    randbigauss=rand_bi_gauss(n1=n_1, n2=n_2,mu1=mu_1,mu2=mu_2,sigma1=sigma_1,sigma2=sigma_2)
    #création de la base d'apprentissage et de validation 
    X_train, X_test, y_train, y_test = train_test_split(randbigauss[0],
                                                        randbigauss[1],
                                                        train_size=0.7,
                                                        random_state=0)
    #création du modéle LDA
    estimator = LinearDiscriminantAnalysis()
    estimator.fit(X_train, y_train)
    
    #Caractéristiques du modéle
    print("rand_bi_gauss : n1={:.0f} n2={:.0f} mu1=[{:.1f};{:.1f}]  mu2=[{:.1f};{:.1f}] sigma1=[{:.1f};{:.1f}]  sigma2=[{:.1f};{:.1f}] " \
              .format(n_1,n_2,mu_1[0],mu_1[1],
                      mu_2[0],mu_2[1],sigma_1[0],
                      sigma_1[1],sigma_2[0],sigma_2[1]))
    print("")
    print("Model result {}: ".format(type(estimator).__name__))
    print("--------------------------------------------------")
    print("     Accuracy of {}: {:.1f}%.".format(type(estimator).__name__,
                                                 estimator.score(X_test,y_test) * 100.0))
    
    erreur_apprentissage=1-accuracy_score(estimator.predict(X_train), y_train)
    erreur_test=1-accuracy_score(estimator.predict(X_test), y_test)
    print("     Error on train set : {:.1f}%.".format(erreur_apprentissage * 100))
    print("     Error on validation set : {:.1f}%.".format(erreur_test * 100))
    
    y_pred = estimator.predict(X_test)
    precision, recall, _, _ = precision_recall_fscore_support(y_test,y_pred)
    for i in range(2):
        print("     Accuracy is {:.2f}% and recall is {:.2f}% for class {}." \
          .format(precision[i] * 100,recall[i] * 100,i))
        

    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("rand_bi_gauss : n1={:.0f} n2={:.0f} mu1=[{:.1f};{:.1f}]  mu2=[{:.1f};{:.1f}] sigma1=[{:.1f};{:.1f}]  sigma2=[{:.1f};{:.1f}] " \
              .format(n_1,n_2,mu_1[0],mu_1[1],mu_2[0],
                      mu_2[1],sigma_1[0],sigma_1[1],
                      sigma_2[0],sigma_2[1]))
    
    axes.set_xlim(-10,10)
    plt.subplot(1,3,1)
    # Affichage du jeu de données
    plt.title("Dataset")
    plot_2d(randbigauss[0],randbigauss[1])
    #plt.show()
    
    plt.subplot(1,3,2) 
    # Affichage sur le jeu d'apprentissage :
    plt.title("Training result")

    # - du segment [mu_1;mu_2]
    plt.plot([mu_1[0], mu_2[0]], [mu_1[1],mu_2[1]], 'r-') 
    # - de la frontiére et des predictions
    plot_2d(X_train,y_train)
    frontiere(estimator.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=True)

    plt.subplot(1,3,3)
    
    # Affichage sur le jeu de validation :
    plt.title("Validation result")
    # - du segment [mu_1;mu_2]
    plt.plot([mu_1[0], mu_2[0]], [mu_1[1],mu_2[1]], 'r-') 
    # - de la frontiére et des predictions
    plot_2d(X_test,y_test)
    frontiere(estimator.predict, 
              X_test, step=100, 
              cmap_choice=cm.coolwarm, tiny=True)
    
    plt.show(figure)
analyse_lda(n_1=500, n_2=500, mu_1=[1, 1], mu_2=[-1, -1], sigma_1=[0.1, 0.1],sigma_2=[0.1, 0.1])
analyse_lda(n_1=500, n_2=500, mu_1=[2, 2], mu_2=[-5, 5], sigma_1=[2,2 ],sigma_2=[2, 2])

analyse_lda(n_1=500, n_2=500, mu_1=[2, -6], mu_2=[-1, 5], sigma_1=[3.5,3.5 ],sigma_2=[3.5,3.5])
def analyse_lda2(n_1=100, n_2=100, mu_1=[1, 1], mu_2=[-1, -1], sigma_1=[0.1, 0.1],sigma_2=[0.1, 0.1]):
    
    #generation du jeu de données
    randbigauss=rand_bi_gauss(n1=n_1, n2=n_2,mu1=mu_1,mu2=mu_2,sigma1=sigma_1,sigma2=sigma_2)
    
    #création de la base d'apprentissage et de validation
    X_train, X_test, y_train, y_test = train_test_split(randbigauss[0], randbigauss[1], train_size=0.7,random_state=0)

    #création du modéle LDA
    estimator = LinearDiscriminantAnalysis()
    estimator.fit(X_train, y_train)
    #calcul de la moyenne de l'échantillon d'apprentissage pour le calcul de la médiatrice
    X1_1=0
    X2_1=0
    X1_2=0
    X2_2=0
    nb1_1=0
    nb2_1=0
    nb1_2=0
    nb2_2=0
    for i in range(len(X_train)):
        if y_train[i]==1:
            X1_1=X1_1+X_train[i][0]
            nb1_1=nb1_1+1
            X1_2=X1_2+X_train[i][1]
            nb1_2=nb1_2+1
        else:
            X2_1=X2_1+X_train[i][0]
            nb2_1=nb2_1+1
            X2_2=X2_2+X_train[i][1]
            nb2_2=nb2_2+1           
    m1=[X1_1/nb1_1,X1_2/nb1_2]
    m2=[X2_1/nb2_1,X2_2/nb2_2]
    #Caractéristiques du modéle
    print("rand_bi_gauss : n1={:.0f} n2={:.0f} mu1=[{:.1f};{:.1f}]  mu2=[{:.1f};{:.1f}] sigma1=[{:.1f};{:.1f}]  sigma2=[{:.1f};{:.1f}] " \
              .format(n_1,n_2,mu_1[0],mu_1[1],mu_2[0],mu_2[1],sigma_1[0],sigma_1[1],sigma_2[0],sigma_2[1]))
    print("")
    print("Model result {}: ".format(type(estimator).__name__))
    print("--------------------------------------------------")
    print("     Accuracy of {}: {:.1f}%.".format(type(estimator).__name__,estimator.score(X_test,y_test) * 100.0))
    erreur_apprentissage=1-accuracy_score(estimator.predict(X_train), y_train)
    erreur_test=1-accuracy_score(estimator.predict(X_test), y_test)
    print("     Error on train set : {:.1f}%.".format(erreur_apprentissage * 100))
    print("     Error on validation set : {:.1f}%.".format(erreur_test * 100))
    
    y_pred = estimator.predict(X_test)
    precision, recall, _, _ = precision_recall_fscore_support(y_test,y_pred)
    for i in range(2):
        print("     Accuracy is {:.2f}% and recall is {:.2f}% pour la classe {}." \
          .format(precision[i] * 100,recall[i] * 100,i))
    # calcul de la mediatrice
    media=mediatrice(m1,m2)
    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("rand_bi_gauss : n1={:.0f} n2={:.0f} mu1=[{:.1f};{:.1f}]  mu2=[{:.1f};{:.1f}] sigma1=[{:.1f};{:.1f}]  sigma2=[{:.1f};{:.1f}] " \
              .format(n_1,n_2,mu_1[0],mu_1[1],mu_2[0],mu_2[1],sigma_1[0],sigma_1[1],sigma_2[0],sigma_2[1]))
    axes.set_xlim(-10,10)
    plt.subplot(1,3,1)
    # Affichage du jeu de données
    plt.title("Dataset")
    plot_2d(randbigauss[0],randbigauss[1])
    #plt.show()
    
    plt.subplot(1,3,2)
    
    # Affichage sur le jeu de validation :
    plt.title("Validation result")
    # - de la médiatrice du segment [mu_1;mu_2]
    plt.plot([media[0][0],media[1][0]], [media[0][1], media[1][1]], 'r--') 
    # - du segment [mu_1;mu_2]
    plt.plot([mu_1[0], mu_2[0]], [mu_1[1],mu_2[1]], 'r-') 
    # - de la frontiére et des predictions
    plot_2d(X_test,y_test)
    frontiere(estimator.predict, X_test, step=1000, cmap_choice=cm.coolwarm, tiny=True)
    #plt.show()

    plt.subplot(1,3,3) 
    # Affichage sur le jeu d'apprentissage :
    plt.title("Training result")

    # - de la médiatrice du segment [mu_1;mu_2]
    plt.plot([media[0][0],media[1][0]], [media[0][1], media[1][1]], 'r--') 
    # - du segment [mu_1;mu_2]
    plt.plot([mu_1[0], mu_2[0]], [mu_1[1],mu_2[1]], 'r-') 
    # - de la frontiére et des predictions
    plot_2d(X_train,y_train)
    frontiere(estimator.predict, X_train, step=1000, cmap_choice=cm.coolwarm, tiny=True)
    #plt.show()
    plt.show(figure)
analyse_lda2(n_1=500, n_2=500, mu_1=[10, 10], mu_2=[-5, -3], sigma_1=[2,2],sigma_2=[2, 2])

analyse_lda2(n_1=500, n_2=500, mu_1=[2, 2], mu_2=[-1, -1], sigma_1=[1, 1],sigma_2=[1, 1])

analyse_lda2(n_1=500, n_2=500, mu_1=[1, 1], mu_2=[-1, -1], sigma_1=[1, 2],sigma_2=[0.1, 0.5])

analyse_lda2(n_1=500, n_2=500, mu_1=[4, 2], mu_2=[-5, -2], sigma_1=[2, 2],sigma_2=[2,2])

analyse_lda2(n_1=500, n_2=500, mu_1=[4, 2], mu_2=[-5, -2], sigma_1=[1, 2],sigma_2=[0.1, 0.5])
analyse_lda2(n_1=500, n_2=500, mu_1=[4, 2], mu_2=[-5, -2], sigma_1=[1, 2],sigma_2=[1, 2])
def analyse_lda3(n_1=100, n_2=100, mu_1=[1, 1], mu_2=[-1, -1], sigma_1=[0.1, 0.1],sigma_2=[0.1, 0.1]):
    
    #generation du jeu de données
    randbigauss=rand_bi_gauss(n1=n_1, n2=n_2,mu1=mu_1,mu2=mu_2,sigma1=sigma_1,sigma2=sigma_2)
    
    #création de la base d'apprentissage et de validation
    X_train, X_test, y_train, y_test = train_test_split(randbigauss[0], randbigauss[1], train_size=0.7,random_state=0)

    #création du modéle LDA
    estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    estimator.fit(X_train, y_train)
    #calcul de la moyenne de l'échantillon d'apprentissage pour le calcul de la médiatrice
    X1_1=0
    X2_1=0
    X1_2=0
    X2_2=0
    nb1_1=0
    nb2_1=0
    nb1_2=0
    nb2_2=0
    for i in range(len(X_train)):
        if y_train[i]==1:
            X1_1=X1_1+X_train[i][0]
            nb1_1=nb1_1+1
            X1_2=X1_2+X_train[i][1]
            nb1_2=nb1_2+1
        else:
            X2_1=X2_1+X_train[i][0]
            nb2_1=nb2_1+1
            X2_2=X2_2+X_train[i][1]
            nb2_2=nb2_2+1           
    m1=[X1_1/nb1_1,X1_2/nb1_2]
    m2=[X2_1/nb2_1,X2_2/nb2_2]
    #Caractéristiques du modéle
    print("rand_bi_gauss : n1={:.0f} n2={:.0f} mu1=[{:.1f};{:.1f}]  mu2=[{:.1f};{:.1f}] sigma1=[{:.1f};{:.1f}]  sigma2=[{:.1f};{:.1f}] " \
              .format(n_1,n_2,mu_1[0],mu_1[1],mu_2[0],mu_2[1],sigma_1[0],sigma_1[1],sigma_2[0],sigma_2[1]))
    print("")
    print("Model result {}: ".format(type(estimator).__name__))
    print("--------------------------------------------------")
    print("     Accuracy of {}: {:.1f}%.".format(type(estimator).__name__,estimator.score(X_test,y_test) * 100.0))
    erreur_apprentissage=1-accuracy_score(estimator.predict(X_train), y_train)
    erreur_test=1-accuracy_score(estimator.predict(X_test), y_test)
    print("     Error on train set : {:.1f}%.".format(erreur_apprentissage * 100))
    print("     Error on validation set : {:.1f}%.".format(erreur_test * 100))
    
    y_pred = estimator.predict(X_test)
    precision, recall, _, _ = precision_recall_fscore_support(y_test,y_pred)
    for i in range(2):
        print("     Accuracy is {:.2f}% and recall is {:.2f}% pour la classe {}." \
          .format(precision[i] * 100,recall[i] * 100,i))
    # calcul de la mediatrice
    media=mediatrice(m1,m2)
    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("rand_bi_gauss : n1={:.0f} n2={:.0f} mu1=[{:.1f};{:.1f}]  mu2=[{:.1f};{:.1f}] sigma1=[{:.1f};{:.1f}]  sigma2=[{:.1f};{:.1f}] " \
              .format(n_1,n_2,mu_1[0],mu_1[1],mu_2[0],mu_2[1],sigma_1[0],sigma_1[1],sigma_2[0],sigma_2[1]))
    axes.set_xlim(-10,10)
    plt.subplot(1,3,1)
    # Affichage du jeu de données
    plt.title("Dataset")
    plot_2d(randbigauss[0],randbigauss[1])
    #plt.show()
    
    plt.subplot(1,3,2)
    
    # Affichage sur le jeu de validation :
    plt.title("Validation result")
    # - de la médiatrice du segment [mu_1;mu_2]
    plt.plot([media[0][0],media[1][0]], [media[0][1], media[1][1]], 'r--') 
    # - du segment [mu_1;mu_2]
    plt.plot([mu_1[0], mu_2[0]], [mu_1[1],mu_2[1]], 'r-') 
    # - de la frontiére et des predictions
    plot_2d(X_test,y_test)
    frontiere(estimator.predict, X_test, step=1000, cmap_choice=cm.coolwarm, tiny=True)
    #plt.show()

    plt.subplot(1,3,3) 
    # Affichage sur le jeu d'apprentissage :
    plt.title("Training result")

    # - de la médiatrice du segment [mu_1;mu_2]
    plt.plot([media[0][0],media[1][0]], [media[0][1], media[1][1]], 'r--') 
    # - du segment [mu_1;mu_2]
    plt.plot([mu_1[0], mu_2[0]], [mu_1[1],mu_2[1]], 'r-') 
    # - de la frontiére et des predictions
    plot_2d(X_train,y_train)
    frontiere(estimator.predict, X_train, step=1000, cmap_choice=cm.coolwarm, tiny=True)
    #plt.show()
    plt.show(figure)
analyse_lda2(n_1=500, n_2=500, mu_1=[4, 2], mu_2=[-5, -2], sigma_1=[2, 2],sigma_2=[2,2])

analyse_lda3(n_1=500, n_2=500, mu_1=[4, 2], mu_2=[-5, -2], sigma_1=[1, 2],sigma_2=[0.1, 0.5])
def analyse_lda4(n_1=100, n_2=100, mu_1=[1, 1], mu_2=[-1, -1], sigma_1=[0.1, 0.1],sigma_2=[0.1, 0.1]):
    
    #generation du jeu de données #EN : generate data 
    randbigauss=rand_bi_gauss(n1=n_1, n2=n_2,mu1=mu_1,mu2=mu_2,sigma1=sigma_1,sigma2=sigma_2)
    #création de la base d'apprentissage et de validation 
    X_train, X_test, y_train, y_test = train_test_split(randbigauss[0],
                                                        randbigauss[1],
                                                        train_size=0.7,
                                                        random_state=0)
    #création du modéle LDA
    estimator = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    estimator.fit(X_train, y_train)
    
    #Caractéristiques du modéle
    print("rand_bi_gauss : n1={:.0f} n2={:.0f} mu1=[{:.1f};{:.1f}]  mu2=[{:.1f};{:.1f}] sigma1=[{:.1f};{:.1f}]  sigma2=[{:.1f};{:.1f}] " \
              .format(n_1,n_2,mu_1[0],mu_1[1],
                      mu_2[0],mu_2[1],sigma_1[0],
                      sigma_1[1],sigma_2[0],sigma_2[1]))
    print("")
    print("Model result {}: ".format(type(estimator).__name__))
    print("--------------------------------------------------")
    print("     Accuracy of {}: {:.1f}%.".format(type(estimator).__name__,
                                                 estimator.score(X_test,y_test) * 100.0))
    
    erreur_apprentissage=1-accuracy_score(estimator.predict(X_train), y_train)
    erreur_test=1-accuracy_score(estimator.predict(X_test), y_test)
    print("     Error on train set : {:.1f}%.".format(erreur_apprentissage * 100))
    print("     Error on validation set : {:.1f}%.".format(erreur_test * 100))
    
    y_pred = estimator.predict(X_test)
    precision, recall, _, _ = precision_recall_fscore_support(y_test,y_pred)
    for i in range(2):
        print("     Accuracy is {:.2f}% and recall is {:.2f}% for class {}." \
          .format(precision[i] * 100,recall[i] * 100,i))
        

    figure=plt.figure()
    figure,axes=plt.subplots(figsize=(15,5))
    figure.suptitle("rand_bi_gauss : n1={:.0f} n2={:.0f} mu1=[{:.1f};{:.1f}]  mu2=[{:.1f};{:.1f}] sigma1=[{:.1f};{:.1f}]  sigma2=[{:.1f};{:.1f}] " \
              .format(n_1,n_2,mu_1[0],mu_1[1],mu_2[0],
                      mu_2[1],sigma_1[0],sigma_1[1],
                      sigma_2[0],sigma_2[1]))
    
    axes.set_xlim(-10,10)
    plt.subplot(1,3,1)
    # Affichage du jeu de données
    plt.title("Dataset")
    plot_2d(randbigauss[0],randbigauss[1])
    #plt.show()
    
    plt.subplot(1,3,2) 
    # Affichage sur le jeu d'apprentissage :
    plt.title("Training result")

    # - du segment [mu_1;mu_2]
    plt.plot([mu_1[0], mu_2[0]], [mu_1[1],mu_2[1]], 'r-') 
    # - de la frontiére et des predictions
    plot_2d(X_train,y_train)
    frontiere(estimator.predict, X_train, step=100, cmap_choice=cm.coolwarm, tiny=True)

    plt.subplot(1,3,3)
    
    # Affichage sur le jeu de validation :
    plt.title("Validation result")
    # - du segment [mu_1;mu_2]
    plt.plot([mu_1[0], mu_2[0]], [mu_1[1],mu_2[1]], 'r-') 
    # - de la frontiére et des predictions
    plot_2d(X_test,y_test)
    frontiere(estimator.predict, 
              X_test, step=100, 
              cmap_choice=cm.coolwarm, tiny=True)
    
    plt.show(figure)
analyse_lda(n_1=500, n_2=500, mu_1=[2, -6], mu_2=[-1, 5], sigma_1=[3.5,3.5 ],sigma_2=[3.5,3.5])
analyse_lda4(n_1=500, n_2=500, mu_1=[2, -6], mu_2=[-1, 5], sigma_1=[3.5,3.5 ],sigma_2=[3.5,3.5])
analyse_lda(n_1=500, n_2=500, mu_1=[2, 2], mu_2=[-5, 5], sigma_1=[2,2 ],sigma_2=[2, 2])
analyse_lda4(n_1=500, n_2=500, mu_1=[2, 2], mu_2=[-5, 5], sigma_1=[2,2 ],sigma_2=[2, 2])