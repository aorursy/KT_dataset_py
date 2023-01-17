import numpy as np

import matplotlib.pyplot as plt



# les arrays sont batis avec les dimensions suivantes :

# pluie , arroseur , watson , holmes

# et chaque dimension : faux , vrai



prob_pluie = np.array ( [ 0.8,0.2] ).reshape( 2 , 1 , 1 , 1 )

print( "Pr( Pluie ) ={}\n".format( np.squeeze ( prob_pluie ) ) )

prob_arroseur = np.array ( [0.9,0.1] ).reshape( 1 , 2 , 1 , 1 )

print( "Pr ( Arroseur ) ={}\n".format( np.squeeze(prob_arroseur) ) )

watson = np.array( [ [0.8,0.2] , [ 0 , 1 ] ] ) .reshape ( 2 , 1 , 2 , 1 )

print( "Pr (Watson | Pluie ) ={}\n" . format ( np.squeeze( watson ) ) )

holmes = np.array([[1,0],[0.1,0.9],[0,1],[0,1]]).reshape(2,2,1,2)

print( "Pr ( Holmes | Pluie ,arroseur ) ={}\n" .format ( np.squeeze( holmes ) ) )

watson [ 0 , : , 1 , : ] # prob watson mouille − plui e

holmes [ 0 , 1 , 0 , 1 ] # prob gazon holmes mouille si a r r o s e u r − p l u i e
#### 1-a:

'''P(W=1)'''

P_W=(watson*prob_pluie).sum(0).squeeze( )[ 1 ] # prob gazon watson mouille

print( "Pr(W = 1) = {}\n".format( P_W ) )
#### 1-b:

prob_H=(holmes*prob_pluie*prob_arroseur).sum(0).sum(0).squeeze()[1]
'''P(W=1|H=1)'''

P_WH=(watson * holmes * prob_pluie*prob_arroseur).sum(0).sum(0)[1,1]/prob_H

print( "Pr(W = 1|H = 1) = {}\n".format( P_WH ) )
#### 1-c:

'''P(W=1,H=1,A=0)'''

P1=(holmes*prob_pluie*prob_arroseur*watson).sum(0).squeeze()[0,1,1]

'''P(H=1,A=0)'''

P2=(holmes*prob_pluie*prob_arroseur*watson).sum(0).sum(1).squeeze()[0,1]

'''P(W=1|H=1,A=0)'''

print( "Pr(W = 1|H = 1, A = 0) = {}\n".format( np.squeeze(P1/P2) ) )
#### 1-d:

'''P(W=1|A=0)'''

#independent donc p(W=1)

P_W=(watson*prob_pluie).sum(0).squeeze( )[ 1 ] # prob gazon watson m o uill e

print( "Pr(W = 1|H = 1, A = 0) = {}\n".format( np.squeeze(P_W) ) )
P_W_A=(holmes*prob_pluie*prob_arroseur*watson).sum(0).sum(2).squeeze()[0,1]

P_A=prob_arroseur.squeeze()[0]

print( "Pr(W = 1|H = 1, A = 0) = {}\n".format( np.squeeze(P_W_A/P_A) ) )
#### 1-e:

'''P(W = 1|P = 1)'''

print( "Pr(W = 1|P = 1) = {}\n".format( watson[1,:,1,:].squeeze() ) )
""" Import Libraries """

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.model_selection import train_test_split
""" Import Data """

digits= datasets. load_digits( )



X = digits.data

y = digits.target
""" Shuffle Data """

#import Shuffle and Seed from Random Library 

from random import shuffle,seed

#Set seed

seed(1)

#Shuffle Data

ind_list = [i for i in range(len(y))]

shuffle(ind_list)

X  = X[ind_list, :,]

y = y[ind_list,]
y_one_hot = np.zeros ( ( y.shape [ 0 ] , len ( np.unique ( y ) ) ) )

y_one_hot [ np.arange ( y.shape [ 0 ] ) , y ] = 1 # one hot target or shape NxK
""" Ajout de colonne remplie de 1 à X et initialisation de W avec des valeurs aléatoires faibles """

print("La dimension de X avant d'ajouter la colonne des biais est ",X.shape)

#Add Biais column

X = np.hstack((np.ones((X.shape[0],1)),X))    

print("La dimension de X après l'ajout de la colonne des biais est ",X.shape)





""" Initialiser les valeurs de W aléatoirement (loi Normale) """

#W c'est theta dans ce cas 

W = np.random.normal ( 0 , 0.01 , ( len ( np.unique ( y ) ) , X.shape [ 1 ] ) ) # weights of shape Kx(L+1)
X_train , X_test , y_train , y_test = train_test_split(X, y_one_hot , test_size =0.3 , random_state=42)

X_test , X_validation , y_test , y_validation = train_test_split ( X_test , y_test , test_size =0.5 , random_state=42)
def softmax ( x ) :

    """Sotmax function """

    # assurez vous que la fonction est numeriquement stable

    # e.g . softmax (np.array( [ 1000 , 10000 , 100000] , ndmin=2) )

    p = np.zeros(x.shape)

    m=np.max(x,axis=1)

    for i in range(x.shape[0]):

        p[i,:]  = np.exp(x[i,:]-m[i])

        p[i,:] /= np.sum(p[i,:])

    return(p)
""" Test demandé pour la stabilité """

softmax(np.array( [ 1000 , 10000 , 100000] , ndmin=2))
def get_accuracy (X, y , W) :

    """ Donne la précision """

    y_proba=softmax(np.dot(X,np.transpose(W)))

    y_test=np.zeros(y_proba.shape)

    y_test[np.arange ( y_test.shape [ 0 ] ) , np.argmax(y_proba, axis=1) ] = 1 

    return(float(np.sum(np.sum(y_test==y,axis=1)==W.shape[0]))/X.shape[0])
def get_grads ( y , y_pred , X) :

    #y_pred est la prediction en probibilité 

    dJ=np.dot(np.transpose(X),(y_pred-y))/X.shape[0]

    return(dJ)
#0*np.log(0)
def get_loss ( y , y_pred ) :

    #pour des raisons de stabilité, on remplace les prob egale 0 et 1 par epsilon et 1-epsilon

    y_pred[y_pred>1-1.0e-5]=1-1.0e-5

    y_pred[y_pred<1.0e-5]=1.0e-5

    J=-np.sum(np.log(y_pred)*y)

    return(J)
def train_model(nb_epochs, minibatch_size, lr,W):

    losses = [ ]

    val_losses=[]

    accuracies = [ ]

    best_W = None

    best_accuracy = 0

    for epoch in range ( nb_epochs ) :

        loss = 0

        accuracy = 0

        for i in range ( 0 , X_train.shape [ 0 ] , minibatch_size ):

            if i+minibatch_size>X_train.shape [ 0 ]:

                end=X_train.shape [ 0 ]

            else:

                end=i+minibatch_size



            logits = np.dot(X_train[i:end,:],np.transpose(W))

            y_pred = softmax(logits)

            loss+= get_loss(y_train[i:end,:],y_pred)  

            W=W-lr*np.transpose(get_grads(y_train[i:end,:],y_pred, X_train[i:end,:]))

   

        losses.append ( loss/y_train.shape[0] ) # compute the l o s s on the t r a i n s e t

        val_losses.append(get_loss(y_validation,softmax(np.dot(X_validation, np.transpose(W))))/y_validation.shape[0] )

        accuracy = get_accuracy(X_validation,y_validation,W)

        accuracies.append ( accuracy ) # compute the acc u r a c y on the v a l i d a t i o n s e t

        if accuracy > best_accuracy :

            best_accuracy=accuracy

            best_W=W# s e l e c t the b e s t p a r ame te r s based on the v a l i d a t i o n ac cu r ac y

    return(best_W,losses,val_losses)

lr = 0.001

nb_epochs = 50

minibatch_size = len ( y ) // 20

best_W,losses,val_losses=train_model(nb_epochs, minibatch_size, lr,W)
accuracy_on_unseen_data = get_accuracy ( X_test , y_test , best_W )

print ( accuracy_on_unseen_data ) # 0.897506925208
plt.imshow (best_W [ 4 , 0: 64].reshape ( 8 , 8 ) )

plt.show()
plt.plot ( losses ,color="blue")

plt.plot(val_losses,color="orange")

plt.legend(('train', 'validation'),loc='upper right')

plt.title('Loss function')

plt.xlabel("epochs")

plt.ylabel("negative log likelihood")

plt.show()
print("La moyenne des poids associés aux dimensions contenant des vraies données: " ,np.mean(best_W))

print("La variance des poids associés aux dimensions contenant des vraies données: ",np.var(best_W))
lrs=[0.1, 0.01, 0.001]

minibatch_sizes=[len(y), len(y)//20, len(y)//200, len(y)//1000]

for lr in lrs:

    for minibatch_size in minibatch_sizes:

        W = np.random.normal ( 0 , 0.01 , ( len ( np.unique ( y ) ) , X.shape [ 1 ] ) ) #initialize weights

        best_W,losses,val_losses=train_model(nb_epochs, minibatch_size, lr,W)

        accuracy_on_unseen_data = get_accuracy ( X_test , y_test , best_W )

        legende="lr:"+str(lr)+" & minibatch_size:"+str(minibatch_size)

        print(legende)

        print ("l'accaracy est: ", accuracy_on_unseen_data ) 

        plt.imshow (best_W [ 4 , 0: 64].reshape ( 8 , 8 ) )

        plt.show()

        plt.plot ( losses ,color="blue")

        plt.plot(val_losses,color="orange")

        plt.legend(('train', 'validation'),loc='upper right')

        plt.title('Loss function')

        plt.xlabel("epochs")

        plt.ylabel("negative log likelihood")

        plt.show()



""" Load Dataset """

digits = datasets. load_digits( )



X = digits.data

y = digits.target

y_one_hot = np.zeros ( ( y.shape [ 0 ] , len ( np.unique ( y ) ) ) )

y_one_hot [ np.arange ( y.shape [ 0 ] ) , y ] = 1 # one hot t a r g e t o r shape NxK
""" Ajout des 8 dimensions """

dimensions = np.random.uniform ( np.min(X),np.max(X), (X.shape[0],8 ) ) # add 8 random uniform dimensions  

X=np.hstack((X,dimensions))
""" Ajout de la colonne du biais """

X = np.hstack((np.ones((X.shape[0],1)),X))
""" Split Train Validation Test """

X_train , X_test , y_train , y_test = train_test_split(X, y_one_hot , test_size =0.3 , random_state=42)



X_test , X_validation , y_test , y_validation = train_test_split ( X_test , y_test , test_size =0.5 , random_state=42)
""" Initialiser les poids W """

W = np.random.normal ( 0 , 0.01 , ( len ( np.unique ( y ) ) , X.shape [ 1 ] ) )
""" Choix des hyperparametres """

nb_examples=X.shape[0]

lr = 0.001

nb_epochs = 50

minibatch_size = len ( y ) // 20



""" La fonction Softmax est restée intacte """

def softmax ( x ) :

    p = np.zeros(x.shape)

    m=np.max(x,axis=1)

    for i in range(x.shape[0]):

        p[i,:]  = np.exp(x[i,:]-m[i])

        p[i,:] /= np.sum(p[i,:])

    return(p)

         

    

""" La fonction Accuracy reste la même """

def get_accuracy (X, y , W) :

    y_proba=softmax(np.dot(X,np.transpose(W)))

    y_test=np.zeros(y_proba.shape)

    y_test[np.arange ( y_test.shape [ 0 ] ) , np.argmax(y_proba, axis=1) ] = 1 

    return(float(np.sum(np.sum(y_test==y,axis=1)==W.shape[0]))/X.shape[0])
""" Calcule le gradient de la fonction Loss """

def get_grads ( y , y_pred , X,alpha, beta,W) :

    # y_pred est la prediction en probibilité 

    # On met les poids du biais à zero car on ne le prend pas en considération 

    W=W.copy()

    W[:,0]=0

    dJ=np.dot(np.transpose(X),(y_pred-y))/X.shape[0]

    dJ+=(alpha*np.transpose(W)*2+beta*np.sign(np.transpose(W)))*X.shape[0]/nb_examples

    return(dJ)

""" Calcul la fonction Loss """

def get_loss ( y , y_pred ,alpha,beta,W) :

    #pour des raisons de stabilité, on remplace les prob egale 0 et 1 par des proba plus petite 

    y_pred[y_pred>1-1.0e-5]=1-1.0e-5

    y_pred[y_pred<1.0e-5]=1.0e-5

    J=-np.sum(np.log(y_pred)*y)

    W=W.copy()

    W[:,0]=0

    # On rajoute le terme de la regularisation 

    J=J+alpha*np.sum(W**2)+beta*np.sum(np.absolute(W))

    return(J)
def train_model(nb_epochs, minibatch_size, lr,W,alpha,beta):

    losses = [ ]

    val_losses=[]

    accuracies = [ ]

    best_W = None

    best_accuracy = 0

    for epoch in range ( nb_epochs ) :

        loss = 0

        accuracy = 0

        for i in range ( 0 , X_train.shape [ 0 ] , minibatch_size ):

            if i+minibatch_size>X_train.shape [ 0 ]:

                end=X_train.shape [ 0 ]

            else:

                end=i+minibatch_size





            logits = np.dot(X_train[i:end,:],np.transpose(W))

            y_pred = softmax(logits)

            loss+= get_loss(y_train[i:end,:],y_pred,alpha,beta,W)  

            W=W-lr*np.transpose(get_grads(y_train[i:end,:],y_pred, X_train[i:end,:],alpha,beta,W))

   

        losses.append ( loss/y_train.shape[0] ) # compute the l o s s on the t r a i n s e t

        val_losses.append(get_loss(y_validation,softmax(np.dot(X_validation, np.transpose(W))),alpha,beta,W)/y_validation.shape[0] )

        accuracy = get_accuracy(X_validation,y_validation,W)

        accuracies.append ( accuracy ) # compute the acc u r a c y on the v a l i d a t i o n s e t

        if accuracy > best_accuracy :

            best_accuracy=accuracy

            best_W=W     # select the best parameters based on the validation accuracy

    return(best_W,losses,val_losses,best_accuracy)



#test avec un exemple de alpha et beta 

best_W,losses,val_losses,accuracy=train_model(nb_epochs, minibatch_size, lr,W,alpha=0.1,beta=0.001)

accuracy_on_unseen_data = get_accuracy ( X_test , y_test , best_W )

print ( "L'accuracy pour le test est: ",accuracy_on_unseen_data ) 



#Dessin des poids pour le chiffre 4 

plt.imshow (best_W [ 4 , 0: 64].reshape ( 8 , 8 ) )

plt.show()

alphas=[0.00001,0.0001,0.001,0.01,0.1]

betas=[0.00001,0.0001,0.001,0.01,0.1]

#liste pour tout enregistrer

valeur_alpha=[]

valeur_beta=[]

moyenne_sans=[]

moyenne_8=[]

variance_sans=[]

variance_8=[]





best_acc=0

best_alpha=None

best_beta=None

best_W=None

i=0

for alpha in alphas: 

    for beta in betas: 

        i+=1

        print(i,"/",len(alphas)*len(betas))

        W = np.random.normal ( 0 , 0.01 , ( len ( np.unique ( y ) ) , X.shape [ 1 ] ) ) # w ei g h t s o f shape KxL

        W,losses,val_losses,accuracy = train_model(100, minibatch_size, lr,W,alpha,beta)

        

        #Remplir les donnes des moyennes et variances

        valeur_alpha.append(alpha)

        valeur_beta.append(beta)

        moyenne_sans.append(np.mean(W[:,:65]))

        moyenne_8.append(np.mean(W[:,-8:]))

        variance_sans.append(np.var(W[:,:65]))

        variance_8.append(np.var(W[:,-8:]))

        

        if accuracy>best_acc:

            best_acc=accuracy

            best_alpha=alpha

            best_beta=beta

            best_W=W
print("la meilleure combinaison trouvée est pour (λ1,λ2) est (", best_alpha ,",",best_beta,")")
plt.plot(np.log10(valeur_alpha),moyenne_8,'ro')

plt.plot(np.log10(valeur_alpha),moyenne_sans,'bo')

plt.plot(np.log10(valeur_alpha),variance_8,'go')

plt.plot(np.log10(valeur_alpha),variance_sans,'co')

plt.axis(ymax=0.005)

plt.legend(('Moyenne poids des 8 dimensions', 'Moyennes des poids contenant les vraies valeurs','Variance poids des 8 dimensions', 'Variance des poids contenant les vraies valeurs'),loc='upper right')

plt.title('Courbe en fonction de log10(alpha)')

plt.xlabel("log10(alpha)")

plt.show()
plt.plot(np.log10(valeur_beta),moyenne_8,'ro')

plt.plot(np.log10(valeur_beta),moyenne_sans,'bo')

plt.plot(np.log10(valeur_beta),variance_8,'go')

plt.plot(np.log10(valeur_beta),variance_sans,'co')

plt.axis(ymax=0.005)

plt.legend(('Moyenne poids des 8 dimensions', 'Moyennes des poids contenant les vraies valeurs','Variance poids des 8 dimensions', 'Variance des poids contenant les vraies valeurs'),loc='upper right')

plt.title('Courbe en fonction de log10(beta)')

plt.xlabel("log10(beta)")

plt.show()
# This import registers the 3D projection, but is otherwise unused.

from mpl_toolkits.mplot3d import Axes3D  



import matplotlib.pyplot as plt



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



n = 100





for c, m, zlow, zhigh in [('r', 'o', -50, -25)]:

    xs = np.log10(valeur_alpha)

    ys = np.log10(valeur_beta)

    zs = np.matrix(np.log10(abs(np.array(moyenne_8))))

    ax.scatter(xs, ys, zs, c=c, marker=m)



ax.set_xlabel('log(Alpha)')

ax.set_ylabel('log(Beta)')

ax.set_zlabel('log(|moyenne|) poids 8 dimensions')



plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



n = 100





for c, m, zlow, zhigh in [('r', 'o', -50, -25)]:

    xs = np.log10(valeur_alpha)

    ys = np.log10(valeur_beta)

    zs = np.matrix(np.log10(abs(np.array(variance_8))))

    ax.scatter(xs, ys, zs, c=c, marker=m)



ax.set_xlabel('log(Alpha)')

ax.set_ylabel('log(Beta)')

ax.set_zlabel('log(|variance|) poids 8 dimensions')



plt.show()
print("La moyenne des poids associés aux dimensions contenant des vraies données: " ,np.mean(best_W[:,:65]))

print("La variance des poids associés aux dimensions contenant des vraies données: ",np.var(best_W[:,:65]))

print("#######################################")

print("La moyenne des poids associés à ces dimensions contenant des valeurs aléatoires est: " ,np.mean(best_W[:,-8:]))

print("La variance des poids associés à ces dimensions contenant des valeurs aléatoires est: ",np.var(best_W[:,-8:]))