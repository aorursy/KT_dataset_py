import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



import os

print(os.listdir("../input"))

dataset = pd.read_csv("../input/covtype.csv")

dataset.head(10)



# Any results you write to the current directory are saved as output.
labels = dataset["Cover_Type"]

del dataset["Cover_Type"]









X = dataset.values# la matrice de donnees





#instanciation

sc = StandardScaler()

#transformation–centrage-réduction

Z = sc.fit_transform(X)

print(Z)



#moyenne

print(np.mean(Z,axis=0))



#écart-type

print(np.std(Z,axis=0,ddof=0))



#instanciation

acp = PCA(svd_solver='full')



#calculs

coord = acp.fit_transform(Z)# les coordonnees factorielles

#nombre de composantes calculées

print(acp.n_components_) 

#variance expliquée

print(acp.explained_variance_)



eigval = acp.explained_variance_



#proportion de variance expliquée,l'info expliquee par les axes

print(acp.explained_variance_ratio_)

#valeur corrigée

#j = X.shape[0]

#eigval = (j-1)/j*acp.explained_variance_

#print(eigval)



#critere du coude

mu = X.shape[1]# ( j'ai remplacé p par mu)

plt.plot(np.arange(1,mu+1),eigval)

plt.title("critère du coude")

plt.ylabel("valeurs propres")

plt.xlabel("nombre de facteurs")

plt.show()





#cumul de variance expliquée

plt.plot(np.arange(1,mu+1),np.cumsum(acp.explained_variance_ratio_))

plt.title("variance expliquée")

plt.ylabel("ratio variance cumulée")

plt.xlabel("nombre de facteurs")

plt.show()



##################################################################################



pca = PCA(n_components=20)

print(pca.fit(X))

pca.explained_variance_ratio_



plt.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)

plt.title("Variance expliquée")





pca.components_



sum(pca.explained_variance_ratio_)







X_pca = pca.fit_transform(X)



n=581012  # nombre de composantes

k=20   # nombre de features

n_a=10  # nombre d'actions



D=X_pca # data



theta=np.random.random( (n_a,k) ) + 0.00001 # le theta à prédire









choix=np.zeros(n)

recomp=np.zeros(n)

explore=np.zeros(n)

norms  =np.zeros(n)

b      =np.zeros_like(theta)

A      =np.zeros( (n_a, k,k)  )

for a in range (0,n_a):

    A[a]=np.identity(k)



A_inv = [np.identity(k) for i in range (0,n_a)]



theta_hat =np.zeros_like(theta) # features temporaires, meilleures suppositions actuelles

p      =np.zeros(n_a)

alpha   =5





# LinUCB



for i in range(0,n):

 

    x_i = D[i]   # vect du contexte

    

    for a in range (0,n_a):        

        theta_hat[a]  = A_inv[a].dot(b[a])      

        ta         = x_i.T.dot(A_inv[a]).dot(x_i) 

        a_upper_ci = alpha * np.sqrt(ta)     # partie supérieure de l'intervalle 

        a_mean     = theta_hat[a].dot(x_i)   # estimation actuelle de la mean

        p[a]       = a_mean + a_upper_ci     # borne sup IC

      

      

    norms[i]       = np.linalg.norm(theta_hat - theta,'fro') # convergence

    

    choix[i] = p.argmax()   #fonction de maxmisation

    

    # le resultat obtenu

    

    recomp[i] =((choix[i])==labels[i]) #la récompense

   

    # mettre à jour le vecteur d'input

    A[int(choix[i])]      += np.outer(x_i,x_i)

    A_inv[int(choix[i])]  = np.linalg.inv(A[int(choix[i])])

    b[int(choix[i])]      += recomp[i] * x_i

    







regret=(np.ones(n) - recomp)







plt.subplot(221)

plt.plot(recomp.cumsum())

plt.title("Recompense Cumulée ")

plt.subplot(222)

plt.plot(regret.cumsum(), color = "red")

plt.title("Regret Cumulée ")

plt.show()