# Import des librairies
import os 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
# Chargement de la base de données
# On limite à 50000 entrées afin de travailler en ligne efficacement
df = pd.read_csv("../input/deaths/kill_match_stats_final_0.csv")
df = df[df.map=="ERANGEL"]
df = df.dropna()
n_entries=50000
deaths = df.head(n_entries)
deaths.head()
deaths['distance'] = np.sqrt(np.power(deaths['victim_position_x']-deaths['killer_position_x'],2)+np.power(deaths['victim_position_y']-deaths['killer_position_y'],2))
# Simplification du dataset aux données nous intéressant
deaths = deaths[['victim_name','victim_placement','distance','time','killer_name','killer_placement']]
deaths.head()
features = ['time','killer_placement','distance','victim_placement']

x = deaths[features]
x = StandardScaler().fit_transform(x)

print("Les moyennes :",np.mean(x,axis=0)) # les moyennes devraient être égales à 0
print("\nLes écarts-types :",np.std(x,axis=0,ddof=0)) # les écarts type devraient être égals à 1
print("\nLa matrice de corrélation :\n",(1/n_entries)*np.matmul(np.transpose(x),x))
n_components=len(features)
pca = PCA()
principal_comps = pca.fit_transform(x)
print("Nombre de composantes principales : ",pca.n_components_)
print("\nValeurs propres :",pca.explained_variance_)
print("\nProportion des variances (pourcentage) : ",pca.explained_variance_ratio_*100)
corvar=pca.get_covariance()
print("\nMatrice des covariances :\n",corvar)
plt.plot(np.arange(1,n_components+1),np.cumsum(pca.explained_variance_ratio_))
plt.title("Variance cumulée par rapport au nombre de composantes")
plt.ylabel("Variance cumulée")
plt.xlabel("Nombre de composantes")
plt.show()
n_components=2
pca = PCA(n_components)
principal_comps = pca.fit_transform(x)
print("Nombre de composantes principales : ",pca.n_components_)
print("\nValeurs propres :",pca.explained_variance_)
print("\nProportion des variances (pourcentage) : ",pca.explained_variance_ratio_*100)
# Calcul des vecteurs propres
# On obtient la matrice de changement de base
print("Matrice de changement de base :\n",pca.components_)
print("\nComposantes principales :\n",principal_comps)
# Positionnement des individus dans le premier plan
fig, axes = plt.subplots()
tmp_df=pd.DataFrame(columns=list(deaths))
# Placement des étiquettes des observations
for i in range(10):
    r=random.randint(0,n_entries)
    tmp_df.loc[i]=deaths.iloc[r]
    plt.annotate(tmp_df.loc[i].victim_name,(principal_comps[r,0],principal_comps[r,1]))
    
# Ajouter les axes
plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
plt.margins(x=-.25, y=-0.465)
# Affichage
plt.show()
tmp_df
# Contribution des individus dans l'inertie totale
dist = np.sum(x**2,axis=1)
print("Contribution des individus dans l'inertie totale :\n",dist)
# Qualité de la représentation des individus
cos2 = principal_comps**2
for i in range(n_components):
    cos2[:,i] = cos2[:,i]/dist
print("\nQualité de la représentation des individus (pourcentage) :\n",cos2*100)
# Contributions aux axes
ctr = principal_comps**2
for j in range(n_components):
    ctr[:,j] = ctr[:,j]/(n_entries*pca.explained_variance_[j])

print("Contribution aux axes (10 permiers) : \n")
print(pd.DataFrame({'Axe 1':ctr[:,0],'Axe 2':ctr[:,1]}).head(10))
# Corrélations entre variables et facteurs
sqrt_vp = np.sqrt(pca.explained_variance_)
varfac = np.zeros((len(features),len(features)))
for i in range(n_components): 
    varfac[:,i] = pca.components_[i,:] * sqrt_vp[i]

print(pd.DataFrame({'Variable':features,'Composante 1':varfac[:,0],'Composante 2':varfac[:,1]}))
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
# Affichage des étiquettes (noms des variables)
for j in range(len(features)):
    plt.annotate(features[j],(varfac[j,0],varfac[j,1]))

# Ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
# Ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
# Affichage
plt.show()