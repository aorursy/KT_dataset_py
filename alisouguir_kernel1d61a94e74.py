#cet script a pour role du calculer la force critique de voilement  d'apres le paramètre adimentionnel  w  sans passer par les calcules matricielles intenses  qui prend beacoup de temps
#en utilisant une approche  d'intelligence artificielle qui donnes des resultats robustes  dans une fraction de  seconde
#importation des bibs d'algebre lineaire
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#importation de la base de données à partir du fichier excel  contenant w et fcr
data=pd.read_csv("../input/geniecivile/home/mohaned/Downloads/e5.csv")
#visualisation du donnees
data

#inforamation sur les types de donnes
data.info()
#des  statistiques generales sur les donnes
data.describe()
#graphe modelisant  la  relation entre w et f qui est clairement non linaire
plt.scatter(data.iloc[:,0],data.iloc[:,1])
#normalization du  donnes en soustraire le moyen et en divise par la variance
from sklearn import preprocessing
x_array = np.array(data['ww'])
y_array = np.array(data['ff'])
l1=preprocessing.normalize([x_array])
data.iloc[:,0]=l1[0]
data=data.sample(frac=1)
data=data.reset_index(drop=True)
X=data.iloc[:,0]
y=data.iloc[:,1]
#observation du changement du l'echelle du donnees
data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#repartition du donnees entre train et test pour evalué le model mathematique developpé d'une maniere  correcte
X_train, X_test, y_train, y_test = train_test_split(X, y)
#preparation du model mathematique basé  sur le concept des arbres binaires
import xgboost as xgb
regressor = xgb.XGBRegressor(
    n_estimators=50,
    reg_lambda=1,
    gamma=0,
    max_depth=3
)
y_train.reset_index(drop=True)
X.shape
#entrainnnement du donnees sur notre model mathematiques pour choisir les meilleurs parametres
regressor.fit(np.array(X_train).reshape(493,1),np.array(y_train).reshape(493,1))
#test and validation
y_pred = regressor.predict(np.array(X_test).reshape(165,1))
print(y_pred)

mean_squared_error(y_test, y_pred)
#cette  fonction prend comme input le valeur  initiale du w et la valeur finale du w  et le nombre du point a creer dans cette intervale
#puis en utilise notre model pour predire la valeur de f qui correspant a chaque valeur dans le vecteur de l'input w 
def calcul(input_r,input_t,start_l,end_l,num_pts):
    input_L=np.linspace(start = start_l, stop = end_l, num = num_pts)
    w=input_L/np.sqrt(input_r*input_t)
    k=np.array(w)
    y_pred = regressor.predict(preprocessing.normalize([np.array(k)]).reshape(num_pts,1))
    return k,y_pred
#exemple:  L varie entre 10 et 100 et  on veut prendre 10 point dans cette intervale equi-espacés
input_w,output_force=calcul(200,1.2,115,6000,10)
print(input_w)

print(output_force)
import matplotlib.pyplot as plt
plt.scatter(input_w,output_force)
plt.plot(input_w,output_force,"gp--")
d = {'w':input_w, 'f':output_force}
df = pd.DataFrame(data=d)
df.to_csv("geniecivile.csv")

