# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot

from sklearn.linear_model import Ridge # ridge

from sklearn.preprocessing import PolynomialFeatures # PolynomialFeatures, per costruire equazioni di grado superiore al primo

from sklearn.pipeline import make_pipeline # make_pipeline, per costruire equazioni di grado superiore al primo





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris = pd.read_csv('../input/iris/Iris.csv', nrows=1500, usecols=[0, 3], encoding='latin-1') # creazione del dataframe con campi identifivo del fiore (Id) e lunghezza del petalo in cm (PetalLengthCm)

iris.head(10) # stampa delle prime 10 righe del dataframe
plt.rcParams['figure.figsize'] = [15, 10]



plt.xlabel('Id flowers')

plt.ylabel('Petal length (cm)')

plt.scatter(iris.Id, iris.PetalLengthCm, marker='o', color='black')

plt.show()
# Applicazione della Ridge Regression con alpha uguale a 0, su equazione di primo grado



length_petal = list(iris.PetalLengthCm)



x_train = []

for y in iris.index:

    x_train.append([y])

    

ridg = Ridge(alpha=0) # riduce al minimo la funzione obiettivo: ||y - Xw||^2_2 + alpha * ||w||^2_2 con alpha a 0

ridg.fit(x_train, length_petal) # linearizza il modello

prediction = ridg.predict(x_train) # effettua la previsione utilizzando il modello lineare 



# disegno del plot

plt.xlabel('Id flowers')

plt.ylabel('Petal Length (cm)')

plt.scatter(iris.Id, iris.PetalLengthCm, marker='o', color='black')

plt.plot(iris.Id, prediction, color='orange', linewidth='1')

plt.show()
# alpha uguale a 1

ridg1 = Ridge(alpha=1) # riduce al minimo la funzione obiettivo: ||y - Xw||^2_2 + alpha * ||w||^2_2 con alpha a 1

ridg1.fit(x_train, length_petal) # linearizza il modello

prediction1 = ridg1.predict(x_train) # effettua la previsione utilizzando il modello lineare



# alpha uguale a 100

ridg2 = Ridge(alpha=100) # riduce al minimo la funzione obiettivo: ||y - Xw||^2_2 + alpha * ||w||^2_2 con alpha a 100

ridg2.fit(x_train, length_petal) # linearizza il modello

prediction2 = ridg2.predict(x_train) # effettua la previsione utilizzando il modello lineare





# disegno plot

plt.xlabel('Id flowers')

plt.ylabel('Petal Length (cm)')

plt.scatter(iris.Id, iris.PetalLengthCm, marker='o', color='black')

plt.plot(iris.Id, prediction, color='orange', linewidth='1')

plt.plot(iris.Id, prediction1, 'r--', color='purple')

plt.plot(iris.Id, prediction2, '.', color='green')

plt.legend(['alpha = 0','alpha = 1','alpha = 100'], numpoints=1)

plt.show()



# alpha uguale a 50000

ridg3 = Ridge(alpha=50000) # riduce al minimo la funzione obiettivo: ||y - Xw||^2_2 + alpha * ||w||^2_2 con alpha a 50000

ridg3.fit(x_train, length_petal) # linearizza il modello

prediction3 = ridg3.predict(x_train) # effettua la previsione utilizzando il modello lineare



# alpha uguale a 1000000

ridg4 = Ridge(alpha=1000000) # riduce al minimo la funzione obiettivo: ||y - Xw||^2_2 + alpha * ||w||^2_2 con alpha a 1000000

ridg4.fit(x_train, length_petal) # linearizza il modello

prediction4 = ridg4.predict(x_train) # effettua la previsione utilizzando il modello lineare







# disegno plot

plt.xlabel('Id flowers')

plt.ylabel('Petal Length (cm)')

plt.scatter(iris.Id, iris.PetalLengthCm, marker='o', color='black')

plt.plot(iris.Id, prediction, color='orange', linewidth='1')

plt.plot(iris.Id, prediction3, 'r--', color='blue')

plt.plot(iris.Id, prediction4, 'r--', color='pink')

plt.legend(['alpha = 0','alpha = 50000','alpha = 1000000'], numpoints=1)

plt.show()
# polinomio di secondo grado



alphas = [0, 50000, 1000000] # valore di alpha, mi baso sui valori provati con la retta

prediction = []

for n in range(0, 3): # per i 3 valori di alpha

    # PolynomialFeatures: genera caratteristiche polinomiali e di interazione (combinazione dei coefficienti da grado 0 a 2)

    # make_pipeline: costruisce una pipeline dagli stimatori dati

    ridg = make_pipeline(PolynomialFeatures(2), Ridge(alpha=alphas[n])) 

    ridg.fit(x_train, length_petal)

    prediction.append(ridg.predict(x_train))



# disegno plot

plt.xlabel('Id flowers')

plt.ylabel('Petal Length (cm)')

plt.scatter(iris.Id, iris.PetalLengthCm, marker='o', color='black')

plt.plot(iris.Id, prediction[0], color='orange', linewidth='1')

plt.plot(iris.Id, prediction[1], 'r--', color='blue')

plt.plot(iris.Id, prediction[2], 'r--', color='pink')

plt.legend(['alpha = 0','alpha = 50000','alpha = 1000000'], numpoints=1)

plt.show()
# Confronto fra le equazioni polinomiali di 1, 2 e 4 grado con alfa 50 000





alpha = 50000

prediction = []

degree_eq = [1,2,4]

for n in degree_eq: # iterazione sui gradi dell'equazione

    ridg = make_pipeline(PolynomialFeatures(n), Ridge(alpha)) 

    ridg.fit(x_train, length_petal)

    prediction.append(ridg.predict(x_train))



# disegno plot

plt.xlabel('Id flowers')

plt.ylabel('Petal Length (cm)')

plt.scatter(iris.Id, iris.PetalLengthCm, marker='o', color='black')

plt.plot(iris.Id, prediction[0], color='green', linewidth='1')

plt.plot(iris.Id, prediction[1], 'r--', color='orange')

plt.plot(iris.Id, prediction[2], color='purple', linewidth='1')

plt.legend(['equazione di grado 1','equazione di grado 2', 'equazione di grado 4'], numpoints=1)

plt.show()
# polinomio di quarto grado



alphas = [0, 50000, 1000000] # valore di alpha, mi baso sui valori provati con la retta

prediction = []

for n in range(0, 3): # per i 3 valori di alpha

    ridg = make_pipeline(PolynomialFeatures(4), Ridge(alpha=alphas[n])) 

    ridg.fit(x_train, length_petal)

    prediction.append(ridg.predict(x_train))



# disegno plot

plt.xlabel('Id flowers')

plt.ylabel('Petal Length (cm)')

plt.scatter(iris.Id, iris.PetalLengthCm, marker='o', color='black')

plt.plot(iris.Id, prediction[0], color='orange', linewidth='1')

plt.plot(iris.Id, prediction[1], 'r--', color='blue')

plt.plot(iris.Id, prediction[2], 'r--', color='pink')

plt.legend(['alpha = 0','alpha = 50000','alpha = 1000000'], numpoints=1)

plt.show()



# dal quinto grado in poi il risultato non e' piu' accurato. Fa pensare che il problema sia mal condizionato