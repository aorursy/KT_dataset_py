# Importul bibliotecilor Numpy, Matplotlib și SciKit-Learn - sklearn

import matplotlib.pyplot as plt

import numpy as np



# Importul obiectelor datasets și linear_model pentru regresie din biblioteca sklearn

from sklearn import datasets, linear_model



# Încarcarea seturilor de date

pretul_casei = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]

dimensiune1 = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]

print(dimensiune1)



# Redimensionarea datelelor de input - variabilele independente pentru regresie

dimensiune2 = np.array(dimensiune1).reshape((-1,1))

print(dimensiune2)
# Utilizând modulul de potrivire (model fit) pentru regresia liniară, utilizatorii pot modela datele frecvent si rapid

regr = linear_model.LinearRegression()

regr.fit(dimensiune2, pretul_casei)

print("Coeficienti: \n", regr.coef_)

print("Termenul liber: \n", regr.intercept_)
# Introducem o noua valoare pentru variabila independenta dimensiune ca sa prezicem care va fi pretul casei

dimensiune_noua = 1400

pret = (dimensiune_noua * regr.coef_) + regr.intercept_

print(pret)

print(regr.predict([[dimensiune_noua]]))
# Formula obtinută pentru modelul antrenat

def graph(formula, x_range):

    x = np.array(x_range)

    y = eval(formula)

    plt.plot(x, y)



# Trasarea liniei de predictie peste punctele de date inițiale

graph('regr.coef_*x + regr.intercept_', range(1000, 2700))

plt.scatter(dimensiune2, pretul_casei, color='black')

plt.ylabel('Prețul casei')

plt.xlabel('Dimensiunea casei')

plt.show()