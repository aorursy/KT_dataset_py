# Importul 

import pandas as pd



# Importul datelor folosind biblioteca pandas

df = pd.read_csv('../input/diabetes_data.csv')



# Verificarea datelor daca au fost importate corect

df.head()
# Verificarea numarului de randuri si coloane din setul de date

df.shape
# Crearea unui dataframe cu toate datele de antrenare mai putin datele din coloana rezultat (diabetes)

X = df.drop(columns=['diabetes'])



# Verificarea faptului ca a fost scoasa coloana rezultat (diabetes) din setul de date

X.head()
# Stocarea separata seriei de date rezultat (diabetes) intr-un vector y

y = df['diabetes'].values



# Vizualizarea valorilor seriei rezultat (diabetes)

y[0:5]
# Importul din biblioteca scikitlearn a metodei train_test_split

from sklearn.model_selection import train_test_split



# Separarea setului de date in doua seturi, unul pentru test 920% dintre date) si unul pentru antrenare (80% dintre date)

X_antrenare, X_test, y_antrenare, y_test = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)
# Importul metodei de clasificare prin algoritmul KNN din biblioteca scikit-learn

from sklearn.neighbors import KNeighborsClassifier



# Crearea algoritmului de clasificare KNN

knn = KNeighborsClassifier(n_neighbors = 3)



# Antrenarea modelului de clasificare folosind algoritmul KNN

knn.fit(X_antrenare,y_antrenare)
# Afisarea primelor 5 predictii pentru setul de date de test

knn.predict(X_test)[0:]
# Verificare preciziei modelului pe setul de date de test

knn.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

import numpy as np



# Crearea unui model nou bazat pe algoritmul KNN

knn_cv = KNeighborsClassifier(n_neighbors=3)



# Antrenarea modelului cu valoarea lui cv = 5 

cv_scores = cross_val_score(knn_cv, X, y, cv=5)



# Afisarea fiecarui valorilor cv de acuratete si valoarea medie a acestor scoruri

print(cv_scores)

print('cv_scores mean:{}'.format(np.mean(cv_scores)))
from sklearn.model_selection import GridSearchCV



# Crearea unui nou model de clasificare bazat pe algoritmul KNN

knn2 = KNeighborsClassifier()



# Crearea unui dicionar Python cu toate valorile pe care vrem sa le testam pentru n-vecini (n-neighbors)

param_grid = {'n_neighbors': np.arange(1, 25)}



# Utilizarea functiei gridsearch pentru testarea tuturor valorilor pentru n_vecini

knn_gscv = GridSearchCV(knn2, param_grid, cv=5)



# Potrivurea modelului pe datele noastre

knn_gscv.fit(X, y)
# Verificarea celor mai bune valori pentru n-vecini

knn_gscv.best_params_
# Verificarea scorului mediu pentru cele mai bune valori pentru n-vecini

knn_gscv.best_score_