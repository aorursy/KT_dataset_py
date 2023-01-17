import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine = pd.read_csv('../input/wine.data.txt', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])

wine.describe().transpose().drop(['mean', '25%', '50%', '75%'], axis=1)

X = wine.drop('Cultivator',axis=1)
y = wine['Cultivator']
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=500)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print( confusion_matrix(y_test, predictions) )
print('----------------------------------------------------')
print(classification_report(y_test, predictions))
wine[100:].head(1)

Alchol = 12.05
Malic_Acid = 5.73
Ash = 1.04
Alcalinity_of_Ash = 17.4
Magnesium = 92
Total_phenols = 2.72
Falvanoids = 5.16
Nonflavanoid_phenols = 0.26
Proanthocyanins = 1.35
Color_intensity = 13.2
Hue = 1.12
OD280 = 2.91
Proline = 478

# Влияние признаков на результат классификации
# [0.     0.02058824  0.    0.                 0.         0.             0.41409342  0.                    0.               0.41253197       0.   0.     0.15278638]
# Alchol, Malic_Acid, Ash,  Alcalinity_of_Ash, Magnesium, Total_phenols, Falvanoids, Nonflavanoid_phenols, Proanthocyanins, Color_intensity, Hue, OD280, Proline

newWine_X = [[Alchol, Malic_Acid, Ash, Alcalinity_of_Ash, Magnesium, Total_phenols, Falvanoids, Nonflavanoid_phenols, Proanthocyanins, Color_intensity, Hue, OD280, Proline]];
newWine_X = scaler.transform(newWine_X)
result = mlp.predict(newWine_X);
print('Сорт вина: ' + str(result[0]))