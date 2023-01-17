import pandas as pd

import numpy as np



# Loading dataset

dataset = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
dataset = dataset.dropna(axis=0, thresh=50)

dataset = dataset.dropna(axis=1, thresh=160)



X = dataset.iloc[:, 1:42]

X = X.drop(X.columns[1], axis =1)

y = dataset.iloc[:, 2]
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "median")

imputer = imputer.fit(X.iloc[:, 4:17])

X.iloc[:, 4:17] = imputer.transform(X.iloc[:, 4:17])

imputer = imputer.fit(X.iloc[:, 35:])

X.iloc[:, 35:] = imputer.transform(X.iloc[:, 35:])

detection = X.iloc[:, 18:35]
negDetectedneg, negDetectedpos, posDetectedneg, posDetectedpos = 0, 0, 0, 0

for line in range(191):

    initial_row = 18

    for row in range(17):

        if X.iloc[line,(initial_row + row)] == 'not_detected':

            if y[line] == 0:

                negDetectedneg = negDetectedneg + 1

            else:

                negDetectedpos = negDetectedpos + 1

        elif X.iloc[line,(initial_row + row)] == 'detected':

            if y[line] == 0:

                posDetectedneg = posDetectedneg + 1

            else:

                posDetectedpos = posDetectedpos + 1



print('Negative covid marked as "not_detected": ', negDetectedneg)

print('Positive covid marked as "not_detected": ', negDetectedpos)

print('Negative covid marked as "detected": ', posDetectedneg)

print('Positive covid marked as "detected": ', posDetectedpos)
import math

for col in range(17):

    for line in range(191):

        if type(detection.iloc[line,col]) == float:

            if math.isnan(detection.iloc[line,col]):

                detection.iloc[line,col] = 'not_detected'



for col in range(17):

    X.iloc[:, (18 + col)] = detection.iloc[:, col]
for col in range(17):

    labelencoder_X = LabelEncoder()

    X.iloc[:, (18 + col)] = labelencoder_X.fit_transform(X.iloc[:, (18 + col)])
# Dividindo o dataset em conjunto de treino e conjunto de teste

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



import keras

from keras.models import Sequential

from keras.layers import Dense



# Inicializando a rede neural

classifier = Sequential()



# Adicionando a camada de entrada e a primeira camada escondida

classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 40))



# Adicionando a segunda camada escondida

classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))



# Adicionando a camada de saída

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compilando a rede neural

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Encaixando a rede neural no conjunto de treino

classifier.fit(X_train, y_train, batch_size = 10, epochs = 200)
# Predizendo os resultados do conjunto de teste

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
from sklearn.metrics import accuracy_score

print('Acurácia: %.2f%%' % (accuracy_score(y_test, y_pred)*100))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('\nMatriz de Confusão\n\n',cm)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))