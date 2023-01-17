#Import delle librerie

import numpy as np

import pandas as pd 

import seaborn as sns 

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics
col_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm', 'Species']

data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv", names = col_names, header=0) 

data.sample(5) #Stampa di alcuni elementi del dataset
data.info()
import seaborn as sns

sns.pairplot( data=data, vars=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'), hue='Species' )
data.describe()
df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

df_norm.sample(n=5)
df_norm.describe()
target = data[['Species']].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])

target.sample(n=5)
df = pd.concat([df_norm, target], axis=1)

df.sample(n=5)
train, test = train_test_split(df, test_size = 0.3)

X_train = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_train = train.Species

X_test = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_test = test.Species
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV



grid = {'solver': ['lbfgs', 'sgd', 'adam'], 'activation': ['identity', 'logistic', 'tanh', 'relu']}

clf_cv = GridSearchCV(MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=(3,3), alpha=1e-5), grid, n_jobs=-1, cv=10)



clf_cv.fit(X_train, y_train)



print("GridSearch():\n")

combinazioni = 1

for x in grid.values():

    combinazioni *= len(x)

print('Per l\'applicazione della GridSearch ci sono {} combinazioni'.format(combinazioni))

print("Migliore configurazione: ",clf_cv.best_params_)

best_config_gs = clf_cv.best_params_

print("Accuracy CV:",clf_cv.best_score_)

ppn_cv = clf_cv.best_estimator_

print('Test accuracy: %.3f' % clf_cv.score(X_test, y_test))

mlp = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=(3,3), alpha=1e-5, **best_config_gs)



mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)

predict_test = mlp.predict(X_test)
#Matrice di confusione e report di classificazione per il Train

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_train,predict_train))

print(classification_report(y_train,predict_train))
#Matrice di confusione e report di classificazione per il Test

print(confusion_matrix(y_test,predict_test))

print(classification_report(y_test,predict_test))