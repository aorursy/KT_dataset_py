# Imports
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.pipeline import *

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# recuperation des fichiers
data = pd.read_csv("/kaggle/input/esaip-data-mining-2020/train_set.csv") # 27K lignes
data_test = pd.read_csv("/kaggle/input/esaip-data-mining-2020/test_set.csv") # 3K lignes

# Mise en place des variables X et Y (ce que nous voulons prédire) 
y = data['DEFAULT'] 
x = data.drop(['DEFAULT'], axis=1)
# graphiques avec mise en valeur de la target
sns.catplot("DEFAULT", col="SEX", data=data, kind="count")
sns.catplot("DEFAULT", col="EDUCATION", data=data, kind="count", col_wrap=2)
sns.catplot("DEFAULT", col="MARRIAGE", data=data, kind="count", col_wrap=2)


sns.distplot(data["LIMIT_BAL"])
sns.pairplot(data, vars=["AGE", "SEX", "LIMIT_BAL"], hue="DEFAULT")
sns.heatmap(data.corr())
# Pas d"analyse particulère sur des relations ou visible à l"oeil sur ces grpahiques.
# Peu de cas de défaut de payement constaté
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # la règle du 80/20
print(x_train.shape) # train
print(x_test.shape) # test

# Comparaison des modèles
model = KNeighborsClassifier()
sgd = SGDClassifier()
randomForest = RandomForestClassifier()
# Model sans optimisation
# Attention il s'agit d'un premier test qui comporte un biais car
# il s'entraine et s'evalue sur les mêmes données 
model.fit(x_train, y_train)
model.score(x_test, y_test)
# Exploration d'autres algorythmes possible 
sgd.fit(x_train, y_train)
sgd.score(x_test,y_test)
randomForest.fit(x_train, y_train)
randomForest.score(x_test, y_test)
model_split = KNeighborsClassifier()
model_split.fit(x_train, y_train)
model = KNeighborsClassifier()
k = np.arange(1,30)

train_score, val_score = validation_curve(model, x_train, y_train, 'n_neighbors', k, cv=5)


plt.plot(k, val_score.mean(axis=1), label="validation")
plt.plot(k, train_score.mean(axis=1), label="train")
plt.legend()

# Nous pouvons observer ci dessus que nous convergeons vers un score 
# aux alentours de 80% de prédiction

# Params
model = RandomForestClassifier()
k = np.arange(100, 120 )

train_score, val_score = validation_curve(model, x_train, y_train, 'n_estimators', k, cv=10)

plt.plot(k, val_score.mean(axis=1), label="validation")
plt.plot(k, train_score.mean(axis=1), label="train")
plt.legend()
## pas encore compris ce score, comment le train peut rester à 100% ? 
param = {"n_neighbors": np.arange(1, 20), "metric": ['euclidean', 'manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param, cv=5)
grid.fit(x_train, y_train)
# parametres optimisés pour le model
grid.best_estimator_
# Meilleurs score optenu => env + 2%  
grid.best_score_

model = grid.best_estimator_
N, train, val = learning_curve(model, x_train, y_train, train_sizes=np.linspace(0.1, 1, 20), cv=5)

plt.plot(N, train.mean(axis=1), label='train')
plt.plot(N, val.mean(axis=1), label='val')
plt.legend()
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
scaler =  RobustScaler()
x_robust = scaler.fit_transform(x_train)
x_robust_test = scaler.fit_transform(x_test)
model = KNeighborsClassifier(n_neighbors=16, metric="manhattan")
model.fit(x_robust, y_train)
scaler =  RobustScaler()
x_robust = scaler.fit_transform(x_train)
x_robust_test = scaler.fit_transform(x_test)
model = KNeighborsClassifier(n_neighbors=16, metric="manhattan")
model.fit(x_robust, y_train)
# test avec la random forest

randomForest.fit(x_robust, y_train)
randomForest.score(x_robust_test, y_test)
model.score(x_robust_test, y_test)
# miminise les données absurde (pas dans notre cas)
# Réutilisation des paramêtres du Grids search CV 
# amélioration de + 2% environ
minMaxScaler = MinMaxScaler()
stdScaler = StandardScaler()


x_minmax_train = minMaxScaler.fit_transform(x_train)
x_minmax_test = minMaxScaler.fit_transform(x_test)

x_std_test = stdScaler.fit_transform(x_test)
x_std_train = stdScaler.fit_transform(x_train)

modelStd = KNeighborsClassifier(n_neighbors=16, metric="manhattan")
modelMinMax = KNeighborsClassifier(n_neighbors=16, metric="manhattan")
modelStd.fit(x_std_train, y_train)
modelMinMax.fit(x_minmax_train, y_train)
modelStd.score(x_std_test, y_test)
modelMinMax.score(x_minmax_test, y_test)
from sklearn.pipeline import *
model = make_pipeline(RobustScaler(), KNeighborsClassifier())
model
params = {
    "kneighborsclassifier__n_neighbors": [21, 25, 30 ], # choix après plusieurs essais 
    "kneighborsclassifier__weights" : ["uniform", "distance"], 
    "kneighborsclassifier__algorithm" : ["ball_tree", "kd_tree", 'brute'],
    "kneighborsclassifier__metric": ['manhattan', 'euclidean']
}
grid = GridSearchCV(model, param_grid=params, cv=10 )
grid.get_params().keys()
grid.fit(x_train, y_train)
grid.best_score_
# +1% de prédiction gagné sur le model
# 0.812 k=21
grid.best_params_
# Avec la normalisation des données les paramètres optimisés ont changés
best_knn_model = grid.best_estimator_
# Application de la meilleur configuration de pipeline au modèle
best_knn_model.score(x_test, y_test)
# test du model sur des données inconnues
# score proche du best_score du grid
prediction = best_knn_model.predict(data_test) 
plt.hist(prediction)
prediction = pd.DataFrame(columns=['DEFAULT'], data=prediction)
data_export= pd.concat([data_test['ID'], prediction], 1) 
data_export.head(14)
data_export.to_csv("predictions.csv")
from sklearn.pipeline import *
forest = RandomForestClassifier()
max_depth = [5, 8, 15, 25, 30]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict( max_depth = max_depth,
              min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3)
gridF.fit(x_train, y_train)
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier
model_4 = VotingClassifier([
    ('KNN_simple', KNeighborsClassifier()),  # utilisation de plusieurs modèles variés
    ('RF', randomForest), 
    ('KNN', model)
], voting="hard")
model_4.fit(x_train, y_train)
model_4.score(x_test, y_test)
pipeline_2 = make_pipeline(RobustScaler(), model_4)
pipeline_2.fit(x_train, y_train)
pipeline_2.score(x_test, y_test)
forest = RandomForestClassifier()
max_depth = [5, 8, 15, 25, 30]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict( max_depth = max_depth,
              min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3)
gridF.fit(x_train, y_train)
gridF.score(x_test, y_test)
pipelineGrid = make_pipeline(RobustScaler(), gridF)
pipelineGridMM = make_pipeline(MinMaxScaler(), gridF)
pipelineGridStd = make_pipeline(StandardScaler(), gridF)
pipelineGrid.fit(x_train, y_train)
pipelineGridMM.fit(x_train, y_train)
pipelineGridStd.fit(x_train, y_train)
print(pipelineGrid.score(x_test, y_test))
print(pipelineGridMM.score(x_test, y_test))
print(pipelineGridStd.score(x_test, y_test))
voting = StackingClassifier([
    ("p1", pipelineGrid), 
    ("p2", pipelineGridMM), 
    ("p3", pipelineGridStd), 
], final_estimator=RandomForestClassifier(n_estimators=500))
voting.fit(x_train, y_train)
voting.score(x_test, y_test)
# Ancien score 0.908 => sur le train attention 
# 0.815 test 
prediction = voting.predict(data_test) 
prediction = pd.DataFrame(columns=['DEFAULT'], data=prediction)
data_export= pd.concat([data_test['ID'], prediction], 1) 
data_export.set_index("ID", inplace=True)
data_export.to_csv("sample.csv")
BaggingClassifiergClassifiergClassifieringClassRandomForestClassifierdomForestClassifiereringClassifieragging= BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=100)
modelBagginggging.fit(x_train, y_train)
modelBagging.score(x_test, y_test)
voting = VotingClassifier([
    ("p1", pipelineGrid), 
    ("p2", pipelineGridMM), 
    ("p3", pipelineGridStd), 
], voting='hard')
voting.fit(x_train, y_train)
voting.score(x_test, y_test)
#0815
prediction = voting.predict(data_test) 
prediction = pd.DataFrame(columns=['DEFAULT'], data=prediction)
data_export= pd.concat([data_test['ID'], prediction], 1) 
data_export.set_index("ID", inplace=True)
data_export.to_csv("sample.csv")
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
# A utility method to create a tf.data dataset from a Pandas Dataframe
# code trouvé sur la doc tensorflow pour convrtir un dataframe.
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('DEFAULT')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def dt_to_test_df(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
# découpe en train et test set
train, test = train_test_split(data, test_size=0.2) # règle du 80/20.


feature_columns = []

# numeric cols et selection des features
for header in [ 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6' ]:
  feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# decoupages des sets
# pour bien séparer les valeurs et pas overfit
batch_size = 10
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
predict = dt_to_test_df(data_test)
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(4, activation='sigmoid'),
  layers.Dense(12, activation='relu'),
  layers.Dense(1, activation="sigmoid")
])
# recette secrete ne pas divulguer, j'ai empiler les couches comme j'ai pu
model.compile(optimizer='SGD',
              loss='mean_squared_error', # j'ai appris qu'elle etait bien  
              metrics=['accuracy']) # optimisation pour les réponses

model.fit(train_ds,
          validation_data=val_ds,
          epochs=4)
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
prediction = model.predict(predict)
prediction.shape
pos = 0
for i in prediction:
    if i > 0 :
        pos += 1
    
print(pos)
()