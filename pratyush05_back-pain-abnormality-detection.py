import numpy as np

import pandas as pd
data = pd.read_csv('../input/Dataset_spine.csv')
data.head()
data.drop(['Unnamed: 13'], axis=1, inplace=True)



data.head()
data['Class_att'] = data['Class_att'].map({'Abnormal': 1, 'Normal': 0})



data.head()
data = data.rename(columns={'Col1': 'pelvic_incidence', 

                            'Col2': 'pelvic_tilt', 

                            'Col3': 'lumbar_lordosis_angle', 

                            'Col4': 'sacral_slope', 

                            'Col5': 'pelvic_radius', 

                            'Col6': 'degree_spondylolisthesis', 

                            'Col7': 'pelvic_slope', 

                            'Col8': 'direct_tilt', 

                            'Col9': 'thoracic_slope', 

                            'Col10': 'cervical_tilt', 

                            'Col11': 'sacrum_angle', 

                            'Col12': 'scoliosis_slope', 

                            'Class_att': 'class'})
data.head()
data.info()
data.describe()
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')
plt.figure(figsize=(12,9))

sns.heatmap(data.corr(), annot=True)
sns.pairplot(data, hue='class', palette='Set1')
sns.countplot(x='class', data=data, palette='Set2')
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
scaler = StandardScaler()



y = data['class'].values

X = scaler.fit_transform(data[data.columns[:-1]])
var = []

for n in range(1, 12):

    pca = PCA(n_components=n)

    pca.fit(X)

    var.append(np.sum(pca.explained_variance_ratio_))
plt.figure(figsize=(10,6))

plt.plot(range(1,12), var, color='red', linestyle='dashed', marker='o', markerfacecolor='black', markersize=10)

plt.title('Variance vs. Components')

plt.xlabel('Components')

plt.ylabel('Variance')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
from tpot import TPOTClassifier
pipeline = TPOTClassifier(generations=20, population_size=100, cv=5, n_jobs=-1, random_state=101, verbosity=2)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
from sklearn.metrics import classification_report, confusion_matrix
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
import keras

from keras.layers import Dense, Dropout

from keras.models import Sequential
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(12,)))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=32, epochs=1000, verbose=2, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])