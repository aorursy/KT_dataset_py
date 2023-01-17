import pandas as pd
import numpy as np
import scipy as sc

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings 
warnings.filterwarnings("ignore")
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head()
#Check size of data frame
data.shape
#Check info of data frame
data.info()
sns.set_style("whitegrid")
plt.figure(figsize = (10, 5))
sns.countplot(x = 'DEATH_EVENT', data = data,  palette = "deep")
cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
for i in cols:
    index = cols.index(i)
    plt.subplot(2, 3, index + 1)
    sns.countplot(x = i, data = data, hue="DEATH_EVENT", palette = "deep")
fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
for i in cols:
    index = cols.index(i)
    plt.subplot(2,3,index + 1)
    sns.distplot(data[i])
cols = list(data.columns)

plt.figure(figsize = (10,10))
chart = sns.heatmap(data[cols].corr(), annot = True, cmap = "Blues")
bottom, top = chart.get_ylim()
chart.set_ylim(bottom + 0.5, top - 0.5)
print("Find most important features relative to target")
corr = abs(data.corr())
corr.sort_values(['DEATH_EVENT'], ascending=False, inplace=True)
corr.DEATH_EVENT
X = data[['time', 'serum_creatinine', 'ejection_fraction', 'age', 'serum_sodium']]
Y = data['DEATH_EVENT']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
model1 = GaussianNB()
model1.fit(x_train, y_train)
prediksi_training1 = model1.predict(x_train)
prediksi_testing1 = model1.predict(x_test)

print ("Training: ", accuracy_score(y_train, prediksi_training1))
print ("Testing: ", accuracy_score(y_test, prediksi_testing1))
model2 = RandomForestClassifier()
model2.fit(x_train, y_train)
prediksi_training2 = model2.predict(x_train)
prediksi_testing2 = model2.predict(x_test)

print ("Training: ", accuracy_score(y_train, prediksi_training2))
print ("Testing: ", accuracy_score(y_test, prediksi_testing2))
model3 = GradientBoostingClassifier()
model3.fit(x_train, y_train)
prediksi_training3 = model3.predict(x_train)
prediksi_testing3 = model3.predict(x_test)

print ("Training: ", accuracy_score(y_train, prediksi_training3))
print ("Testing: ", accuracy_score(y_test, prediksi_testing3))
model4 = LogisticRegression()
model4.fit(x_train, y_train)
prediksi_training4 = model4.predict(x_train)
prediksi_testing4 = model4.predict(x_test)

print ("Training: ", accuracy_score(y_train, prediksi_training4))
print ("Testing: ", accuracy_score(y_test, prediksi_testing4))
training1 = accuracy_score(y_train, prediksi_training1)
training2 = accuracy_score(y_train, prediksi_training2)
training3 = accuracy_score(y_train, prediksi_training3)
training4 = accuracy_score(y_train, prediksi_training4)

testing1 = accuracy_score(y_test, prediksi_testing1)
testing2 = accuracy_score(y_test, prediksi_testing2)
testing3 = accuracy_score(y_test, prediksi_testing3)
testing4 = accuracy_score(y_test, prediksi_testing4)


Hasil_Training = [training1, training2, training3, training4]
Hasil_Testing = [testing1, testing2, testing3, testing4];

columns = ['Naive bayes', 'Random Forest', 'Gradient Boosting',  'Logistic Regresi']
Gabungan = {'Model': columns, 'Akurasi Training': Hasil_Training, 'Akurasi Testing': Hasil_Testing}

Hasil_prediksi = pd.DataFrame(data = Gabungan)
Hasil_prediksi