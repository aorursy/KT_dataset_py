#Import Library 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings ("ignore")
#Load Dataset
data = pd.read_csv('../input/bank-marketing-dataset/bank.csv')
data.head(3)
print ("Jumlah baris adalah ", data.shape[0])
print ("Jumlah kolom adalah ", data.shape[1])
#Cek info dari data set
data.info()
data.describe()
#Cek Setiap nilai unik pada kolom kategori

kolom_kategori = data.dtypes[data.dtypes == 'object'].index

for subscript in kolom_kategori:
    print ("Nama kolom: ", subscript);
    print (data[subscript].unique());
    print ("--" * 50);
#Cek perbandingan target
target = data.deposit.value_counts().rename_axis('Name').reset_index(name='Frekuensi')
fig = px.bar(target, x = "Name", y= "Frekuensi", color = "Frekuensi", title = "Perbandingan Hasil Marketing Campaign")
fig.show()
#Melakukan pelabelan pada variabel prediktor default, housing, loan dan deposit
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cols = ['default', 'housing', 'loan', 'deposit']

for subscript1 in cols:
    data[subscript1] = le.fit_transform(data[subscript1])
#Pisahkan variable prediktor (independent) dan variable target (dependent)
prediktor = pd.get_dummies(data.drop('deposit', axis = 1))
target = data['deposit']
#Pisahkan data training dan data testing, dengan porsi 80:20
X_train, X_test, y_train, y_test = train_test_split (prediktor, target, test_size = 0.2, random_state = 0)
model1 = GaussianNB()
model1.fit(X_train, y_train)
predicted_train = model1.predict(X_train)
predicted_test = model1.predict(X_test)
print ("Training:", accuracy_score(y_train, predicted_train))
print ("Testing:", accuracy_score(y_test, predicted_test))
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
predicted_train2 = model2.predict(X_train)
predicted_test2 = model2.predict(X_test)
print ("Training:", accuracy_score(y_train, predicted_train2))
print ("Testing:", accuracy_score(y_test, predicted_test2))
model3 = LogisticRegression()
model3.fit(X_train, y_train)
predicted_train3 = model3.predict(X_train)
predicted_test3 = model3.predict(X_test)
print ("Training:", accuracy_score(y_train, predicted_train3))
print ("Testing:", accuracy_score(y_test, predicted_test3))
model5 = GradientBoostingClassifier()
model5.fit(X_train, y_train)
predicted_train5 = model5.predict(X_train)
predicted_test5 = model5.predict(X_test)
print ("Training: ", accuracy_score(y_train, predicted_train5))
print ("Testing: ", accuracy_score(y_test, predicted_test5))
Hasil = {'Model': ['Naive Bayes Classifier', 'Random Forest Classifier', 'Logistic Regression', 'Gradient Boosting'],\
        'Training': [accuracy_score(y_train, predicted_train), accuracy_score(y_train, predicted_train2), accuracy_score(y_train, predicted_train3), accuracy_score(y_train, predicted_train5)],\
        'Testing': [accuracy_score(y_test, predicted_test), accuracy_score(y_test, predicted_test2), accuracy_score(y_test, predicted_test3), accuracy_score(y_test, predicted_test5)]}
Hasil = pd.DataFrame(data = Hasil)
Hasil
fig = go.Figure(data=[
    go.Bar(name='Training', x= Hasil.Model, y=Hasil.Training),
    go.Bar(name='Testing', x=Hasil.Model, y=Hasil.Testing)
])
# Change the bar mode
fig.update_layout(barmode='group', title = "Hasil Modeling",  xaxis_title='Model',yaxis_title='Akurasi')
fig.show()