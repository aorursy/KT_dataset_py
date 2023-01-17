# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Memuat file drug200.csv menjadi pandas dataframe

dataframe = pd.read_csv('../input/drug-classification/drug200.csv')
# Menampilkan 5 baris pertama dari dataframe

dataframe.head()
# Menampilkan informasi dari dataframe

dataframe.info()
dataframe_int = dataframe.copy()
from sklearn.preprocessing import LabelEncoder
# Membuat objek/instance yang bernama encoder

encoder = LabelEncoder()
categorical_data = ['Sex','BP','Cholesterol','Drug']
for kolom in categorical_data:

    dataframe_int[kolom] = encoder.fit_transform(dataframe[kolom])
# Sekarang data sudah berupa angka sepenuhnya

dataframe_int.head()
dataframe_int.info()
for kolom in categorical_data:

    print(kolom,dataframe_int[kolom].unique())
for kolom in categorical_data:

    print(kolom,dataframe[kolom].unique())
# Menampilkan matrix korelasi antar kolom

dataframe_int.corr()
# Untuk membantu melakukan analisa, akan lebih nyaman jika dilakukan visualisasi data

plt.figure(figsize=(10,8))

plt.title('Matrix Korelasi Data')

sns.heatmap(dataframe_int.corr(),annot=True,linewidths=3)

plt.show()

plt.savefig('Matrix Korelasi Data')
def distribusi():

    fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12,8))

    plt.suptitle('Distribusi',fontsize=24)

    

    def kolom_generator():

        for kolom in dataframe_int:

            yield kolom

    kolom = kolom_generator()



    for i in range(0,2):

        for j in range(0,3):

            k = next(kolom)

            dataframe_int[k].plot(kind='hist',ax=axes[i,j])

            axes[i,j].set_title(k)

    plt.savefig('Distribusi Data.png')

    plt.show()
distribusi()
data = dataframe_int.drop('Drug',axis=1)

label = dataframe_int['Drug']
data.head()
label.head()
# Kita dapat memisahkan data menjadi data latihan dan data tes dengan train_test_split yang terdapat pada module Sklearn

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,label,test_size=0.2)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
from sklearn.svm import SVC
# Membuat objek/instance dengan nama model

model = SVC(gamma='scale')
# Melatih model dengan data latihan

model.fit(x_train,y_train)
# Membuat prediksi terhadap data tes

prediction = model.predict(x_test)
prediction
# Menampilkan akurasi prediksi model

model.score(x_test,y_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,r2_score
def display_conf(y_test,prediction,filename):

    sns.heatmap(confusion_matrix(y_test,prediction),annot=True,linewidths=3,cbar=False)

    plt.title('Confusion Matrix')

    plt.ylabel('Actual')

    plt.xlabel('Prediction')

    plt.savefig(filename)

    plt.show()
display_conf(y_test,prediction,'ConfusionMatrix-0.png')
r2_score(y_test,prediction)
print(classification_report(y_test,prediction))
from sklearn.model_selection import GridSearchCV
SVC(gamma='scale').get_params()
param_grid = {'C':[0.01,0.1,1,10,100],

              'gamma':[100,10,1,0,1,0.01]}
best_model = GridSearchCV(SVC(),param_grid,cv=5,refit=True)
best_model.fit(x_train,y_train)
# Model dengan parameter terbaik

best_model.best_estimator_
# Membuat prediksi dengan model yang telah ditingkatkan

prediction = best_model.predict(x_test)
# Menampilkan confusion matrix pada prediksi yang baru

display_conf(y_test,prediction,'ConfusionMatrix-1.png')
r2_score(y_test,prediction)
print(classification_report(y_test,prediction))
# Menampilkan data yang sebenarnya

encoder.inverse_transform(np.array(y_test))
# Menampilkan data yang diprediksi oleh model

encoder.inverse_transform(prediction)
# Menampilkan format data

x_train.head(1)
def self_prediction():

    age = input('Age : ')

    sex = input('Sex : ')

    bp = input('BP : ')

    chol = input('Cholesterol : ')

    NatoK = input('Na_to_K : ')

    

    # data harus berbentuk (1,5) yaitu [[age,sex,bp,chol,NatoK]]

    print('\nPrediction')

    print('Patient consumed : ',encoder.inverse_transform(best_model.predict([[age,sex,bp,chol,NatoK]]))[0])
self_prediction()
import pickle
with open('AI_DrugClassifier.pkl','wb') as file:

    pickle.dump(best_model,file)
with open('AI_DrugClassifier.pkl','rb') as file:

    ml_model = pickle.load(file)
ml_model.best_estimator_
submission = pd.DataFrame()

submission['Actual'] = y_test

submission['Prediction'] = prediction
submission.head()
submission.to_csv('Submission.csv')