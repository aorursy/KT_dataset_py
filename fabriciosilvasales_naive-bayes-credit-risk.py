import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base = pd.read_csv("/kaggle/input/credit_data.csv")
base['default'].value_counts()
m = np.mean(base['age'])



base.loc[base.age < 0, 'age'] = m

base.loc[base.age.isna() , 'age'] = m
previsores = base.iloc[:,1:4].values

classe = base.iloc[:, 4].values
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

previsores = scaler.fit_transform(previsores)
from sklearn.model_selection import train_test_split



previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB



classificador = GaussianNB()

classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
from sklearn.metrics import accuracy_score, plot_confusion_matrix

import matplotlib.pyplot as plt



plot_confusion_matrix(classificador, previsores_teste, classe_teste,display_labels=["Mau Pagador","Bom Pagador"])



precisao = accuracy_score(classe_teste, previsoes)



print("Acurácea do Modelo: "+str(precisao*100)+"%")



plt.show()