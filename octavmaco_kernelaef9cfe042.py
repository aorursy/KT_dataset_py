# Importul bibliotecilor de lucru

import numpy as np

import pandas as pd

from sklearn import svm



import matplotlib.pyplot as plt

import seaborn as sns; sns.set(font_scale=1.2)

%matplotlib inline
retete = pd.read_csv("../input/Briose_vs_Tarte.csv")

print(retete.head())
sns.lmplot('Faina', 'Zahar', data=retete, hue='Tip', palette='Set1', fit_reg=False, scatter_kws={"s": 70});
tip_label = np.where(retete['Tip']=='Briosa', 0, 1)

retete_caracteristici = retete.columns.values[1:].tolist()

retete_caracteristici

ingrediente = retete[['Faina','Zahar']].values

print(ingrediente)
# Antrenarea modelului

model = svm.SVC(kernel='linear')

model.fit(ingrediente, tip_label)
# Obținerea hiperplanurilor de separație

w = model.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(30,60)

yy = a * xx - (model.intercept_[0]) / w[1]



# Reprezentarea grafică a paralelor la hiperplan care trec prin vectorii suport

b = model.support_vectors_[-0]

yy_down = a * xx + (b[1] - a * b[0])

b = model.support_vectors_[-1]

yy_up = a * xx + (b[1] - a * b[0])

sns.lmplot('Faina', 'Zahar', data=retete, hue='Tip', palette='Set1', fit_reg=False, scatter_kws={"s": 70});

plt.plot(xx, yy, linewidth=2, color='black')

plt.plot(xx, yy_down, 'k--')

plt.plot(xx, yy_up, 'k--')
#crearea unei functii pentru a prezice daca o prajitura este briosa sau tarta

def briosa_sau_tarta(faina, zahar):

    if(model.predict([[faina, zahar]]))==0:

        print('Te uiti la o reteta de briose!')

    else:

        print("Te uiti la o reteta de tarte!")

        

#Facem o predictie pentru o cantiatate de 50% faina si 20% zahar

briosa_sau_tarta(50, 20)
#Sa desenam pe grafic reteta nou introdusa

sns.lmplot('Faina', 'Zahar', data=retete, hue='Tip', palette='Set1', fit_reg=False, scatter_kws={"s": 70});

plt.plot(xx, yy, linewidth=2, color='black')

plt.plot(50,20, 'yo', markersize='9')