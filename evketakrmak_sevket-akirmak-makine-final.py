import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

import plotly.offline as ply

ply.init_notebook_mode(connected=True)

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")

data.head()
data.rename(columns={"Last_Update":"Tarih","Confirmed":"Vaka","Deaths":"Vefat","Recovered":"Tedavi_Edilen"},inplace=True)





vaka_orani = [0]

olum_orani = [0]

vaka_artisi = [0]

olum_artisi = [0]

test_orani = [0]

test_artisi = [0]



aktif_hasta = data["Vaka"]-(data["Vefat"]+data["Tedavi_Edilen"])

pasif_hasta = data["Vefat"]+data["Tedavi_Edilen"]





for i in range(len(data)-1):

    

    vakaArtisi = data["Vaka"][i+1] - data["Vaka"][i]

    

    vakaOrani = round((data["Vaka"][i+1]-data["Vaka"][i])/

                     data["Vaka"][i],2)

    

    olumOrani = round((data["Vefat"][i+1] - data["Vefat"][i])/

                     data["Vefat"][i],2)

    

    olumArtisi = data["Vefat"][i+1] - data["Vefat"][i]

        

        

    vaka_artisi.append(vakaArtisi)

    vaka_orani.append(vakaOrani)

    olum_orani.append(olumOrani)

    olum_artisi.append(olumArtisi)

    

    

    

    

data["Vaka Artış Sayısı"] = vaka_artisi

data["Vaka Artış Oranı"] = vaka_orani

data["Vefat Artış Sayısı"] = olum_artisi

data["Vefat Artış Oranı"] = olum_orani

data["Aktif Hasta Sayısı"] = aktif_hasta

data["Pasif Hasta Sayısı"] = pasif_hasta



    





data.fillna(0, inplace=True)

data = data.replace([np.inf,-np.inf], np.nan)

data.fillna(0, inplace=True)
data
olum = go.Scatter(

    x = data.Tarih,

    y = data.Vefat,

    mode = "lines+markers",

    name = "Vefat",

    marker = dict(color = 'rgba(255, 0, 0, 0.8)'),

    text = data.Vefat

)



tedavi = go.Scatter(

    x = data.Tarih,

    y = data.Tedavi_Edilen,

    mode = "lines+markers",

    name = "Tedavi Edilen",

    marker = dict(color = "rgba(0, 180, 0, 0.8)"),

    text = data.Tedavi_Edilen

)



data2 = [olum,tedavi]



layout = dict(

    title = "Toplam Ölüm Ve Toplam Tedavi Sayıları",

    xaxis = dict(title = "Tarih"),

    yaxis = dict(title = "Kişi Sayısı"),

    xaxis_tickangle = -45,

             )



fig = dict(data = data2, layout = layout)

iplot(fig)
#Vakalar hariç tüm sutunları seçiyoruz.

x_cols = [x for x in data.columns if (x == 'Vefat' or x == 'Tedavi_Edilen')]



X_data = data[x_cols]

y_data = data['Vaka']





#Ardından vaka sayılarına bakarak kaç kişinin tedavi edileceğini ön görmeye çalışyoruz:

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1)



knn = knn.fit(X_data, y_data)



y_pred = knn.predict(X_data)





def accuracy(real, predict):

    return sum(real == predict) / float(real.shape[0])



print(accuracy(y_data, y_pred))
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score





myList = list(range(1,40))



scores = []







for k in myList:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn = knn.fit(X_data, y_data)

    y_pred = knn.predict(X_data)

    def accuracy(real, predict):

        return sum(real == predict) / float(real.shape[0])

    scores.append(accuracy(y_data, y_pred))

    

plt.plot(myList, scores)

plt.xlabel('Neighbors K Sayisi')

plt.ylabel('Yanlis Siniflandirma Hatası')

plt.show()
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB



x_cols = [x for x in data.columns if (x == 'Vefat' or x == 'Tedavi_Edilen')]



X_data = data[x_cols]

y_data = data['Vaka']

cv_N = 2



nb = {'gaussian': GaussianNB(),

      'bernoulli': BernoulliNB(),

      'multinomial': MultinomialNB()}

scores = {}



for key, model in nb.items():

    s = cross_val_score(model, X_data, y_data, cv=cv_N, n_jobs=cv_N, scoring='accuracy')

    scores[key] = np.mean(s)



scores
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.8, random_state=0)



mnb = MultinomialNB()

y_pred = mnb.fit(X_train, y_train).predict(X_test)



print("Toplam nokta: %d Hatalı Nokta : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print("Doğruluk oranı: ", np.mean(y_pred == y_test))
from sklearn.kernel_approximation import Nystroem

from sklearn.model_selection import GridSearchCV

from sklearn import svm



X_data = data[x_cols]

y_data = data['Vaka']



X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.20, random_state = 0)



nystroemSVC = Nystroem(kernel='rbf', gamma=1.0, n_components=100)

X_train = nystroemSVC.fit_transform(X_train)

X_test = nystroemSVC.transform(X_test)



params_grid = {'C': [0.0001, 0.001, 0.01,0.099, 0.1, 1],

          'gamma': [0.00099, 0.001, 0.01, 0.1, 1],

          'kernel':['rbf', 'linear', 'poly'] }



grid_clf = GridSearchCV(SVC(class_weight='balanced'), params_grid)



grid_clf = grid_clf.fit(X_train, y_train)



print(grid_clf.best_params_)
sns.set_style('white')

sns.set_context('talk')

sns.set_palette('dark')



# Plot of the noisy (sparse)

ax = data.set_index('Vaka')['Tedavi_Edilen'].plot(ls='', marker='o', label='data')

ax.plot(X_data, y_data, ls='--', marker='', label='real function')



ax.legend()

ax.set(xlabel='x data', ylabel='y data');