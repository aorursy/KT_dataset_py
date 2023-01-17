

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/corona-north-macedonia/corona_north_macedonia.csv')
df.info()
df.tail(1)
x_data = np.array(list(range(len(df))))
y_data = np.array(df['vkupno_novi'])
x = np.linspace(1,len(df),len(df))

kvadratni = np.polyfit(x_data,y_data, 2)
y = kvadratni[0]*(x_data**2) + kvadratni[1]*x_data + kvadratni[2]
def func(x, a, b, c):
    return a * np.exp(b * x) + c
def func_1(x, b):
    return np.exp(b * x)
popt, pcov = curve_fit(func, x_data, y_data)
popt1, pcov1 = curve_fit(func_1, x_data, y_data)

y_exp = func(x_data, *popt)
y_exp_1 = func_1(x_data, *popt1)

y_exp_predicted = np.exp(0.2*x_data)
R2 = r2_score(y_data, y_exp)
R2_1 = r2_score(y_data, y_exp_1)
R2_predicted = r2_score(y_exp_predicted,y_data)
R2_poly = r2_score(y_data, y)

parametri = []
parametri.append(popt[0])
parametri.append(popt[1])
parametri.append(popt[2])
parametri.append(R2)
parametri = np.array(parametri)
parametri1 = []
parametri1.append(popt1[0])
parametri1.append(R2_1)
parametri1 = np.array(parametri1)
parametri_predicted = []
parametri_predicted.append(0.20)
parametri_predicted.append(R2_predicted)
parametri_predicted = np.array(parametri_predicted)

parametri_poly = []
parametri_poly.append(kvadratni[0])
parametri_poly.append(kvadratni[1])
parametri_poly.append(kvadratni[2])
parametri_poly.append(R2_poly)
parametri_poly = np.array(parametri_poly)
testirani = 2240 + 406 + 480 + 392 + 257 + 342 + 389 + 414 + 371 + 202 + 386 + 351 + 341 + 496 + 586 + 404 + 330 + 710 + 500 + 660 +437 + 697 + 314 + 263 + 345 + 660 + 649 + 408 + 728 + 335 + 289 + 362 + 279 + 386 + 389 +199+ 249
plt.figure(figsize=(20,10))
plt.plot(df['datum'],df['vkupno_novi'], '-o')
plt.title("Вкупен број на covid-19 заболени / Total covid-19 cases", fontsize=20)
plt.xlabel("Датум / Date", fontsize=20)
plt.ylabel("Број на заболени / Total covid-19 cases", fontsize=20)
plt.xticks(rotation = 45, fontsize=15)
plt.yticks(fontsize=15)
plt.figure(figsize=(20,10))
plt.plot(df['datum'],df['novi_pozitivni'], '-o')
plt.scatter(df['datum'],df['novi_pozitivni'])
plt.title("Нови позитивни / New covid-19 cases", fontsize=20)
plt.xlabel("Датум / Date", fontsize=20)
plt.ylabel("Новозаболени днвено /  Daily new covid-19 cases", fontsize=20)
plt.xticks(fontsize=15, rotation = 45)
plt.yticks(fontsize=15)
plt.figure(figsize=(20,10))
plt.plot(df['datum'],df['active'], '-o')
plt.title("Активни позитивни вкупно / Active cases ", fontsize=20)
plt.xlabel("Датум / Date", fontsize=20)
plt.ylabel("Активни позитивни вкупно /  Total covid-19 currently infected", fontsize=20)
plt.xticks(fontsize=15, rotation = 45)
plt.yticks(fontsize=15)
plt.figure(figsize=(20,10))
plt.plot([], [], ' ', label=" Вкупно тестирани / Tested %d (03.05.2020 last update)" %testirani)
plt.plot(df['datum'],df['vkupno_novi'],'-o', label = ' Real data')
plt.plot(df['datum'],y, label = ' polinomial interpolation y= %2.2fx^2 %2.2fx %+2.2f,  R2_score = %2.3f' % tuple(parametri_poly))
plt.plot(df['datum'], y_exp_1, label = ' exponential interpolation y = e^%2.2fx, R2_score = %2.3f ' % tuple(parametri1))
#plt.plot(df['datum'], y_exp, label = ' exponential interpolation y = %2.2fe^%2.2fx%+2.2f, R2_score = %2.3f' % tuple(parametri))
plt.legend(fontsize=15)
plt.title("Вкупен број на коронавирус заболени / Total covid-19 cases", fontsize=20)
plt.xlabel("Датум / Date", fontsize=20)
plt.ylabel("Број на заболени / Total covid-19 cases", fontsize=20)
plt.xticks(fontsize=15, rotation = 45)
plt.yticks(fontsize=15)
# plt.figure(figsize=(20,10))
# plt.plot([], [], ' ', label=" Вкупно тестирани / Tested %d (15.04.2020 last update)" %testirani)
# plt.plot(df['datum'],df['vkupno_novi'],'-o', label = ' Real data')
# #plt.plot(df['datum'],y, label = ' polinomial interpolation y= %2.2fx^2 %2.2fx %+2.2f' % tuple(kvadratni))
# plt.plot(df['datum'], y_exp_1, label = ' exponential interpolation y= e^%2.2fx, R2_score = %2.3f ' % tuple(parametri1))
# plt.plot(df['datum'], y_exp_predicted, label = ' exponential interpolation y= e^%2.2fx, R2_score = %2.3f' % tuple(parametri_predicted))
# plt.legend(fontsize=20)
# plt.title("Вкупен број на коронавирус заболени / Total coronavirus cases", fontsize=20)
# plt.xlabel("Датум / Date", fontsize=20)
# plt.ylabel("Број на заболени / Total coronavirus cases", fontsize=20)
# plt.xticks(fontsize=15, rotation = '45')
# plt.yticks(fontsize=15)

