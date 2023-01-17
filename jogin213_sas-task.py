# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataPrep = pd.read_csv("/kaggle/input/sas-final/AGD_COMPLEX_final.csv", sep=';')
binSwieto = (dataPrep.Swieto == "T").astype(int)

dataPrep['Swieto'] = binSwieto

del binSwieto

dataPrep.Platnosc.fillna('nie dotyczy', inplace=True)

dataPrep.cena_sprzedazy.fillna("0", inplace=True)

dataPrep.cena_sprzedazy = dataPrep.cena_sprzedazy.str.replace(',','.')

dataPrep.cena_sprzedazy = dataPrep.cena_sprzedazy.astype('float')

toAgg = dataPrep[["Grupa_produktowa", "Platnosc", "Wojewodztwo", "Rok", 'Miesiac', "cena_sprzedazy"]]

print(toAgg.dtypes)

for i in ['Grupa_produktowa', 'Platnosc', 'Wojewodztwo']:

    toAgg[i] = toAgg[i].astype('category')

toAgg = toAgg.groupby(["Grupa_produktowa", "Platnosc", "Wojewodztwo", "Rok", "Miesiac"])
aggreg = toAgg.agg(['mean', 'median', 'min', 'max', 'std'])

aggreg.sort_values(by = ['Grupa_produktowa', "Platnosc", "Wojewodztwo", "Rok", "Miesiac"], inplace = True)

aggreg.reset_index(inplace=True)

aggreg['Rok i miesiac'] = aggreg['Miesiac'].apply(str) + "." + aggreg['Rok'].apply(str)

aggreg.drop(['Miesiac', 'Rok'], axis=1, inplace=True)
aggColumns = ['Grupa produktowa','Forma Płatności','Województwo','Sprzedaż (średnia)','Sprzedaż (mediana)', \

              'Sprzedaż (minimum)','Sprzedaż (maksimum)','Sprzedaż (odchylenie standardowe)','Miesiąc i rok transakcji']

aggreg.columns = aggColumns

aggreg = aggreg[['Grupa produktowa','Forma Płatności','Województwo','Miesiąc i rok transakcji','Sprzedaż (średnia)','Sprzedaż (mediana)', \

              'Sprzedaż (minimum)','Sprzedaż (maksimum)','Sprzedaż (odchylenie standardowe)']]

aggreg = aggreg.round(decimals=2)
for i in ['Sprzedaż (średnia)', 'Sprzedaż (mediana)', 'Sprzedaż (minimum)', 'Sprzedaż (maksimum)', 'Sprzedaż (odchylenie standardowe)']:

    aggreg[i].fillna(0, inplace=True)

aggreg.head(100)
import os

#os.chdir(r'/kaggle/working/output')

aggreg.to_csv(r'/kaggle/working/amt_grouped3.csv')
dataPrep.drop(['Data', 'Kod', 'Powiat_ID', 'Wojew_ID'], axis=1, inplace=True)
for i in ['Grupa_produktowa', 'Platnosc', 'Wojewodztwo', 'Kategoria', 'Region', 'Potencjal', 'Kanal', 'Produkt_ID']:

    dataPrep[i] = dataPrep[i].astype('category')
cols = list(dataPrep.columns)

for i in cols:

    print(dataPrep[i].value_counts())

len(dataPrep['Produkt_ID'].value_counts())
idCena = set((x['Produkt_ID'], x['Cena']) for index, x in dataPrep.iterrows())

print(len(idCena))
dataPrep.drop(['Produkt_ID', 'Cena'], axis=1, inplace=True)
dataPrep.drop(['Ilosc'], axis=1, inplace=True)
czas = ['Miesiac']

miejsce = ['Region']

dProduktu = ['Grupa_produktowa']

cechaY = 'cena_sprzedazy'

cechyPelne = dataPrep.columns[:-7]

cechy = [*czas, *miejsce, *dProduktu]

print(cechy)
def my_regression(df, cechy, cechaY):

    import sklearn

    from sklearn.metrics import mean_squared_error, r2_score

    from sklearn import linear_model

    X = df[cechy]

    y = df[cechaY]

    y.fillna(0, inplace=True)

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)

    #X_train = scaler.transform(X_train)

    #X_test = scaler.transform(X_test)

    y_train,y_test = y_train.values.ravel(),y_test.values.ravel()

    

    regr = linear_model.LinearRegression()

    regr.fit(X_train,y_train)

    y_pred = regr.predict(X_test)

    

    print('wskaznik r^2: %.2f' % r2_score(y_test, y_pred))

    

    noweCechy = X.columns

    wspolczynniki = regr.coef_

    return y_test,y_pred,wspolczynniki, noweCechy



print("Bez pomijania cech:")

my_regression(dataPrep,cechyPelne, 'cena_sprzedazy')

print("Przy przyjetych ograniczeniach:")

OLE_y_test,OLE_y_pred,OLE_coef, nCechy = my_regression(dataPrep, cechy, 'cena_sprzedazy')

for i in range(len(nCechy)):

    print(nCechy[i] + ": %2f"%OLE_coef[i])

#print('Linear Regression coefficients', OLE_coef)