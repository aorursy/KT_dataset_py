import numpy as np #do roznych obliczen
import pandas as pd #do operacji na danych
import matplotlib.pyplot as plt #do wykresow
import seaborn as sns #do jeszcze ladniejszych wykresow
%matplotlib inline 
pd.set_option('display.max_columns', 500)
#Wczytujemy dane treningowe i testowe w postaci *.csv.
#header=0 - informuje, gdzie znajduje się wiersz z nazwami kolumn
#index_col=0 - informuje, gdzie znajduje się kolumna, po której indeksujemy nasz zbiór
df_train = pd.read_csv('../input/train.csv', header=0,index_col=0)
df_test = pd.read_csv('../input/test.csv', header=0,index_col=0)

#Polaczenie zbioru treningowego z testowym, by mozna je bylo razem modyfikowac
df_all=pd.concat([df_train, df_test], sort=False)

#Nadanie nazw dataframe'om
df_train.name='Train Dataset'
df_all.name='All Dataset'
df_test.name='Test Dataset'
#Funkcja ulatwiajaca wyswietlanie informacji o zbiorze w konsoli
def print_some_info(df):
    '''Funkcja do wyswietlania informacji o datasecie w formie zbiorczej'''
    
    names_print=['Przykladowy sampel', 'Zestawienie nazw kolumn, liczby wystapien i typow danych', 'Wymiar datasetu',
                 'Brak danych w poszczegolnych kolumnach']
    n_len=[int((100-len(name))/2) for name in names_print ]
    
    print('\n \n Zestawienie dla datasetu: '+ '*'*len(df.name) + df.name  + '*'*len(df.name) )
    
    print('\n'+'~'*n_len[0] + names_print[0]  + '~'*n_len[0]+'\n')
    print(df.sample(4))
    print('\n'+'~'*n_len[1] + names_print[1]  + '~'*n_len[1]+'\n')
    print(df.info())
    print('\n'+'~'*n_len[2] + names_print[2]  + '~'*n_len[2]+'\n')
    print(df.shape)
    print('\n'+'~'*n_len[3] + names_print[3]  + '~'*n_len[3]+'\n')
    print(df.isnull().sum())
df_all.describe

#Wywolanie funkcji dla datesetow
print_some_info(df_all)
print_some_info(df_train)
print_some_info(df_test)

#Inne przydatne funkcje do wstepnej analizy zbioru:
#df.head(n) #pokazuje n pierwszych wierszy df
#df.tail(n) #pokazuje n ostatnich wierszy df
#df.describe() #opis statystyczny
#df[-5:], df[:10] #filtrowanie po wierszach (5 ostatnich, 10 pierwszych)
#df[["Age","Pclass"]][5:30] #filtrowanie po wierszach z wyborem kolumn
#df[(df['Age'] > 5.0) & (df['Age'] < 7.0 ) ] #filtorwanie po kolumnie
#df[(df['Cabin'].str.contains('B2',na=False)) ] #flitrowanie po tekscie
#df[df['Embarked'].isnull()] #filtrowanie po pustych wartosciach
#df.select_dtypes(include=[np.int, np.float]).head() #flitrowanie po typie danych
#df.groupby(['Pclass','Sex'])['Survived'].sum() # grupowanie po kategorii
#Sprawdzam wartosci jakim zakodowana jest plec
print('~~~Unikalne wartosci dla kategorii Sex~~~')
print(df_all.Sex.unique())
print('\n')

#Sprawdzam czy nie ma brakujacych danych w zbiorze
print('~~~Ilosc pustych danych w zbiorze~~~')
print(df_train.Sex.isnull().sum())
print('\n')

#Sprawdzam liczbe mezczyzn i kobiet na statku (test+train)
print('~~~Liczba mezczyzn i kobiet na statku (test+train)~~~')
print(df_all.Sex.value_counts())
print('\n')

#Sprawdzam liczbe mezczyzn i kobiet na statku (tylko train)
print('~~~Liczba mezczyzn i kobiet na statku (train)~~~')
print(df_train.Sex.value_counts())
print('\n')

#Udzial kobiet i mezczyzn wsrod tych, ktorzy przetrwali
print('~~~Liczba mezczyzn i kobiet wsrod uratowanych~~~')
print(df_train.groupby(['Sex'])['Survived'].sum())
print('\n')

#Stosunek przezywalnosci kabiet i mezczyzn
print('~~~Stosunek przezywalnosc kobiet i mezczyzn~~~')
print(df_train.groupby(['Sex'])['Survived'].mean()*100)
print('\n')


#Plotowanie wykresu z zaleznoscia przyzywalnosci od plci
plt.figure(figsize=(4,4))
sns.set_style("whitegrid")
ax = sns.barplot(x="Sex", y="Survived", data=df_train).set(title = 'Stosunek przeżywalności w zależności od płci',
                                                           xlabel = 'Płeć', ylabel = 'Stosunek przeżywalności',
                                                           xticklabels=['Mężczyzna', 'Kobieta'])
#Zastapienie nazw plci 'male' przez 0, 'female' przez 1
def replace_sex_names_with_code(df):
    df.Sex.replace(['male', 'female'], [0, 1], inplace = True)
    return (df)

for i in [df_all, df_train, df_test]:
    replace_sex_names_with_code(i)

#Inne sposoby zakodowania danych
#df_train['_Sex'] = pd.Categorical(df_train.Sex).codes #utworzenie nowej kolumny z zakodowanymi plciami
#sprawdz get_dummies()
#dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#df['sex_cat'] = pd.factorize( df['Sex'] )[0]
#Sprawdzam wartosci jakim zakodowana jest klasa
print('~~~Unikalne wartosci dla kategorii Pclass~~~')
print(df_all.Pclass.unique())
print('\n')

#Sprawdzam czy nie ma brakujacych danych w zbiorze
print('~~~Ilosc pustych danych w zbiorze~~~')
print(df_all.Pclass.isnull().sum())
print('\n')

#Liczba pasazerow poszczegolnych klas (zbior train)
print('~~~Liczba pasazerow w danej klasie (train)~~~')
print(df_train.Pclass.value_counts())
print('\n')

#Udzial poszczegolnych klas wsrod tych, ktorzy przetrwali
print('~~~Stosunek uratowanych z poszczegolnych klas~~~')
print(df_train.groupby(['Pclass'])['Survived'].mean()*100)
print('\n')

#Udzial poszczegolnych klas wsrod tych, ktorzy przetrwali z uwzglednieniem plci
print('~~~Stosunek uratowanych z poszczegolnych klas z uwzglednieniem plci~~~')
print(df_train.groupby(['Pclass', 'Sex'])['Survived'].mean()*100)
print('\n')
#Plotowanie wykresu z zaleznoscia przyzywalnosci od plci
fig, ax =plt.subplots(1,2, figsize=(10,5))
#fig(figsize=(4,4))
sns.set_style("whitegrid")
sns.barplot(x="Pclass", y="Survived", data=df_train, ax=ax[0]).set(title = 'Stosunek przeżywalności w zależności klasy',
                                                           xlabel = 'Klasa', ylabel = 'Stosunek przeżywalności')
sns.barplot(x="Pclass", y="Survived",hue='Sex', data=df_train, ax=ax[1]).set(title = 'Stosunek przeżywalności w zależności klasy i płci',
                                                           xlabel = 'Płeć i klasa', ylabel = 'Stosunek przeżywalności')
                                                          
print('~~~Przykladowy wydruk zmiennej Name~~~')
print(df_all.Name.head(4))
print('\n')

#Liczba unikalnych wartosci
print('~~~Liczba unikalnych wartosci dla kategorii Age~~~')
print(df_all.Name.nunique())
print('\n')

#Powtarzajace sie wartosci
print('~~~Powtarzajace sie wartosci~~~')
print(df_all[df_all.Name.duplicated(keep=False)])
print('\n')

#Sprawdzam czy nie ma brakujacych danych w zbiorze
print('~~~Ilosc pustych danych w zbiorze Name~~~')
print(df_all.Name.isnull().sum())
print('\n')
#Ze zbioru wyluskuje nazwiska (pierwszy czlon Name), oraz zapisuje je tylko za pomoca malych liter
#by te same nazwiska nie byly traktowane jako rozne ze wzgledu na wielkosc liter
#Nastepnie obliczam ile osob nosi takie samo nazwisko i drukuje przylad
print('\n~~~Liczba o tym samym nazwisku (wydruk pierwszych 5)~~~')
LastNameSum=df_train['Name'].map(lambda x: x.split(',')[0].lower()).value_counts().reset_index().rename(index=str, columns={"index": "_Name","Name": "counts" })
print(LastNameSum.head(5))

#Tworze nowa kolumne w DF o nazwie _Name, ktora przechowuje nazwiska pasazerow
df_train['_Name']=df_train['Name'].map(lambda x: x.split(',')[0].lower())
df_all['_Name']=df_all['Name'].map(lambda x: x.split(',')[0].lower())

#Sprawdzam ile jest podzbiorow o danej liczbie osob, ktore tworza pseudo-rodzine (maja takie samo nazwisko), 
#ile jest pseudo-rodzin o liczbie czlonkow 9,8,7, etc.
print('\n~~~Liczba osob o takim samym nazwisku | Liczba takich przypadkow~~~')
print(LastNameSum['counts'].value_counts())

#Tworze zestawienie ilosci osob o takim samym nazwisku i ile osob przetrwalo
LastNameSum=LastNameSum.sort_values('_Name').reset_index().set_index('_Name')
LastNameSurvivedSum=df_train.groupby(['_Name'])['Survived'].sum().reset_index()
LastNameSurvivedSum=LastNameSurvivedSum.sort_values('_Name').reset_index().set_index('_Name')
LastNameResult = pd.concat([LastNameSum['counts'], LastNameSurvivedSum['Survived']], axis=1, join_axes=[LastNameSum.index]).sort_values('counts', ascending=False)
print('\n~~~Liczba o tym samym nazwisku | Liczba tych co przetrwali (wydruk pierwszych 5)~~~~~~')
print(LastNameResult.head(5))

#Tworze zestawienie ile osob jest w pozbiorze o danej wielkosci grupy (liczbie osob o tym samym nazwisku), ile z tych osob przetrwalo
SurvivedDependLastNameSizeGroup=LastNameResult.groupby(['counts'])['Survived'].sum()
PeopleNumberDependLastNameSizeGroup=LastNameResult.sort_values('counts')
PeopleNumberDependLastNameSizeGroup=LastNameResult['counts'].value_counts(sort=False)
PeopleNumberDependLastNameSizeGroup=PeopleNumberDependLastNameSizeGroup.index*PeopleNumberDependLastNameSizeGroup
SurvivedPercent=SurvivedDependLastNameSizeGroup/PeopleNumberDependLastNameSizeGroup*100
LastNameResultWithNumbers = pd.concat([PeopleNumberDependLastNameSizeGroup, SurvivedDependLastNameSizeGroup, SurvivedPercent], axis=1, join_axes=[PeopleNumberDependLastNameSizeGroup.index]).rename(index=str, columns={0: "All",1: "Survived [%]" })
print('\n~~~Rozmiar grupy o tym samym nazwisku | Wszyscy  | Uratowani | Uratowani w [%]~~~~~~')
print(LastNameResultWithNumbers)

#Wykres rozkladu ilosci wszystkich pasazerow oraz tych co przetrwali w zaleznosci ile osob mialo takie samo nazwisko
ax=LastNameResultWithNumbers.plot(kind='bar',y=['All', 'Survived'], label=['Wszyscy', 'Uratowani'])
plt.legend()
ax.set_xlabel('Liczba osob o takim samym nazwisku')
ax.set_ylabel('Suma osob')
ax.set_title('Rozklad liczby pasazerow o takim samym nazwisku')
plt.show()
#Sprawdzenie liczebnosci osob o konkretnym tytule
df_all['_Title']=df_all['Name'].map(lambda x: x.split(', ')[1].split('.')[0].lower() )
print('\n~~~Ilosc osob o danym tytule~~~')
print(df_all['_Title'].value_counts())

#Sprawdzenie rozkladu wieku w zaleznosci od tytuly
print('\n~~~Tytul|Srednia|Mediana|Min|Max wieku w zaleznosci od tytulu~~~')
print(df_all.groupby('_Title')['Age'].agg([np.mean, np.median, np.min, np.max]))

#Agregacja tytulow i ich zakodowanie
popular_titles = ["mr", "miss", "mrs", "master", "dr", "rev"]
df_all['Title'] = df_all['_Title'].map(lambda x: x if x in popular_titles else "other")
df_all['Title_encoded'] = pd.factorize( df_all['Title'] )[0]
print('\n~~~Ilosc osob o danym tytule po agregacji~~~')
print(df_all['Title'].value_counts())

#Rozklad wieku po agregacji, zebranie mediany wieku, by uzupelnic puste pola w wieku
print('\n~~~Tytul|Srednia|Mediana|Min|Max wieku w zaleznosci od tytulu~~~')
print(df_all.groupby('Title')['Age'].agg([np.mean, np.median, np.min, np.max]))
missing_ages=df_all.groupby('Title')['Age'].agg([np.mean, np.median, np.min, np.max]).to_dict()['median']

#Sprawdzenie szans przezycia w zaleznosci od tytulu
print('\n~~~Szansa przezycia w zaleznosci od tytulu~~~')
print(df_all.groupby('Title')['Survived'].agg([np.mean]))

#Alternatywny sposob wyznaczenie tytulow
#pat = r",\s([^ .]+)\.?\s+"
#df_all['Title'] =  df_all['Name'].str.extract(pat,expand=True)[0]
#df_all.groupby('Title')['Title'].count()
#df_all.loc[df_all['Title'].isin(['Mille','Ms','Lady']),'Title'] = 'Miss'
#df_all.loc[df_all['Title'].isin(['Mme','Sir']),'Title'] = 'Mrs'
#df_all.loc[~df_all['Title'].isin(['Miss','Master','Mr','Mrs']),'Title'] = 'Other' # NOT IN
#df_all['_Title'] = pd.Categorical(df_all.Title).codes
#df_all.groupby('Title')['Title'].count()
#Sprawdzam wartosci jakim zakodowana jest klasa
print('~~~Unikalne wartosci dla kategorii Age~~~')
print(df_all.Age.unique())
print('\n')

#Liczba unikalnych wartosci
print('~~~Liczba unikalnych wartosci dla kategorii Age~~~')
print(df_all.Age.nunique())
print('\n')


#Sprawdzam czy nie ma brakujacych danych w zbiorze
print('~~~Ilosc pustych danych w zbiorze Age~~~')
print(df_all.Age.isnull().sum())
print('\n')

#Statystyka opisowa 
print('~~~Statystyka opisowa wieku (Age)~~~')
print(df_all.Age.describe())
print('\n')
#Wyplotowanie histogramu z podzialem na 30 zakresow
fig, ax=plt.subplots()
plt_all=plt.hist(df_all['Age'],bins = 30,  range = [0,100],label='Wszyscy')
plt_survived=plt.hist(df_all[df_all['Survived']==1]['Age'], bins=30, range=[0,100], label='Uratowani')
plt.legend()
ax.set_xlabel('Wiek')
ax.set_ylabel('Liczba pasazerow')
ax.set_title('Rozklad wieku pasazerow')
plt.show()
#Uzupelniam puste miejsca wiekiem z tytulow
df_all['Age'] = df_all.apply( lambda x: x['Age'] if str(x['Age']) != 'nan' else missing_ages[x['Title']], axis=1 )
print('\n~~~Puste miejsca po imputacji w zmiennej Age~~~')
print(df_all.Age.isnull().sum())
#Podzial wieku na podzbiory do ciecia
age_bins = [0,1, 3, 8, 15, 20,30, 40,60, 100]
df_all['Age_cut']=pd.cut(df_all["Age"], bins=age_bins)

#Ilosc osob w danym przedziale wiekowym
print('\n~~~Ilosc osob w danym przedziale wiekowym~~~')
print(df_all['Age_cut'].value_counts())

#Szansa przezycia od przedzialu wiekowego
print('\n~~~Szansa przezycia w zaleznosci od przedzialu wieku~~~')
print(df_all.groupby('Age_cut')['Survived'].agg(np.mean))

#Encoding grup wiekowych
age_bins = [0,1, 2,3, 5,8,12, 15,18, 20,25,30,35, 40,50,60, 100]
df_all['Age_cut']=pd.cut(df_all["Age"], bins=age_bins)
df_all['Age_encoded'] = pd.factorize( df_all['Age_cut'] )[0]
#Utworzenie nowej zmiennej Family_size
df_all['Family_size']=df_all['SibSp']+df_all['Parch']+1

print('\n~~~Wielkosc rodziny | Liczba pasazerow~~~')
print(df_all.Family_size.value_counts())

print('\n~~~Wielkosc rodziny | Szanse na przezycie~~~')
print(df_all.groupby('Family_size')['Survived'].mean())

#Utworzenie zmiennej IsAlone informujacej czy pasazer podrozowal sam
df_all['IsAlone']=df_all.apply(lambda x: 1 if x['Family_size']==1 else 0, axis=1)

print('\n~~~Zaleznosc miedzy podrozowaniem samemu a przezyciem')
print(df_all.groupby('IsAlone')['Survived'].mean())
print('\n~~~Satystyka opisowa - Fare - Cena biletu~~~')
print(df_all.Fare.describe())

#Wydrukowanie informacji o osobie, ktorej ceny biletu brakuje
print('\n~~~Wiersz, w ktorym brakuje ceny biletu~~~')
print(df_all[df_all.Fare.isnull()][['Name', 'Pclass', 'Age', 'Embarked']])

#Pozyskanie wartosci klasy, zaokretowania i oplaty
#missPclass=df_all.loc[df_all.Fare.isnull(), 'Pclass'].values
#missEmbarked=df_all.loc[df_all.Fare.isnull(), 'Embarked'].values
#missFare_Age=df_all.loc[df_all.Fare.isnull(), 'Age'].values

#Ceny biletow pasazerow o podobnej specyfice to tego, ktorego ceny brakuje
print('\n~~~Sprawdzenie cen biletow pasazerow w podobnym wieku, klasie i zaokretowaniu~~~')
val = df_all[(df_all['Pclass'] == 3)&(df_all['Embarked'] == 'S')&(df_all['Age'] > 60.5)][['Age','Fare']];
print(val.groupby('Age').agg(['min','max','count','mean','median']))

#Obliczenie sredniej ceny biletow dla osob o podobnej specyfice do brakujacej
val_mean=val.groupby('Age').agg(np.mean).mean()
print('\n~~~Srednia cena biletu osob o podobnej specyfice={}'.format(val_mean.values))

#Uzupelnieni pustych miejsc
df_all.loc[df_all.Fare.isnull(),'Fare']=val_mean.values


#Wstawienie danych w konkretnym miejscu
#df_all.at[1044,'Fare']=val_mean.values
#df_all['Fare']=df_all.apply(lambda x: x['Fare'] if str(x['Fare']) != 'nan' else val_mean.values, axis=1 )
# Skalowanie Fare
from sklearn import preprocessing
df_all['_Fare'] = preprocessing.scale(df_all[['Fare']])[:,0]
#df_all['_Fare'] = preprocessing.normalize(df_all[['Fare']], norm='l2')

#Wyplotowanie histogramu z Fare przed skalowaniem z podzialem na 100 zakresow
fig, ax=plt.subplots()
plt_all_fare=plt.hist(df_all['Fare'], bins=100)
plt_survived_fare=plt.hist(df_all[df_all['Survived']==1]['Fare'], bins=100)
plt.legend()
ax.set_xlabel('Fare - cena biletu')
ax.set_ylabel('Liczba pasazerow')
ax.set_title('Rozklad cen biletow')
plt.show()

#Wyplotowanie histogramu z Fare po skalowaniu z podzialem na 100 zakresow
fig, ax=plt.subplots()
plt_all_fare=plt.hist(df_all['_Fare'], bins=100)
plt_survived_fare=plt.hist(df_all[df_all['Survived']==1]['_Fare'], bins=100)
plt.legend()
ax.set_xlabel('Fare - cena biletu - po skalowaniu')
ax.set_ylabel('Liczba pasazerow')
ax.set_title('Rozklad cen biletow - po skalowaniu')
plt.show()

#Pociecie na przedzialy, by sprawdzic wplyw ceny biletu na przezycie
fare_bins = [0,7,10, 20, 30, 50, 70,100,200,600]
df_all['Fare_cut']=pd.cut(df_all["Fare"], bins=fare_bins)

#Zakodowanie przedzialow, byc moze poprawia wynik
df_all['_Fare_encoded'] = df_all['Fare_cut'].cat.codes

#Ilosc osob w danym przedziale ceny biletu
print('\n~~~Ilosc osob w danym przedziale ceny biletu Fare~~~')
print(df_all['Fare_cut'].value_counts())

#Ilosc osob w danym przedziale ceny biletu z uwzglednieniem klasy
print('\n~~~Ilosc osob w danym przedziale ceny biletu Fare z uwzgledniem klasy~~~')
print(df_all.groupby('Fare_cut')['Pclass'].value_counts())

#Szansa przezycia od przedzialu cenowego biletu
print('\n~~~Szansa przezycia w zaleznosci od przedzialu cenowego biletu~~~')
print(df_all.groupby(['Fare_cut', 'Pclass'])['Survived'].agg(np.mean))
print('\n~~~Przykladowy wydruk informacji o Cabin~~~')
print(df_all.Cabin.sample(4))

#Uzyskanie tylko typow kabin (pierwsz litera) + brakujace nazwane jako missing
df_all['_Cabin'] =df_all.Cabin.astype(str).str[0]
df_all.loc[df_all._Cabin=='n', '_Cabin']='missing'

print('\n~~~Zestawienie typow Cabin (pierwsza litera)')
print(df_all._Cabin.value_counts())

print('\n~~~Zestawienie typow Cabin i klas')
print(df_all.groupby('_Cabin')['Pclass'].value_counts())

#Zakodowanie typow kabin
df_all['_Cabin_encoded']=pd.Categorical(df_all['_Cabin']).codes
#Sprawdzam dla jakich wierszy brakuje danych
print('\n~~~Dla jakich wierszy brakuje zmiennej Embarked~~~')
print(df_all[df_all.Embarked.isnull()][['Name', 'Fare', 'Pclass', 'Ticket']])

#Jakie zaokretowanie maja osoby o tym samym typie kabiny i podobnej cenie biletu
print('\n~~~Jakie zaokretowanie maja osoby o tym samym typie kabiny i podobnej cenie biletu')
print(df_all[(df_all['_Cabin']=='B')&(df_all['Fare']>79)]['Embarked'].value_counts())

print('\n~~~Ilosc pasazerow z konkretnych portow z podzialem na klasy')
print(df_all.groupby('Embarked')['Pclass'].value_counts())

print('\n~~~Zestawienie portow zaokretowania, klasy i mediany ceny biletu~~~')
print(df_all.groupby(['Embarked','Pclass'])['Fare'].median())

print('\n~~~Zestawienie portow zaokretowania i szans na przezycie~~~')
print(df_all.groupby(['Embarked'])['Survived'].mean())

print('\n~~~Zestawienie portow zaokretowania i pasazerow w konkretnych typach kabin')
print(df_all.groupby(['Embarked'])['_Cabin'].value_counts())

#Zakodowanie zmiennej Embarked - portow za pomoca int-ow
df_all['_Embarked_encoded']=pd.Categorical(df_all['Embarked']).codes

print('\n~~~Przykladowe numery biletow~~~')
print(df_all.Ticket.sample(5))

print('\n~~~Liczba unikalnych nr biletow~~~')
print(df_all.Ticket.nunique())

#Zmienna ile osob ma taki sam nr biletu
df_all['_TicketCounts'] = df_all.groupby(['Ticket'])['Ticket'].transform('count')
#Wybranie kolumn, ktore beda uzyte do treningu
columns_names=['Survived', 'Pclass', 'Sex', 'Age_encoded', 'SibSp', 'Parch', 'Title_encoded', 'Family_size','IsAlone','_Fare_encoded','_Cabin_encoded', '_Embarked_encoded', '_TicketCounts', '_Fare' ]

#Sprawdzenie korelacji miedzy zmiennymi
corr = df_all[columns_names].corr()
#print(corr)

#Heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(corr)
#Import roznych modeli ML do testow
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import linear_model
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

#Zebranie estymatow w jednym dictionarze
classifiers = {
    'LogisticRegression':  linear_model.LogisticRegression(),
    'SVC': SVC(class_weight='balanced'),
    'LinearSVC':  LinearSVC(),
    'GaussianNB': GaussianNB(),
    'XGBoost': XGBClassifier(),#max_depth=3, n_estimators=15, subsample=0.8, random_state=2018),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),#(n_estimators=100),
    'KNeighbours': KNeighborsClassifier(),
    'NeuralNetwork': MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
}
#Wybranie kolumn, ktore beda uzyte do wstepnych obliczen
cols=['Pclass', 'Sex', 'Age_encoded', 'Family_size','_Fare', 'Title_encoded']#, 'IsAlone']#'_Fare_encoded']#,'_Cabin_encoded']

#Wydzielenie zbioru treningowego z polaczonego datasetu
def get_train_data(df_all, cols):
    SURV = 891
    X = df_all[:SURV][cols] 
    Y = df_all[:SURV]['Survived']
    return X,Y

X,Y=get_train_data(df_all, cols)
#Chwilowe wylaczenie warningow, zeby nie zaburzaly widoku wykresow
import warnings
warnings.filterwarnings('ignore')

#Import learning_curve, ktora pozwala na cross-validation (zwraca wyniki dla treningu, 
#testu dla roznych zbiorow treningowych i testowych) z dodatkowym uwzglednieniem rozmiaru zbioru treningowego
from sklearn.model_selection import learning_curve

#Import ShuffleSplit, ktory umozliwa losowe wybranie zbioru treningowego i walidacyjny do CV, z uwzgledniem
#rozmiaru zbioru walidacyjnego
from sklearn.model_selection import ShuffleSplit

#Funkcja plot_learning_curve odpowiedzialna jest za wyplotowanie wykresow
#krzywej treningowej oraz krzywej walidacyjnej, wykorzystujac learning_curves,
#pokazuje jak zmienia sie wynik treningu i CV w zaleznosci od rozmiaru zbioru
#treningowego, dodatkowo dodany, by zwracala wartosc srednia wyniku dla CV
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), plot=True):
    
    '''Generate a simple plot of the test and training learning'''
    if plot==True:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    if plot==True:
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
    return plt, test_scores_mean
def test_models(classifiers, X, Y, plot=True):
    '''Funkcja jako argument przyjmuje rodzaj modelu w slowniku, zbior X oraz zbior Y'''
    test_score={} #przechowuje usrednione wyniki CV dla roznych modeli
    for c in classifiers:
        title = "Learning Curves " + c 
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0) #CV 100 iteracji z 20% losowym zbiorem testowym
        estimator = classifiers[c] 
        ax, test_score[c] = plot_learning_curve(estimator, title, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4, plot=plot)
        if plot==True:
            plt.show()
    print('\n~~~Usrednione wyniki testu dla roznych modeli, z uwzgledniem wielkosci zbioru treningowego do 20% do 100% oraz srednia dla wszystkich treningow~~~')
    df_scores=pd.DataFrame.from_dict(test_score).T.rename(index=str, columns={0: '20_Train', 1: '40_Train', 2: '60_Train', 3: '80_Train', 4: '100_Train'})
    df_scores['Mean']=df_scores.mean(numeric_only=True, axis=1)
    df_scores
    return df_scores

test_models(classifiers,X, Y)


#Wszystkie utworzone, zmodyfikowane, zeskalowane, zakodowane lub pierwotne kolumny(, ktore zawieraja tylko liczby calkowite)
cols_all=['Pclass', 'Sex', 'Age_encoded', 'Family_size',
          '_Fare', 'Title_encoded' ,'_Cabin_encoded','_Embarked_encoded',
          '_TicketCounts', 'SibSp', 'Parch','IsAlone','_Fare_encoded']

#Wydzielenie zbioru treningowego z polaczonego datasetu
X,Y=get_train_data(df_all, cols_all)

#Wynik XGBoosta dla wszystkich kolumn
xgb_clas={'XGBoost': XGBClassifier(),}
test_models(xgb_clas, X, Y, plot=False)

#Plot Feature Importance korzystajac z wbudowanych funkcji XGBoosta
from numpy import loadtxt
from xgboost import plot_importance

# fit na danych
model = XGBClassifier()
model.fit(X, Y)

# plot feature importance
plot_importance(model)
plt.show()


#Wszystkie utworzone, zmodyfikowane, zeskalowane, zakodowane lub pierwotne kolumny(, ktore zawieraja tylko liczby calkowite)
cols_cut1=['Pclass', 'Sex',  'Family_size',
          '_Fare', 'Title_encoded' ,'_Embarked_encoded'
          ,'_Cabin_encoded','Age_encoded','_TicketCounts']

#Wydzielenie zbioru treningowego z polaczonego datasetu
X,Y=get_train_data(df_all, cols_cut1)
#Na podstawie https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
#Zadaniem jest:
#wyznaczenie feature importance
#a potem z uzyciem SelectFromModel ze zbioru wybieramy zmienne, ktore sa uzyte do obliczenia accuracy
#zaczynamy od jednej zmiennej z najwyzszym feature importance, a potem dwie zmienne, 3 zmienne i tak az uzwglednimy wszystkie
#sprawdzamy jak uwzglednienie kolejnych zmiennych wplywa na accuracy zbiorcze

from numpy import sort
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# podzial na train i test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# fit na train
model = XGBClassifier()
model.fit(X_train, y_train)

#Drukowanie feature importance wraz z nazwami cech, po sortowaniu, by wiedziec dla czego drukowane sa accuracy
feature_importance_with_names=zip(X.columns,model.feature_importances_)
print('\n~~~Posortowane feature importance od najwiekszego do najmniejszego~~~\n')
print(sorted(feature_importance_with_names, key=lambda x: x[1], reverse=True))
print ('\n')

# predykcja i wydruk accuracy
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)

print('\n~~~Accuracy z uwzglednieniem wszystkich zmiennych~~~\n')
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('\n~~~Wynik accuracy w zaleznosci od ilosci uwzglednionych zmiennych~~~\n')

# posortowanie wg wartosci feture importance
thresholds = sort(model.feature_importances_)

for thresh in thresholds:
    
    
    #wybor zmiennych, ktore feature importance jest wyzszy niz threshold
    #czyli zaczyna od najwiekszej wartosci, ktora spelnia 1 zmienna (_Fare)
    #potem mniejszy threshold, ktory spelniaja juz 2 zmienne (_Fare i Age_encode)
    #itd. az uwzgledni wszystkie zmienne
    #wiec najpierw do obliczen accuracy wykorzystuje 1 zmienna, potem 2,..., a na koniec wszystkie
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    
    #print
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("FeatImportance=%.3f, FeatNumber=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

 

xgb_clas={'XGBoost': XGBClassifier()}
test_models(xgb_clas, X, Y, plot=False)
    
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn.model_selection import cross_val_score

#Obliczenie accuracy przy uwzglednieniu tylko 1 zmiennej
def get_accuracy_for_one_feature(df_all, cols_cut1):
    score = {}
    model=XGBClassifier()
    for i in range(0,len(cols_cut1),1):
        cols_cut2=[cols_cut1[i]]
        X,Y=get_train_data(df_all, cols_cut2)    
        score[cols_cut2[0]]=cross_val_score(model, X, Y, cv=5).mean()


    #posortowanie od najwiekszego accuracy do najmniejszego
    print('\n~~~Accuracy przy uwzglednieniu tylko jednej zmiennej~~~')
    return sorted(score.items(), key=itemgetter(1))[::-1]
    

get_accuracy_for_one_feature(df_all, cols_cut1)
#Normalizacja Fare w 0-1
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

x = df_all['Fare'].values #returns a numpy array
x=x.reshape(df_all.shape[0],1)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_all['_Fare_norm'] = pd.DataFrame(x_scaled)
#Sprawdzenie wyniku accuracy dla pojedynczej zmiennej
col_cut_with_fare=cols_cut1.copy()
col_cut_with_fare.append('_Fare_norm')
get_accuracy_for_one_feature(df_all, col_cut_with_fare)
print('\n~~~Zmienne pozostawione do dalszego tuningu XGBoosta:~~~\n')
print(cols_cut1)
from sklearn.model_selection import GridSearchCV

#zastosowanie tylko ponizszych 5 zmiennych zapewnia wynik na poziomie 0.7894
#cols_cut12=['Pclass', 'Sex', 'Age_encoded', 'Family_size','_Fare']

cols_cut12=cols_cut1
X, Y = get_train_data(df_all, cols_cut12)

#Parametry do sprawdzenia i wybrania najlepszych
cv_params = {'max_depth': [2,3,4,5],
             'n_estimators': range(10,210,50),
            'learning_rate': [0.01,0.02,0.05,0.07,0.1]}

#parametry podane do XGBoosta
#ind_params = {'learning_rate': 0.05, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
#             'objective': 'binary:logistic'}

xgb= XGBClassifier().fit(X, Y)
optimized_xgb = GridSearchCV(xgb, cv_params, scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_xgb.fit(X, Y)
print('\n~~~Accuracy po optymalizacji~~~\n')
optimized_xgb.score(X,Y)
#Predykcja dla wyniku testowego i submit
#Wyznaczenie zbioru testowego
SURV = 891
Xp = df_all[SURV:][cols_cut12]

#Utworzenie pliku wynikowego
result = pd.DataFrame({'PassengerID': df_all[SURV:].index })
result['Survived'] = optimized_xgb.predict(Xp).T.astype(int)
result[['PassengerID','Survived']].to_csv('submission.csv',index=False)