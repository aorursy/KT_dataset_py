import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline 
pd.set_option('display.max_columns', 200)
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

#Wczytanie kolumn z plików csv do dataframe-ów
df_kredyty = pd.read_csv('../input/kredyty.csv', header=0,index_col='id')
df_wczesniejsze = pd.read_csv('../input/wczesniejsze_wnioski.csv', header=0,index_col='id')
df_BIK = pd.read_csv('../input/inne_BIK.csv', header=0,index_col='id')
print('Wymiary tabeli:\n kredyty: '+str(df_kredyty.shape[0])+' wierszy '+str(df_kredyty.shape[1])+' kolumn'+
      '\n wczesniejsze kredyty: '+str(df_wczesniejsze.shape[0])+' wierszy '+str(df_wczesniejsze.shape[1])+' kolumn'+
      '\n informacji BIK: '+str(df_BIK.shape[0])+' wierszy '+str(df_BIK.shape[1])+' kolumn')

# Sprawdzamy unikalne index-y w tabelach, a więc ile jest róznych, unikalnych kilentów w tabelach
# i czy klienci z tabel wczesniejsze oraz BIK sa podzbiorem klientow z tabeli kredyty
kredyty_unique=list(df_kredyty.index.unique());
wczesniejsze_unique=list(df_wczesniejsze.index.unique());
BIK_unique=list(df_BIK.index.unique());

print("Czy index tabeli wczesniejsze zawiera się w tabeli kredyty? -"+str(set(wczesniejsze_unique).issubset(kredyty_unique))+
"\nCzy index tabeli BIK zawiera się w tabeli kredyty? -"+str(set(BIK_unique).issubset(kredyty_unique)))

print("\nLiczba unikalnych klientow w tabeli:\n"+
      "kredyty: "+str(len(kredyty_unique))+
      "\nwczesniejsze wnioski: "+str(len(wczesniejsze_unique))+
      "\nBIK: "+str(len(BIK_unique)))
df_kredyty.sample(5)
#Podsumowanie w 1 tabeli
summary = pd.DataFrame(df_kredyty.dtypes, columns=['Dtype'])
summary['Brakujace_dane'] = pd.DataFrame(df_kredyty.isnull().any())
summary['Suma_brakow'] = pd.DataFrame(df_kredyty.isnull().sum())
summary['Udzial_procent_brakujacych'] = round((df_kredyty.apply(pd.isnull).mean()*100),2)
summary['Suma_unikal_wart']=pd.DataFrame(df_kredyty.nunique())
summary.Dtype = summary.Dtype.astype(str)

#Wyswietlenie 74 wierszy
pd.options.display.max_rows = 74

summary.sort_values(by=['Suma_brakow'],ascending=False)
#summary[summary['Brakujace_dane']==True]
print("Liczba zmiennych z brakującymi danymi: "+str(len(summary[summary['Brakujace_dane']==True])))
print("\nLiczba zmiennych z brakami powyżej 5%: "+str(len(summary[summary['Udzial_procent_brakujacych']>5])))
df_wczesniejsze=df_wczesniejsze.reset_index().set_index('id_2')
df_wczesniejsze.sample(5)
df_wczesniejsze = df_wczesniejsze.replace('XNA', 'NaN')
#Podsumowanie w 1 tabeli
summary = pd.DataFrame(df_wczesniejsze.dtypes, columns=['Dtype'])
summary['Brakujace_dane'] = pd.DataFrame(df_wczesniejsze.isnull().any())
summary['Suma_brakow'] = pd.DataFrame(df_wczesniejsze.isnull().sum())
summary['Udzial_procent_brakujacych'] = round((df_wczesniejsze.apply(pd.isnull).mean()*100),2)
summary['Suma_unikal_wart']=pd.DataFrame(df_wczesniejsze.nunique())
summary.Dtype = summary.Dtype.astype(str)

#Wyswietlenie 74 wierszy
pd.options.display.max_rows = 74

summary.sort_values(by=['Suma_brakow'],ascending=False)
#summary[summary['Brakujace_dane']==True]
print("Liczba zmiennych z brakującymi danymi: "+str(len(summary[summary['Brakujace_dane']==True])))
print("\nLiczba zmiennych z brakami powyżej 5%: "+str(len(summary[summary['Udzial_procent_brakujacych']>5])))
df_BIK=df_BIK.reset_index()
df_BIK.sample(5)
#Podsumowanie w 1 tabeli
summary = pd.DataFrame(df_BIK.dtypes, columns=['Dtype'])
summary['Brakujace_dane'] = pd.DataFrame(df_BIK.isnull().any())
summary['Suma_brakow'] = pd.DataFrame(df_BIK.isnull().sum())
summary['Udzial_procent_brakujacych'] = round((df_BIK.apply(pd.isnull).mean()*100),2)
summary['Suma_unikal_wart']=pd.DataFrame(df_BIK.nunique())
summary.Dtype = summary.Dtype.astype(str)

#Wyswietlenie 74 wierszy
pd.options.display.max_rows = 74

summary.sort_values(by=['Suma_brakow'],ascending=False)
#summary[summary['Brakujace_dane']==True]
print("Liczba zmiennych z brakującymi danymi: "+str(len(summary[summary['Brakujace_dane']==True])))
print("\nLiczba zmiennych z brakami powyżej 5%: "+str(len(summary[summary['Udzial_procent_brakujacych']>5])))
def plot_value_count(x, hue='prognozowana', fig_size=(8,5), xtic_rot=90):
    rcParams['figure.figsize'] = fig_size
    ax=sns.countplot(x=x,hue='prognozowana', data=df_kredyty, order = df_kredyty[x].value_counts().index).set(title = 'Rozkład typu '+x,
                                                               xlabel = 'Rodzaj '+x, ylabel = 'Liczba klientów')
    plt.xticks(rotation=xtic_rot);
    plt.show()

def category_summary(variable):
    kat = pd.DataFrame(df_kredyty[variable].value_counts())
    kat.rename(columns={variable:'Suma_obserwacji'}, inplace=True)
    kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
    kat['Procent_niesplacajacych']=df_kredyty.groupby(variable)['prognozowana'].mean()*100
    return kat
kat = pd.DataFrame(df_kredyty.prognozowana.value_counts())
kat.rename(columns={'prognozowana':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
df_kredyty.prognozowana=df_kredyty.prognozowana.astype('uint8')
ax=sns.countplot(df_kredyty.prognozowana).set(title = 'Rozkład zmiennej prognozowana',
                                                           xlabel = 'Target - prognozowana', ylabel = 'Liczba klientów')
cat_binary_columns=['prognozowana']
kat = pd.DataFrame(df_kredyty.plec.value_counts())
kat.rename(columns={'plec':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
pd.crosstab(df_kredyty.plec, df_kredyty.prognozowana)
df_kredyty.groupby(['plec'])['prognozowana'].mean()*100
df_kredyty.dropna(inplace=True, subset=['plec'])
ax=sns.countplot(x="plec",hue='prognozowana', data=df_kredyty).set(title = 'Rozkład prognozowanej od płci',
                                                           xlabel = 'Płeć', ylabel = 'Liczba klientów')
df_kredyty.plec.replace(['m','k'],[0,1], inplace=True)
df_kredyty.plec=df_kredyty.plec.astype('uint8')
cat_binary_columns.append('plec')
kat = pd.DataFrame(df_kredyty.posiada_samochod.value_counts())
kat.rename(columns={'posiada_samochod':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
pd.crosstab(df_kredyty.posiada_samochod, df_kredyty.prognozowana)
df_kredyty.groupby(['posiada_samochod'])['prognozowana'].mean()*100
df_kredyty.posiada_samochod=df_kredyty.posiada_samochod.astype('uint8')
cat_binary_columns.append('posiada_samochod')
kat = pd.DataFrame(df_kredyty.posiada_nieruchomosc.value_counts())
kat.rename(columns={'posiada_nieruchomosc':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
pd.crosstab(df_kredyty.posiada_nieruchomosc, df_kredyty.prognozowana)
df_kredyty.groupby(['posiada_nieruchomosc'])['prognozowana'].mean()*100
df_kredyty.posiada_nieruchomosc=df_kredyty.posiada_nieruchomosc.astype('uint8')
cat_binary_columns.append('posiada_nieruchomosc')
kat = pd.DataFrame(df_kredyty.telefon.value_counts())
kat.rename(columns={'telefon':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
df_kredyty.telefon=df_kredyty.telefon.astype('uint8')
kat = pd.DataFrame(df_kredyty.telefon_2.value_counts())
kat.rename(columns={'telefon_2':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
df_kredyty.groupby(['telefon_2'])['prognozowana'].mean()*100
df_kredyty.telefon_2=df_kredyty.telefon_2.astype('uint8')
kat = pd.DataFrame(df_kredyty.telefon_3.value_counts())
kat.rename(columns={'telefon_3':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
df_kredyty.groupby(['telefon_3'])['prognozowana'].mean()*100
df_kredyty.telefon_3=df_kredyty.telefon_3.astype('uint8')
kat = pd.DataFrame(df_kredyty.email.value_counts())
kat.rename(columns={'email':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
df_kredyty.groupby(['email'])['prognozowana'].mean()*100
df_kredyty.email=df_kredyty.email.astype('uint8')
cat_binary_columns=cat_binary_columns+['telefon', 'telefon_2', 'telefon_3', 'email', 'telefon_dostepny']
df_temp=df_kredyty[['zamieszkanie_flg_1','zamieszkanie_flg_2','zamieszkanie_flg_3','zamieszkanie_flg_4','zamieszkanie_flg_5','zamieszkanie_flg_6']].apply(pd.Series.value_counts)
df_temp['suma']=df_temp.sum(axis=1)
df_temp
df_kredyty[['zamieszkanie_flg_1','zamieszkanie_flg_2','zamieszkanie_flg_3','zamieszkanie_flg_4','zamieszkanie_flg_5','zamieszkanie_flg_6']].sum(axis=1).unique()
for variable in ['zamieszkanie_flg_1','zamieszkanie_flg_2','zamieszkanie_flg_3',
                 'zamieszkanie_flg_4','zamieszkanie_flg_5','zamieszkanie_flg_6']:
    kat = pd.DataFrame(df_kredyty[variable].value_counts())
    kat.rename(columns={variable:'Suma_obserwacji'}, inplace=True)
    kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
    df_kredyty[variable]=df_kredyty[variable].astype('uint8')
    print("Zestawienie obserwacji dla zmiennej "+variable)
    print(kat)
    print("\nUdział osób niespłacających kredytów w danej grupie dla zmiennej "+variable)
    print(df_kredyty.groupby([variable])['prognozowana'].mean()*100)
    print("\n"+10*"~")
df_kredyty['zamieszkanie_flg_sum']=df_kredyty[['zamieszkanie_flg_1','zamieszkanie_flg_2','zamieszkanie_flg_3','zamieszkanie_flg_4','zamieszkanie_flg_5','zamieszkanie_flg_6']].sum(axis=1)
kat = pd.DataFrame(df_kredyty.zamieszkanie_flg_sum.value_counts())
kat.rename(columns={'zamieszkanie_flg_sum':'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat
df_kredyty.groupby(['zamieszkanie_flg_sum'])['prognozowana'].mean()*100
cat_binary_columns=cat_binary_columns+['zamieszkanie_flg_1','zamieszkanie_flg_2','zamieszkanie_flg_3','zamieszkanie_flg_4','zamieszkanie_flg_5','zamieszkanie_flg_6']
for variable in ['doc_'+str(i) for i in range(2,22)]:
    kat = pd.DataFrame(df_kredyty[variable].value_counts())
    kat.rename(columns={variable:'Suma_obserwacji'}, inplace=True)
    kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
    kat['Procent_niesplacajacych']=df_kredyty.groupby(['zamieszkanie_flg_sum'])['prognozowana'].mean()*100
    df_kredyty[variable]=df_kredyty[variable].astype('uint8')
    print(40*"~"+"\nZestawienie obserwacji dla zmiennej "+variable)
    print(kat)

cat_binary_columns=cat_binary_columns+['doc_'+str(i) for i in range(2,22)]
variable='typ_pozyczki'
kat = pd.DataFrame(df_kredyty[variable].value_counts())
kat.rename(columns={variable:'Suma_obserwacji'}, inplace=True)
kat['Procent_obserwacji'] = kat['Suma_obserwacji']/df_kredyty.shape[0]*100
kat['Procent_niesplacajacych']=df_kredyty.groupby(variable)['prognozowana'].mean()*100
kat
df_kredyty.typ_pozyczki.replace(['typ_1','typ_2'],[0,1], inplace=True)
df_kredyty['typ_pozyczki']=df_kredyty['typ_pozyczki'].astype('uint8')
cat_binary_columns.append('typ_pozyczki')
plot_value_count('typ_os_tow',xtic_rot=0)
category_summary('typ_os_tow')
#df_kredyty.typ_os_tow.replace('NaN',0, inplace=True)
df_kredyty.dropna(subset=['typ_os_tow'], inplace=True)
df_kredyty.typ_os_tow.replace('XNA',0, inplace=True)
df_kredyty['typ_os_tow']=df_kredyty['typ_os_tow'].astype('uint8')
cat_multi_columns=['typ_os_tow']
#df_kredyty = pd.concat([df_kredyty, pd.get_dummies(df_kredyty.typ_os_tow, prefix='typ_os_tow__')], axis = 1)
df_kredyty['_typ_os_tow_binarnie']=df_kredyty.typ_os_tow.replace([1,2,3,4,5,6],1)
df_kredyty['_typ_os_tow_binarnie']=df_kredyty['_typ_os_tow_binarnie'].astype('uint8')
category_summary('_typ_os_tow_binarnie')
cat_binary_columns.append('_typ_os_tow_binarnie')
df_kredyty.rodzaj_zatrudnienia.fillna('brak_danych', inplace=True)
plot_value_count('rodzaj_zatrudnienia')
category_summary('rodzaj_zatrudnienia')
df_kredyty[df_kredyty['rodzaj_zatrudnienia']=='brak_danych']['zawod'].value_counts()
df_kredyty[df_kredyty['rodzaj_zatrudnienia']=='brak_danych']['dochod_roczny'].describe()
df_kredyty.rodzaj_zatrudnienia.replace('brak_danych',np.nan, inplace=True)
df_kredyty['_rodzaj_zatrudnienia']=df_kredyty.rodzaj_zatrudnienia.replace(['XNA', 'urlop', 'student'],'brak_danych')
df_kredyty['_rodzaj_zatrudnienia']=df_kredyty._rodzaj_zatrudnienia.replace('businessman','samozatrudniony')
#df_kredyty = pd.concat([df_kredyty, pd.get_dummies(df_kredyty.rodzaj_zatrudnienia, prefix='rodzaj_zatrudnienia__')], axis = 1)
#df_kredyty = pd.concat([df_kredyty, pd.get_dummies(df_kredyty._rodzaj_zatrudnienia, prefix='_rodzaj_zatrudnienia__')], axis = 1)
cat_multi_columns=cat_multi_columns+['rodzaj_zatrudnienia', '_rodzaj_zatrudnienia']
plot_value_count(x='branza', fig_size=(15,5))
category_summary('branza')
df_kredyty[df_kredyty['branza']=='XNA']['rodzaj_zatrudnienia'].value_counts()
df_kredyty.branza.replace('XNA', 'emerytura', inplace=True)
df_kredyty.groupby('branza')['dochod_roczny'].median()
#df_kredyty = pd.concat([df_kredyty, pd.get_dummies(df_kredyty.branza, prefix='branza__')], axis = 1)
cat_multi_columns.append('branza')
df_kredyty.zawod.fillna("brak_danych", inplace=True)
plot_value_count(x='zawod', fig_size=(11,5))
category_summary('zawod')
df_kredyty[df_kredyty['zawod']=='brak_danych']['rodzaj_zatrudnienia'].value_counts()
index_emerytow=list(df_kredyty[df_kredyty['zawod']=='brak_danych'][df_kredyty['rodzaj_zatrudnienia']=='emeryt rencista']['zawod'].index);
#zle indeksy trzeba zmienic
#df_kredyty.zawod.iloc[index_emerytow] = 'emeryt'
df_kredyty.zawod.replace("brak_danych",np.nan, inplace=True)
cat_multi_columns.append('zawod')
plot_value_count('wyksztalcenie', xtic_rot=45)
category_summary('wyksztalcenie')
#df_kredyty = pd.concat([df_kredyty, pd.get_dummies(df_kredyty.wyksztalcenie, prefix='wyksztalcenie__')], axis = 1)
df_kredyty['_wyksztalcenie_binarnie']=df_kredyty.wyksztalcenie.replace(['zawodowe / ogolne', 'podstawowe','wyzsze niukonczone'],1)
df_kredyty['_wyksztalcenie_binarnie']=df_kredyty._wyksztalcenie_binarnie.replace(['wyzsze', 'businessman'],0)
df_kredyty['_wyksztalcenie_binarnie']=df_kredyty['_wyksztalcenie_binarnie'].astype('uint8')
cat_multi_columns.append('wyksztalcenie')
cat_binary_columns.append('_wyksztalcenie_binarnie')
df_kredyty.stan_cywilny.fillna("brak_danych", inplace=True)
plot_value_count(x='stan_cywilny', fig_size=(6,4), xtic_rot=45)
category_summary('stan_cywilny')
df_kredyty[df_kredyty['stan_cywilny']=='brak_danych']['rodzina'].value_counts()
df_kredyty[df_kredyty['stan_cywilny']=='brak_danych']['liczba_dzieci'].value_counts()
#df_kredyty[df_kredyty['stan_cywilny']=='brak_danych'][df_kredyty['_rodzaj_zatrudnienia']=='emeryt rencista']['liczba_dzieci'].value_counts()
#df_kredyty = pd.concat([df_kredyty, pd.get_dummies(df_kredyty.stan_cywilny, prefix='stan_cywilny__')], axis = 1)
df_kredyty.stan_cywilny.replace("brak_danych",np.nan, inplace=True)
cat_multi_columns.append('stan_cywilny')
#df_kredyty.zakwaterowanie.fillna("brak_danych", inplace=True)
plot_value_count(x='zakwaterowanie', fig_size=(6,4), xtic_rot=45)
category_summary('zakwaterowanie')
#df_kredyty = pd.concat([df_kredyty, pd.get_dummies(df_kredyty.zakwaterowanie, prefix='zakwaterowanie__')], axis = 1)
cat_multi_columns.append('zakwaterowanie')
#df_kredyty.dzien_rozp_proc.fillna("brak_danych", inplace=True)
plot_value_count(x='dzien_rozp_proc', fig_size=(6,4), xtic_rot=45)
category_summary('dzien_rozp_proc')
cat_multi_columns.append('dzien_rozp_proc')
def describe_stats(variable):
    '''Statystyka opisowa danej zmiennej z df_kredyty 
    z uwzglednieniem podzialu na splacajacych i niesplacajacych'''
    stats=pd.DataFrame(df_kredyty[variable].describe())
    stats1=df_kredyty[df_kredyty['prognozowana']==1][variable].describe()
    stats0=df_kredyty[df_kredyty['prognozowana']==0][variable].describe()
    
    stats=stats.transpose()
    stats1=stats1.transpose()
    stats0=stats0.transpose()

    stats=stats.append(stats1)
    stats=stats.append(stats0)
    
    stats=stats.set_index([['wszyscy', 'niesplacajacy - 1', 'splacajcy - 0']])
    stats['skew']=[df_kredyty[variable].skew(), df_kredyty[df_kredyty['prognozowana']==1][variable].skew(),
                   df_kredyty[df_kredyty['prognozowana']==0][variable].skew()]
    
    return stats
#df_kredyty['kwota_kredytu'].apply(lambda x: np.log(x)).describe()
describe_stats('kwota_kredytu')
def plot_dist_box(variable, figsize=(8,5), bins=30):
    
    '''Funkcja do plotowania wykresow rozkladu zmiennych: histogramu, gestosci i pudelkowego'''
    
    df_kredyty_without_nan=df_kredyty.dropna(subset=[variable])
    
    # Histogram
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.4, style="whitegrid")
    sns.distplot(df_kredyty_without_nan[variable], kde = False, bins = bins).set(title = 'Histogram - '+variable, xlabel = variable, ylabel = 'liczba obserwacji')
    plt.show()

    # Histogram z nalozonymi na siebie wartosciami dla splacajacych i nie splacajacych
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.4, style="whitegrid")
    for a in [0, 1]:
        sns.distplot(df_kredyty_without_nan[df_kredyty_without_nan['prognozowana']==a][variable], bins=bins, ax=ax, kde=False, label="prognozowana "+str(a)).set(title = 'Histogram z podziałem na spłacających i niespłacających - '+variable, xlabel = variable, ylabel = 'liczba obserwacji')
    plt.legend()
    plt.show()

    # Wykres gęstości
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.4, style="whitegrid")
    sns.kdeplot(df_kredyty_without_nan[variable], shade = True).set(title = 'Wykres gęstości - '+variable, xlabel = variable, ylabel = '')
    plt.show()

    # Wykres pudełkowy
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.4, style="whitegrid")
    sns.boxplot(df_kredyty_without_nan[variable]).set(title = 'Wykres pudełkowy - '+variable, xlabel = variable, ylabel = '')
    plt.show()
plot_dist_box('kwota_kredytu', bins=50)
num_columns=['kwota_kredytu']
df_kredyty.dropna(subset=['rata'], inplace=True);
describe_stats('rata')
plot_dist_box('rata', bins=50)
num_columns.append('rata')
describe_stats('dochod_roczny')
plot_dist_box('dochod_roczny', bins=30)
num_columns.append('dochod_roczny')
upper_band=df_kredyty.dochod_roczny.quantile(0.99)
df_kredyty['dochod_roczny_2']=df_kredyty[df_kredyty['dochod_roczny']<upper_band]['dochod_roczny']
describe_stats('dochod_roczny_2')
plot_dist_box('dochod_roczny_2', bins=30)
df_kredyty.drop(['dochod_roczny_2'], axis=1);
num_columns.append('dochod_roczny_2')
describe_stats('wart_dobr')
plot_dist_box('wart_dobr', bins=30)
num_columns.append('wart_dobr')
category_summary('liczba_dzieci')
num_columns.append('liczba_dzieci')
df_kredyty['_liczba_dzieci']=df_kredyty.liczba_dzieci.replace([3,4,5,6,7,8,9,10,11,12,14,19], '3_i_wiecej')
category_summary('_liczba_dzieci')
cat_multi_columns.append('_liczba_dzieci')
category_summary('rodzina')
num_columns.append('rodzina')
df_kredyty['_rodzina']=df_kredyty.rodzina.replace([5.0,6.0,7.0,8.0,9.0,10.0,12.0,13.0,14.0,16.0,20.0], '5_i_wiecej')
category_summary('_rodzina')
cat_multi_columns.append('_rodzina')
df_kredyty.dropna(subset=['rodzina'], inplace=True)
df_kredyty.wiek_samochodu.fillna('brak_danych', inplace=True)
category_summary('wiek_samochodu').head(5)
df_kredyty[df_kredyty['wiek_samochodu']=='brak_danych'][df_kredyty['posiada_samochod']==1]['posiada_samochod'].sum()
indx_do_wyrzucenia=list(df_kredyty[df_kredyty['wiek_samochodu']=='brak_danych'][df_kredyty['posiada_samochod']==1]['posiada_samochod'].index)
df_kredyty.drop(index=indx_do_wyrzucenia, inplace=True)
df_kredyty.wiek_samochodu.replace('brak_danych', np.nan, inplace=True)
describe_stats('wiek_samochodu')
plot_dist_box('wiek_samochodu', bins=80)
num_columns.append('wiek_samochodu')
df_kredyty['_wiek_samochodu']=pd.qcut(df_kredyty.wiek_samochodu, q=5)
df_kredyty['_wiek_samochodu'].replace(np.nan,'brak_samochodu', inplace=True)
cat_multi_columns.append('_wiek_samochodu')
df_kredyty['wiek']=df_kredyty.urodzony.apply(lambda x: round(x/365,1))
df_kredyty.drop(['urodzony'],axis=1, inplace=True)
describe_stats('wiek')
plot_dist_box('wiek', bins=30)
num_columns.append('wiek')
df_kredyty.zatrudniony_od.fillna('brak_danych', inplace=True)
category_summary('zatrudniony_od').head()
df_kredyty[df_kredyty['zatrudniony_od']=='brak_danych']['rodzaj_zatrudnienia'].value_counts()
df_kredyty.zatrudniony_od.replace('brak_danych', np.nan, inplace=True)
df_kredyty['zatrudnienie']=df_kredyty.zatrudniony_od.apply(lambda x: -round(x/365,1))
df_kredyty.drop(['zatrudniony_od'], axis=1, inplace=True )
describe_stats('zatrudnienie')

plot_dist_box('zatrudnienie', bins=30)
num_columns.append('zatrudnienie')
df_kredyty['_zatrudnienie']=pd.qcut(df_kredyty.zatrudnienie, q=5)
df_kredyty['_zatrudnienie'].replace(np.nan,'emeryt', inplace=True)
cat_multi_columns.append('_zatrudnienie')
df_kredyty['zarejestrowany']=df_kredyty.zarejestrowany.apply(lambda x: round(x/365,1))
describe_stats('zarejestrowany')
plot_dist_box('zarejestrowany', bins=30)
num_columns.append('zarejestrowany')
df_kredyty.dropna(axis=0, subset=['ostatnia_zmiana_telefonu'], inplace=True)
df_kredyty['ostatnia_zmiana_telefonu']=df_kredyty.ostatnia_zmiana_telefonu.apply(lambda x: round(x/365,2))
describe_stats('ostatnia_zmiana_telefonu')
plot_dist_box('ostatnia_zmiana_telefonu', bins=100)
df_kredyty[df_kredyty['prognozowana']==0]['ostatnia_zmiana_telefonu'].quantile(0.1)*365
df_kredyty[df_kredyty['prognozowana']==1]['ostatnia_zmiana_telefonu'].quantile(0.1)*365
num_columns.append('ostatnia_zmiana_telefonu')
describe_stats('ocena_1')
category_summary('ocena_1').head()
plot_dist_box('ocena_1', bins=20)
category_summary('ocena_2')
plot_value_count(x='ocena_2', fig_size=(6,4), xtic_rot=0)
category_summary('ocena_3')
plot_value_count(x='ocena_3', fig_size=(6,4), xtic_rot=0)
num_columns=num_columns+['ocena_1', 'ocena_2', 'ocena_3']
describe_stats('wskaznik_1')
category_summary('wskaznik_1')
plot_value_count('wskaznik_1')
describe_stats('wskaznik_2')
category_summary('wskaznik_2')
plot_value_count('wskaznik_2')
describe_stats('wskaznik_3')
category_summary('wskaznik_3')
plot_value_count('wskaznik_3')
describe_stats('wskaznik_4')
category_summary('wskaznik_4')
plot_value_count('wskaznik_4')
num_columns_imput=['wskaznik_1', 'wskaznik_2', 'wskaznik_3', 'wskaznik_4']
num_columns=num_columns+['wskaznik_1', 'wskaznik_2', 'wskaznik_3', 'wskaznik_4']
df_kredyty.wskaznik_zew_1.fillna('brak_danych', inplace=True)
df_kredyty[df_kredyty['wskaznik_zew_1']=='brak_danych'][df_kredyty['posiada_samochod']==1]['posiada_samochod'].sum()
category_summary('wskaznik_zew_1').head()
df_kredyty.wskaznik_zew_1.replace('brak_danych', np.nan, inplace=True)
describe_stats('wskaznik_zew_1')
plot_dist_box('wskaznik_zew_1')
describe_stats('wskaznik_zew_2')
plot_dist_box('wskaznik_zew_2')
df_kredyty.wskaznik_zew_3.fillna('brak_danych', inplace=True)
df_kredyty[df_kredyty['wskaznik_zew_3']=='brak_danych']['rodzaj_zatrudnienia'].value_counts()
category_summary('wskaznik_zew_3').head()
df_kredyty.wskaznik_zew_3.replace('brak_danych', np.nan, inplace=True)
describe_stats('wskaznik_zew_3')
plot_dist_box('wskaznik_zew_3')
num_columns_imput.append('wskaznik_zew_2')
num_columns_to_drop=['wskaznik_zew_1', 'wskaznik_zew_3']
num_columns=num_columns+['wskaznik_zew_1','wskaznik_zew_3','wskaznik_zew_2']
#df_kredyty.liczba_prob_g.fillna('brak_danych', inplace=True)
#df_kredyty[df_kredyty['liczba_prob_g']=='brak_danych']['zawod'].value_counts()
#df_kredyty.liczba_prob_g.replace('brak_danych', np.nan, inplace=True)
category_summary('liczba_prob_g')
category_summary('liczba_prob_d')
category_summary('liczba_prob_t')
category_summary('liczba_prob_m')
plot_dist_box('liczba_prob_m')
category_summary('liczba_prob_k')
plot_dist_box('liczba_prob_k')
category_summary('liczba_prob_r')
describe_stats('liczba_prob_r')
plot_dist_box('liczba_prob_r', bins=20)
num_columns=num_columns+['liczba_prob_g','liczba_prob_d', 'liczba_prob_t', 'liczba_prob_k', 'liczba_prob_r']
num_columns_to_drop=num_columns_to_drop+['liczba_prob_g','liczba_prob_d', 'liczba_prob_t', 'liczba_prob_k']
num_columns_imput.append('liczba_prob_r')
category_summary('godz_rozp_proc')
describe_stats('godz_rozp_proc')
plot_dist_box('godz_rozp_proc', bins=24)
num_columns.append('godz_rozp_proc')
category_columns=cat_binary_columns+cat_multi_columns
category_columns.remove('telefon')
numeric_columns=num_columns+['prognozowana']
corr_num_y = pd.DataFrame(df_kredyty[numeric_columns].corr('spearman'),
columns = df_kredyty[numeric_columns].columns,
index = df_kredyty[numeric_columns].columns).loc['prognozowana', :]
corr_num_y = pd.DataFrame(corr_num_y)
corr_num_y=corr_num_y.drop(index='prognozowana')
corr_num_temp=corr_num_y
corr_num_y.reset_index(inplace=True)
 
# wykres słupkowy współczynnika korelacji
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)
sns.barplot(data = corr_num_y.sort_values('prognozowana', ascending=False), x = 'prognozowana', y = 'index', palette = 'Blues_d').set(title = 'Wykres słupkowy - współczynnik korelacji \n zmiennych numerycznych ze zmienną celu', xlabel = 'współczynnik korelacji', ylabel = 'nazwa zmiennej')
plt.show()

corr_num_y['abs']= corr_num_y['prognozowana'].abs()
 
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.barplot(data = corr_num_y.sort_values('abs', ascending=False), x = 'abs', y = 'index', color = '#eb6c6a').set(title = 'Wykres słupkowy - współczynnik korelacji \n zmiennych numerycznych ze zmienną celu \n (wartości bezwzględne).', xlabel = 'współczynnik korelacji', ylabel = 'nazwa zmiennej')
plt.show()

import scipy
def CramersV(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = sum(confusion_matrix.sum())
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
def CalculateCrammersV(tab):
    ret = []
    for m in tab:
        row = []
        for n in tab:
            cross_tab = pd.crosstab(tab[m].values,tab[n].values)
            #print(str(m)+ ' '+str(n))
            row.append(CramersV(cross_tab))
        ret.append(row)
    return pd.DataFrame(ret, columns=tab.columns, index=tab.columns)

cat_y = pd.DataFrame(CalculateCrammersV(df_kredyty[category_columns]).loc['prognozowana', :])
cat_y=cat_y.drop(index='prognozowana')
cat_y.reset_index(inplace = True)

plt.figure(figsize=(10,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.barplot(data = cat_y.sort_values('prognozowana', ascending=False), y = 'prognozowana', x = 'index', palette = ['#eb6c6a', '#f0918f', '#f2a3a2', '#f5b5b4', '#f7c8c7']).set(title = 'Wykres słupkowy - współczynnik V Crammera', xlabel = 'zmienne kategoryczne', ylabel = 'Crammer\'s V')
plt.xticks(rotation=90)
plt.show()
corr_num = pd.DataFrame(df_kredyty[numeric_columns].corr('spearman'),
columns = df_kredyty[numeric_columns].columns,
index = df_kredyty[numeric_columns].columns)



plt.figure(figsize=(15,6))
sns.set(font_scale=1)
sns.heatmap(corr_num.abs(), cmap="Reds", linewidths=.5).set(title='Heatmap-a współczynnika korelacji rang Spearmana')
plt.show()

crammer=CalculateCrammersV(df_kredyty[category_columns])

plt.figure(figsize=(15,6))
sns.set(font_scale=1.4)
sns.heatmap(crammer, cmap="Reds", linewidths=.5).set(title='Heatmap-a współczynnika zależności Crammera')
plt.show()

numeric_columns.remove('prognozowana')
import scipy.stats
import statsmodels.formula.api as sm
import statsmodels.api as sm_api

# dzielę zmienne na dwie grupy
cat_cols = category_columns
num_cols = numeric_columns

# w pętli buduję kolejne modele regresji liniowej
cols = []
for cat in cat_cols:
    rows = []
    for num in num_cols:
        formula = num + '~' +cat
        model = sm.ols(formula=formula,data=df_kredyty, missing='drop',)
        rows.append(np.sqrt(model.fit().rsquared))
    cols.append(rows)
corr_num_cat = pd.DataFrame(cols, index = cat_cols, columns = num_cols)


# wykres zależności
plt.figure(figsize=(15,6))
sns.set(font_scale=1.4)
sns.heatmap(corr_num_cat, cmap="Reds", linewidths=.5).set(title='Heatmap-a współczynnika korelacji wielorakiej')
plt.show()

num_columns_to_drop=['wskaznik_zew_1', 'wskaznik_zew_3', 'liczba_prob_g', 'liczba_prob_d', 'liczba_prob_t', 'liczba_prob_k', 'liczba_dzieci', 'rodzina', 'wiek_samochodu', 'zatrudnienie', 'dochod_roczny_2']

df_kredyty.drop( num_columns_to_drop, axis=1, inplace=True)
for x in num_columns_to_drop:
    num_columns.remove(x)
category_columns_to_stay=['_zatrudnienie', 'zawod', 'branza', '_rodzaj_zatrudnienia','plec', 'wyksztalcenie', 'zamieszkanie_flg_5','doc_20','zamieszkanie_flg_4','zakwaterowanie','_wiek_samochodu','typ_pozyczki','zamieszkanie_flg_6','stan_cywilny','doc_17','telefon_2','telefon_3','_liczba_dzieci']
cat_multi_columns=list(set(cat_multi_columns) & set(category_columns_to_stay))
category_columns=category_columns_to_stay
sum_columns=category_columns+num_columns+['prognozowana']
df_kredyty=df_kredyty[sum_columns]
df_kredyty2=df_kredyty.copy()
#df_kredyty=df_kredyty2.dropna()
df_kredyty.zawod.fillna("brak_danych", inplace=True)
df_kredyty.stan_cywilny.fillna("brak_danych", inplace=True)

for feature in cat_multi_columns:
    df_kredyty = pd.concat([df_kredyty, pd.get_dummies(df_kredyty[feature], prefix=feature+'__')], axis = 1)
    df_kredyty.drop(feature, axis=1,inplace=True)
q1 = df_kredyty[num_columns].quantile(0.25)
q3 = df_kredyty[num_columns].quantile(0.75)
iqr = q3 - q1
#low_boundary = (q1 - 1.5 * iqr)
#upp_boundary = (q3 + 1.5 * iqr)
low_boundary = df_kredyty[num_columns].quantile(0.0005)
upp_boundary = df_kredyty[num_columns].quantile(0.9995)
num_of_outliers_L = (df_kredyty[num_columns][iqr.index] < low_boundary).sum()
num_of_outliers_U = (df_kredyty[num_columns][iqr.index] > upp_boundary).sum()
outliers = pd.DataFrame({'lower_boundary':low_boundary, 'upper_boundary':upp_boundary,'num_of_outliers_L':num_of_outliers_L, 'num_of_outliers_U':num_of_outliers_U})
outliers
w=df_kredyty[num_columns].copy()
for row in outliers.iterrows():
    w = w[(w[row[0]] >= row[1]['lower_boundary']) & (w[row[0]] <= row[1]['upper_boundary'])]
w.shape[0]/df_kredyty[num_columns].shape[0]
from sklearn.model_selection import train_test_split
X=df_kredyty.drop('prognozowana', axis=1)
y=df_kredyty['prognozowana']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.20,
                                                   random_state=23)
X_train.sample(10)
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_num=X_train[num_columns]
X_test_num=X_test[num_columns]
X_train = pd.concat([X_train.drop(labels=num_columns, axis=1),pd.DataFrame(scaler.fit_transform(X_train_num), columns=X_train_num.columns)], axis=1)
X_test = pd.concat([X_test.drop(labels=num_columns, axis=1),pd.DataFrame(scaler.transform(X_test_num), columns=X_test_num.columns)], axis=1)

# zrodlo mateuszgrzyb.pl
from sklearn.model_selection import cross_val_score
def cv_and_score_model(model, x, y, prints):
    """Model scorer. Calculates: average gini score, average recall and stability.

    Parameters:
    -----------
    model: sklearn predictive model, model that will be scored
    
    x : pandas DataFrame, set of x-features

    y : pandas Series, target feature
    """
    
    cv_auc = cross_val_score(model, x, y, cv = 5, scoring = 'roc_auc')
    cv_recall = cross_val_score(model, x, y, cv = 5, scoring = 'recall')
    
    # Calculate Gini score based on AUC.
    cv_gini = (cv_auc * 2) - 1 
    
    # Printing results.
    if prints:
        print('Average Gini: {}.'.format(np.mean(cv_gini).round(3)))
        print('Average AUC: {}.'.format(np.mean(cv_auc).round(3)))
        print('Average recall: {}.'.format(np.mean(cv_recall).round(3)))
        print('Stability: {}%'.format((100 - np.std(cv_gini)*100/cv_gini.mean()).round(3)))

    return cv_gini
#zrodlo https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
#zrodlo mateuszgrzyb.pl
def test_model(model, features,x_tr,y_tr,x_te,y_te, plots):
    """Model scorer. Calculates: average gini score, average recall and stability.

    Parameters:
    -----------
    model: sklearn predictive model, model that will be tested
    
    plots : bool, decission whether to print plots
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, recall_score
    
    
    model.fit(x_tr[features], y_tr)
    y_pred = model.predict(x_te[features])
    

    gini_score = (2* roc_auc_score(y_te, y_pred))-1
    
    # calculate FPR and TPR
    fpr, tpr, thresholds = roc_curve(y_te, y_pred)
    
    print('Gini score: {}.'.format(gini_score.round(3)))
    print('AUC score: {}.'.format(roc_auc_score(y_te, y_pred).round(3)))
    print('Recall score: {}.'.format(recall_score(y_te, y_pred).round(3)))
    
    if plots == True:
        # ROC CURVE
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='blue',
                 lw=lw, label='ROC curve (area under curve = %0.3f)' % roc_auc_score(y_te, y_pred))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
        
        print_confusion_matrix(confusion_matrix(y_te,y_pred),['0','1'])
        plt.show()
from sklearn.tree import DecisionTreeClassifier
model_cart_0 = DecisionTreeClassifier(class_weight='balanced')
cv_and_score_model(model_cart_0, X_train, y_train, True)
model_cart_1 = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
cv_and_score_model(model_cart_1, X_train, y_train, True)
#zrodlo: kaggle.com
'''
from sklearn.feature_selection import RFECV


clf_rf_4 = DecisionTreeClassifier(max_depth=10, class_weight="balanced") 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='roc_auc')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])
features_refcv=list(X_train.columns[rfecv.support_])
'''
features_refcv=['wyksztalcenie___wyzsze', '_zatrudnienie___(10.1, 49.1]','kwota_kredytu', 'rata', 'wart_dobr', 'wiek', 'zarejestrowany', 'ostatnia_zmiana_telefonu', 'wskaznik_zew_2']
model_cart_2 = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
cv_and_score_model(model_cart_2, X_train[features_refcv], y_train, True)
test_model(model_cart_2, features_refcv,X_train,y_train,X_test,y_test, True)
'''
from sklearn.model_selection import GridSearchCV
parameters = {'criterion':('entropy', 'gini'), 'splitter':('best','random'), 'max_depth':np.arange(1,10), 'min_samples_split':np.arange(2,10), 'min_samples_leaf':np.arange(1,5)}
classifier = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), parameters, cv=5)
classifier.fit(X_train, y_train)
classifier.best_params_
'''
from  sklearn.linear_model import SGDClassifier
model_SGD_0=SGDClassifier(class_weight='balanced')
cv_and_score_model(model_SGD_0, X_train, y_train, True)
model_SGD_1=SGDClassifier(max_iter=20,class_weight='balanced')
cv_and_score_model(model_SGD_1, X_train, y_train, True)
model_SGD_2 = SGDClassifier(max_iter=20,class_weight='balanced')
cv_and_score_model(model_SGD_2, X_train[features_refcv], y_train, True)
test_model(model_SGD_2, features_refcv, X_train,y_train,X_test,y_test, True)
#df_BIK=df_BIK.join(df_kredyty.prognozowana, on='id', how='left')
'''category_BIK_columns=['status', 'typ', 'prognozowana', 'waluta_obca']
numeric_BIK_columns=list(set(list(df_BIK.columns))-set(category_BIK_columns))+['prognozowana']'''
'''cat_y = pd.DataFrame(CalculateCrammersV(df_BIK[category_BIK_columns]).loc['prognozowana', :])
cat_y=cat_y.drop(index='prognozowana')
cat_y.reset_index(inplace = True)

plt.figure(figsize=(8,6))
sns.set(font_scale=1.4, style="whitegrid")
sns.barplot(data = cat_y.sort_values('prognozowana', ascending=False), y = 'prognozowana', x = 'index', palette = ['#eb6c6a', '#f0918f', '#f2a3a2', '#f5b5b4', '#f7c8c7']).set(title = 'Wykres słupkowy - współczynnik V Crammera', xlabel = 'zmienne kategoryczne', ylabel = 'Crammer\'s V')
plt.xticks(rotation=0)
plt.show()'''
'''corr_num_y = pd.DataFrame(df_BIK[numeric_BIK_columns].corr('spearman'),
columns = df_BIK[numeric_BIK_columns].columns,
index = df_BIK[numeric_BIK_columns].columns).loc['prognozowana', :]
corr_num_y = pd.DataFrame(corr_num_y)
corr_num_y=corr_num_y.drop(index='prognozowana')
corr_num_temp=corr_num_y
corr_num_y.reset_index(inplace=True)

corr_num_y['abs']= corr_num_y['prognozowana'].abs()
 
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.barplot(data = corr_num_y.sort_values('abs', ascending=False), x = 'abs', y = 'index', color = '#eb6c6a').set(title = 'Wykres słupkowy - współczynnik korelacji \n zmiennych numerycznych ze zmienną celu \n (wartości bezwzględne).', xlabel = 'współczynnik korelacji', ylabel = 'nazwa zmiennej')
plt.show()
'''
'''df_BIK3=df_BIK.copy()
df_BIK3.dropna(subset=['planowane_zakonczenie'], inplace=True)'''
'''bik={'id':df_BIK3['id'].unique(),
     'bik_dni_od_aplikacji':[df_BIK3[df_BIK3['id']==j]['dni_od_aplikacji'].max() for j in df_BIK3['id'].unique()],
    'bik_planowane_zakonczenie':[df_BIK3[df_BIK3['id']==j]['planowane_zakonczenie'].max() for j in df_BIK3['id'].unique()],
    'bik_typ':[df_BIK3[df_BIK3['id']==j]['typ'].max() for j in df_BIK3['id'].unique()],
    'bik_status':[df_BIK3[df_BIK3['id']==j]['status'].max() for j in df_BIK3['id'].unique()]}
df_bik_features=pd.DataFrame(data=bik_features)
'''
#df_kredyty2=df_kredyty.join(df_bik_features.bik_typ, on='id', how='left',sort=True)