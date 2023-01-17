import numpy   as np

import pandas  as pd

import seaborn as sb

import matplotlib.pyplot as plt

import missingno as msno

import unidecode
train = pd.read_csv("../input/murcia-car-challenge/train.csv",            index_col="Id")

test  = pd.read_csv("../input/murcia-car-challenge/test.csv",             index_col="Id")

sub   = pd.read_csv("../input/murcia-car-challenge/sampleSubmission.csv", index_col="Id")
def cleanString(valor):

    return unidecode.unidecode(valor.upper().strip())



def clean(data):

    data['Marca']  = data['Marca'].apply(cleanString)

    data['Modelo'] = data['Modelo'].apply(cleanString)



    data['Marca'].replace("MERCEDES", "MERCEDES-BENZ",     inplace=True)

    data['Modelo'].replace("SERIE ", "SERIE_", regex=True, inplace=True)

    data['Modelo'].replace("CLASE ", "CLASE_", regex=True, inplace=True)

    data['Modelo'].replace("RANGE ROVER ", "", regex=True, inplace=True)

    

    return data



train = clean(train)

test  = clean(test)
def tranform2type(Tiempo):

    if   Tiempo.endswith('días'):  return 'dias'

    elif Tiempo.endswith('horas'): return 'horas'

    elif Tiempo.endswith('hora'):  return 'hora'

    elif Tiempo.endswith('min'):   return 'min'

    elif Tiempo.endswith('seg'):   return 'seg'

    elif Tiempo.endswith('nuevo anuncio'):  return 'nuevo anuncio'

    elif Tiempo.endswith('destacado'):      return 'destacado'



    

def tranform2dias(Tiempo):

    if   Tiempo.endswith('días'):  return int(Tiempo.replace(' días', ''))

    elif Tiempo.endswith('horas'): return int(Tiempo.replace(' horas', ''))/24

    elif Tiempo.endswith('hora'):  return int(Tiempo.replace(' hora', ''))/24

    elif Tiempo.endswith('min'):   return int(Tiempo.replace(' min', ''))/(60*24)

    elif Tiempo.endswith('seg'):   return int(Tiempo.replace(' seg', ''))/(60*60*24)

    elif Tiempo.endswith('nuevo anuncio'):  return 0

    elif Tiempo.endswith('destacado'):      return -1



train["Tiempo_días"]  = train["Tiempo"].apply(tranform2dias) # Variable numérica

train["Tiempo_tipo"]  = train["Tiempo"].apply(tranform2type) # Variable categórica

train["Modelo_1st"]   = train.Modelo.str.split().str.get(0)  # Variable categórica
msno.bar(train);
msno.matrix(train);
# Variables numéricas

def plot_num(variable, title="", min=False, max=False, zeros=True, points=True, opacity=.3, size=(16,4)):

    if not zeros:

        variable=variable[variable!=0]

        title += " (no zeros)"

    if min:

        variable = variable[variable >= min]

        title += " (min: "+str(min)+")"

    if max:

        variable = variable[variable <= max]

        title += " (max: "+str(max)+")"

    fig, ax = plt.subplots(figsize=size)

    ax.set_title(title, fontsize=20)

    ax2 = ax.twinx()

    sb.violinplot(variable, cut=0, palette="Set3", inner="box", ax=ax)

    if points:

        sb.scatterplot(variable, y=variable.index, color="grey", linewidth=0, s=20, alpha=opacity, ax=ax2).invert_yaxis()





# Variables ordinales

def plot_ord(variable, title="", min=False, max=False, zeros=True, size=(16,4)):

    if not zeros:

        variable=variable[variable!=0]

        title += " (no zeros)"

    if min:

        variable = variable[variable >= min]

        title += " (min: "+str(min)+")"

    if max:

        variable = variable[variable <= max]

        title += " (max: "+str(max)+")"

    plt.figure(figsize=size)

    sb.countplot(variable, color='royalblue').set_title(title, fontsize=20);





# Variables categoricas

def plot_cat(variable, title="", top=False, normalize=False, dropna=False, size=(16,4)):

    plt.figure(figsize=size)

    nuiques = str(variable.nunique())

    cats = variable.value_counts(normalize=normalize, dropna=dropna)

    if top:

        cats = cats[:top]

        title += " (top "+str(top)+" de "+nuiques+")"

    else:

        title += " ("+nuiques+")"

    sb.barplot(x=cats, y=cats.index).set_title(title, fontsize=20);
plot_num(train.Precio, title="Precio en Euros", max=100000, points=False)
plot_num(np.log(train.Precio), title="Logaritmo del Precio en Euros", points=False)
plot_num(np.log(train.Precio), title="Logaritmo del Precio en Euros", points=True)
plot_cat(train.Marca, title="Marcas", size=(16,20))
plot_cat(train.Modelo_1st, title="Modelos primera palabra", top=100, size=(16,20))
wv = train[train.Marca=="VOLKSWAGEN"]['Modelo']

plot_cat(wv, title="Modelos VOLKSWAGEN", top=50, size=(16,20))
golfs = train[train.Modelo.str.contains("GOLF")]

plot_cat(golfs.Modelo, title="Modelos WV GOLF", top=20, size=(16,10))
serie3 = train[train.Modelo.str.contains("SERIE_3")]

plot_cat(serie3.Modelo, title="Modelos distintos de BMW SERIE 3", top=20, size=(16,10))
plot_cat(train.Provincia, title="Provincias", size=(16,12))
plot_cat(train.Localidad, title="Localidad", top=50, size=(16,12))
plot_num(train.Tiempo_días, title="Tiempo del anuncio: Días", opacity=.05)
plot_num(train.Tiempo_días, title="Tiempo del anuncio: Días", max=100, points=False)
plot_cat(train.Tiempo_tipo, title="Tiempo del anuncio: Tipo")
plot_ord(train.Año, title="Año del coche")
plot_ord(train.Año, title="Año del coche", min=1990)
plot_num(train.Kms, title="Kilómetros")
plot_num(train.Kms, max=1000000, title="Kilómetros", points=False)
plot_cat(train.Cambio, title="Tipo de cambio de marchas")
plot_num(train.Cv, max=400, title="Caballos", points=False)
plot_cat(train.Combust, title="Tipo de combustible")
plot_ord(train.Puertas, title="Número de puertas")
plot_cat(train.Vendedor, title="Tipo de vendedor")