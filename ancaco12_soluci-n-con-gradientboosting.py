# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

from sklearn import metrics

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import OneHotEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
n_test= 49999

fichero = 'datos_entrenamiento.csv'

tests = 'entrega_para_predecir.csv'

resultados_finales = 'resultados_finales_test.csv'

sample = 'resultados_finales_sampleSubmission.csv'

path_dir = '../input/'
# autor: https://www.kaggle.com/terminus7/pokemon-challenge

def calculate_effectiveness(data):

    '''

        this function creates a new column of each pokemon's effectiveness against it's enemy.

        every effectiveness starts with 1, if an effective type is found on enemy's type, effectiveness * 2

        if not very effective is found on enemy's type, effectiveness / 2

        if not effective is found on enemy's type, effectiveness * 0

        This function creates 4 new columns

            1. P1_type1, pokemon 1 first type effectiveness against the enemy's type

            2. P1_type2, pokemon 1 second type effectiveness against the enemy's type

            3. P2_type1, pokemon 2 first type effectiveness against the enemy's type

            4. P2_type2, pokemon 2 second type effectiveness against the enemy's type

    '''



    very_effective_dict = {'Normal': [],

                           'Fighting': ['Normal', 'Rock', 'Steel', 'Ice', 'Dark'],

                           'Flying': ['Fighting', 'Bug', 'Grass'],

                           'Poison': ['Grass', 'Fairy'],

                           'Ground': ['Poison', 'Rock', 'Steel', 'Fire', 'Electric'],

                           'Rock': ['Flying', 'Bug', 'Fire', 'Ice'],

                           'Bug': ['Grass', 'Psychic', 'Dark'],

                           'Ghost': ['Ghost', 'Psychic'],

                           'Steel': ['Rock', 'Ice', 'Fairy'],

                           'Fire': ['Bug', 'Steel', 'Grass', 'Ice'],

                           'Water': ['Ground', 'Rock', 'Fire'],

                           'Grass': ['Ground', 'Rock', 'Water'],

                           'Electric': ['Flying', 'Water'],

                           'Psychic': ['Fighting', 'Poison'],

                           'Ice': ['Flying', 'Ground', 'Grass', 'Dragon'],

                           'Dragon': ['Dragon'],

                           'Dark': ['Ghost', 'Psychic'],

                           'Fairy': ['Fighting', 'Dragon', 'Dark'],

                           'None': []}



    not_very_effective_dict = {'Normal': ['Rock', 'Steel'],

                               'Fighting': ['Flying', 'Poison', 'Bug', 'Psychic', 'Fairy'],

                               'Flying': ['Rock', 'Steel', 'Electric'],

                               'Poison': ['Poison', 'Rock', 'Ground', 'Ghost'],

                               'Ground': ['Bug', 'Grass'],

                               'Rock': ['Fighting', 'Ground', 'Steel'],

                               'Bug': ['Fighting', 'Flying', 'Poison', 'Ghost', 'Steel', 'Fire', 'Fairy'],

                               'Ghost': ['Dark'],

                               'Steel': ['Steel', 'Fire', 'Water', 'Electric'],

                               'Fire': ['Rock', 'Fire', 'Water', 'Dragon'],

                               'Water': ['Water', 'Grass', 'Dragon'],

                               'Grass': ['Flying', 'Poison', 'Bug', 'Steel', 'Fire', 'Grass', 'Dragon'],

                               'Electric': ['Grass', 'Electric', 'Dragon'],

                               'Psychic': ['Steel', 'Psychic'],

                               'Ice': ['Steel', 'Fire', 'Water', 'Psychic'],

                               'Dragon': ['Steel'],

                               'Dark': ['Fighting', 'Dark', 'Fairy'],

                               'Fairy': ['Posion', 'Steel', 'Fire'],

                               'None': []}



    not_effective_dict = {'Normal': ['Ghost'],

                          'Fighting': ['Ghost'],

                          'Flying': [],

                          'Poison': ['Steel'],

                          'Ground': ['Flying'],

                          'Rock': [],

                          'Bug': [],

                          'Ghost': ['Normal'],

                          'Steel': [],

                          'Fire': [],

                          'Water': [],

                          'Grass': [],

                          'Electric': ['Ground'],

                          'Psychic': ['Dark'],

                          'Ice': [],

                          'Dragon': ['Fairy'],

                          'Dark': [],

                          'Fairy': [],

                          'None': []}



    p1_type1_list = []

    p1_type2_list = []

    p2_type1_list = []

    p2_type2_list = []



    for row in data.itertuples():

        nested_type = [[1, 1], [1, 1]]



        tipos_pok_1 = [row.tipo1_id1, row.tipo2_id1]

        tipos_pok_2 = [row.tipo1_id2, row.tipo2_id2]



        # manipulating values if found on dictionary

        for i in range(0, 2):

            for j in range(0, 2):

                if tipos_pok_2[j] in very_effective_dict.get(tipos_pok_1[i]):

                    nested_type[0][i] *= 2

                if tipos_pok_2[j] in not_very_effective_dict.get(tipos_pok_1[i]):

                    nested_type[0][i] /= 2

                if tipos_pok_2[j] in not_effective_dict.get(tipos_pok_1[i]):

                    nested_type[0][i] *= 0



                if tipos_pok_1[j] in very_effective_dict.get(tipos_pok_2[i]):

                    nested_type[1][i] *= 2

                if tipos_pok_1[j] in not_very_effective_dict.get(tipos_pok_2[i]):

                    nested_type[1][i] /= 2

                if tipos_pok_1[j] in not_effective_dict.get(tipos_pok_2[i]):

                    nested_type[1][i] *= 0



        p1_type1_list.append(nested_type[0][0])

        p1_type2_list.append(nested_type[0][1])

        p2_type1_list.append(nested_type[1][0])

        p2_type2_list.append(nested_type[1][1])



    data = data.assign(P1_type1=p1_type1_list, P1_type2=p1_type2_list,

                       P2_type1=p2_type1_list, P2_type2=p2_type2_list)



    return data
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------DATASET-----------------------------------------------------------

def get_data():



    #-------pokemon.csv

    df_pokemon = pd.read_csv(path_dir + 'pokemon.csv')

    df_pokemon = df_pokemon.fillna({'Name': 'None', 'Type 1': 'None', 'Type 2': 'None'})

    #df_pokemon = df_pokemon.dropna()

    #cambiando nombre de variable

    df_pokemon = df_pokemon.rename(index=str, columns={"#": "id_pokemon"})

    # encoding

    df_pokemon['Legendary'] = np.where(df_pokemon['Legendary'] == True, 1, 0)

    # encoding name, type1 y type2

    valores_type1 = df_pokemon['Type 1'].values

    valores_type2 = df_pokemon['Type 2'].values

    valores_name = df_pokemon['Name'].values



    #print(df_pokemon.isna().sum())



    le1 = preprocessing.LabelEncoder()

    le2 = preprocessing.LabelEncoder()

    lename = preprocessing.LabelEncoder()

    encoding1 = le1.fit_transform(valores_type1)

    encoding2 = le2.fit_transform(valores_type2)

    encodingName = lename.fit_transform(valores_name)



    # asignando

    df_pokemon['Type 1'] = encoding1

    df_pokemon['Type 2'] = encoding2

    df_pokemon['Name'] = encodingName



    # rapido -> 1, Lento -> 0

    sum_speeds = np.sum(df_pokemon['Speed'].values)

    total_speeds = len(df_pokemon['Speed'])

    media_speeds = sum_speeds / total_speeds

    df_pokemon['Rapidez'] = np.where(df_pokemon['Speed'] > media_speeds, 1, 0)



    #-------battles.csv

    df_battles = pd.read_csv(path_dir + 'battles.csv')

    # quitamos el numero de batalla

    df_battles = df_battles[['First_pokemon','Second_pokemon', 'Winner']]

    print(df_battles.columns)



    #winrates

    #df_pokemon = utils.get_winrate(df_pokemon, df_battles)

    print(df_pokemon.head())



    #-------test.csv

    df_test = pd.read_csv(path_dir + 'test.csv')



    return df_pokemon, df_battles, df_test, le1, le2, lename



# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

def juntar_csvs():

    df_pokemon, df_battles, df_test, le1, le2, lename = get_data()



    #vectorizacion

    pokemon_values = df_pokemon.values #(800, cols)

    battles_values = df_battles.values #(50000, 3)

    

    ids_pokemon = pokemon_values[:,0]

    # obtenemos valores unicos y los indices inversos para luego reconstruir el array original

    ids_pok1, inv1 = np.unique(battles_values[:, 0], return_inverse=True)

    ids_pok2, inv2 = np.unique(battles_values[:, 1], return_inverse=True)

    resultados_batallas = battles_values[:, 2]



    # buscamos donde estan las caracteristicas de cada pokemon en las batallas

    indices1 = np.intersect1d(ids_pok1, ids_pokemon, return_indices=True)

    indices2 = np.intersect1d(ids_pok2, ids_pokemon, return_indices=True)



    # asignamos las caracteristicas

    vals_pok1 = pokemon_values[indices1[2], 1:]

    vals_pok2 = pokemon_values[indices2[2], 1:]



    # pokemons sin batallas

    sin_battles = pokemon_values[

        np.where(

            np.logical_not(

                np.isin(ids_pokemon, ids_pok1)))]

    # 16 en total

    print('Pokemons que no han peleado:', len(sin_battles))



    # y reconstruimos el array original

    lon_values = len(battles_values)

    # (50000, 11) cada uno

    pok1 = vals_pok1[inv1]

    pok2 = vals_pok2[inv2]

    #columnas = pok2.shape[1] * 2

    columnas = pok2.shape[1] + 3 #nombre2,tipo1_id2,tipo2_id2, el mas rapido

    print(pok2.shape)



    # aplicamos diff

    pok_final = np.ones((lon_values, columnas))

    pok_final[:, :3] = pok1[:, :3]#nombre1,tipo1_id1,tipo2_id1

    pok_final[:, 3:6] = pok2[:, :3]#nombre2,tipo2_id2,tipo2_id2

    pok_final[:, 6:] = pok1[:, 3:] - pok2[:, 3:]

    # el mas rapido

    #pok_final[:, -1] = np.where(pok1[:, -4] > pok2[:, -4], battles_values[:, 0], battles_values[:, 1])



    # aqui juntamos el resto para crear el dataset con el que entrenar

    #juntar_carac = np.concatenate((pok1, pok2), axis=1)

    juntar_carac = pok_final

    caracteristicas_y_resultados = np.ones((lon_values, columnas + 1)) # (50000, 15)

    caracteristicas_y_resultados[:,:-1] = juntar_carac

    caracteristicas_y_resultados[:,-1] = resultados_batallas



    # ids contrincante 1, ids contrincante 2 y el que golpea primero (añadido)

    valores = np.array((battles_values[:, 0], battles_values[:, 1], battles_values[:, 0])) #(3, 50000)

    valores = valores.T #(50000, 3)



    lista = np.concatenate((valores, caracteristicas_y_resultados), axis=1)

    lista = lista.astype(int)



    # guardo el fichero

    df_lista = pd.DataFrame(lista, columns=['First_pokemon', 'Second_pokemon', 'id_primer_ataq',

                                            'nombre1', 'tipo1_id1', 'tipo2_id1',

                                            'nombre2', 'tipo1_id2', 'tipo2_id2',

                                            'diff_HP','diff_Attack','diff_Defense','diff_Sp. Atk','diff_Sp. Def','diff_Speed',

                                            'diff_Generation', 'diff_Legendary',

                                            'diff_Rapidez',

                                            'Winner'])



    # efectividad de las habilidades

    # primero pasamos a las antiguas labels

    df_lista['tipo1_id1'] = le1.inverse_transform(df_lista['tipo1_id1'])

    df_lista['tipo2_id1'] = le2.inverse_transform(df_lista['tipo2_id1'])

    df_lista['tipo1_id2'] = le1.inverse_transform(df_lista['tipo1_id2'])

    df_lista['tipo2_id2'] = le2.inverse_transform(df_lista['tipo2_id2'])

    df_lista['nombre1'] = lename.inverse_transform(df_lista['nombre1'])

    df_lista['nombre2'] = lename.inverse_transform(df_lista['nombre2'])



    # y luego aplicamos los valores

    df_lista = calculate_effectiveness(df_lista)



    # reordenamos para colocar la columnas Winner al final

    winners = df_lista['Winner'].values

    df_lista = df_lista.drop(['Winner'], axis=1)

    df_lista['Winner'] = winners



    #y volvemos a aplicar los encodings

    df_lista['tipo1_id1'] = le1.fit_transform(df_lista['tipo1_id1'])

    df_lista['tipo2_id1'] = le2.fit_transform(df_lista['tipo2_id1'])

    df_lista['tipo1_id2'] = le1.fit_transform(df_lista['tipo1_id2'])

    df_lista['tipo2_id2'] = le2.fit_transform(df_lista['tipo2_id2'])

    df_lista['nombre1'] = lename.fit_transform(df_lista['nombre1'])

    df_lista['nombre2'] = lename.fit_transform(df_lista['nombre2'])



    # elimino carac que aportan menos --> no aporta

    #df_lista = df_lista.drop(['diff_Generation', 'diff_Legendary'], axis=1)



    df_lista.to_csv(fichero, index=False)

    #np.savetxt(fichero, lista)



    return lista
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

def preparar_test():

    df_pokemon, df_battles, df_test, le1, le2, lename = get_data()



    # vectorizacion

    pokemon_values = df_pokemon.values  # (800, 12)

    tests_values = df_test.values  # (10000, 3)



    ids_pokemon = pokemon_values[:, 0]

    # obtenemos valores unicos y los indices inversos para luego reconstruir el array original

    ids_pok1, inv1 = np.unique(tests_values[:, 1], return_inverse=True)

    ids_pok2, inv2 = np.unique(tests_values[:, 2], return_inverse=True)



    # buscamos donde estan las caracteristicas de cada pokemon en las batallas

    indices1 = np.intersect1d(ids_pok1, ids_pokemon, return_indices=True)

    indices2 = np.intersect1d(ids_pok2, ids_pokemon, return_indices=True)



    # asignamos las caracteristicas

    vals_pok1 = pokemon_values[indices1[2], 1:]

    vals_pok2 = pokemon_values[indices2[2], 1:]



    # pokemons sin batallas

    sin_battles = pokemon_values[

        np.where(

            np.logical_not(

                np.isin(ids_pokemon, ids_pok1)))]

    # 16 en total

    print('Pokemons que no han peleado en test:', len(sin_battles))



    # y reconstruimos el array original

    lon_values = len(tests_values)

    # (10000, 11) cada uno

    pok1 = vals_pok1[inv1]

    pok2 = vals_pok2[inv2]

    columnas = pok2.shape[1] + 3  # nombre2,tipo1_id2,tipo2_id2, Mas_Winrate



    # aplicamos diff

    pok_final = np.ones((lon_values, columnas))

    pok_final[:, :3] = pok1[:, :3]

    pok_final[:, 3:6] = pok2[:, :3]

    pok_final[:, 6:] = pok1[:, 3:] - pok2[:, 3:]

    # winrate

    #pok_final[:, -2] = np.where(pok1[:, -1] > pok2[:, -1], tests_values[:, 0], tests_values[:, 1])

    # el mas rapido

    #pok_final[:, -1] = np.where(pok1[:, -2] > pok2[:, -2], tests_values[:, 0], tests_values[:, 1])



    # aqui juntamos el resto para crear el dataset con el que entrenar

    # juntar_carac = np.concatenate((pok1, pok2), axis=1)

    juntar_carac = pok_final



    # ids contrincante 1, ids contrincante 2 y el que golpea primero (añadido)

    valores = np.array((tests_values[:, 1], tests_values[:, 2], tests_values[:, 1]))  # (3, 10000)

    valores = valores.T  # (10000, 3)



    lista = np.concatenate((valores, juntar_carac), axis=1)

    lista = lista.astype(int)

    print(lista.shape)

    # guardo el fichero

    df_lista = pd.DataFrame(lista, columns=['First_pokemon', 'Second_pokemon', 'id_primer_ataq',

                                            'nombre1', 'tipo1_id1', 'tipo2_id1',

                                            'nombre2', 'tipo1_id2', 'tipo2_id2',

                                            'HP','Attack','Defense','Sp. Atk','Sp. Def','Speed',

                                            'Generation', 'Legendary',

                                            'Rapidez'

                                            ])



    # efectividad de las habilidades

    # primero pasamos a las antiguas labels

    df_lista['tipo1_id1'] = le1.inverse_transform(df_lista['tipo1_id1'])

    df_lista['tipo2_id1'] = le2.inverse_transform(df_lista['tipo2_id1'])

    df_lista['tipo1_id2'] = le1.inverse_transform(df_lista['tipo1_id2'])

    df_lista['tipo2_id2'] = le2.inverse_transform(df_lista['tipo2_id2'])

    df_lista['nombre1'] = lename.inverse_transform(df_lista['nombre1'])

    df_lista['nombre2'] = lename.inverse_transform(df_lista['nombre2'])



    # y luego aplicamos los valores

    df_lista = calculate_effectiveness(df_lista)



    # y volvemos a aplicar los encodings

    df_lista['tipo1_id1'] = le1.fit_transform(df_lista['tipo1_id1'])

    df_lista['tipo2_id1'] = le2.fit_transform(df_lista['tipo2_id1'])

    df_lista['tipo1_id2'] = le1.fit_transform(df_lista['tipo1_id2'])

    df_lista['tipo2_id2'] = le2.fit_transform(df_lista['tipo2_id2'])

    df_lista['nombre1'] = lename.fit_transform(df_lista['nombre1'])

    df_lista['nombre2'] = lename.fit_transform(df_lista['nombre2'])



    # elimino carac que aportan menos --> no aporta

    #df_lista = df_lista.drop(['Generation', 'Legendary'], axis=1)



    df_lista.to_csv(tests, index=False)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------MODELO EMPLEADOS--------------------------------------------------

def GradientBoosting(train_x, train_y, test_x, test_y):



    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, subsample=0.75,max_depth=9, verbose = 1)

    clf.fit(train_x, train_y)



    y_pred=clf.predict(test_x)

    print(clf.feature_importances_)

    print("Accuracy random forest:",metrics.accuracy_score(test_y, y_pred))



    return clf
# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

# ----------------------------------------------------RESULTADOS--------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

def agrupados():



    lista = pd.read_csv(fichero).values



    print(lista.shape)



    X = lista[:, :-1]

    y = lista[:, -1]



    train_x,train_y = X[:n_test], y[:n_test]

    test_x, test_y = X[n_test:], y[n_test:]



    rf = GradientBoosting(train_x, train_y, test_x, test_y)

    #mlp = MLP(train_x, train_y, test_x, test_y)

    #svm = SVM(train_x, train_y, test_x, test_y)



    return rf

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------guardar datos finales-------------------------------------------------------

def resultado_final():



    lista = pd.read_csv(tests).values

    print(lista.shape)



    clf = agrupados()

    y_pred = clf.predict(lista)

    

    print(y_pred)

    y_pred = y_pred.astype(int)



    df_test = pd.read_csv(path_dir + 'test.csv')

    df_test['Winner'] = y_pred

    

    # lo que subo

    df_sample = df_test[['battle_number', 'Winner']]

    df_sample.to_csv(sample, index=False)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# solo_battles()

juntar_csvs()

preparar_test()

#agrupados()

resultado_final()