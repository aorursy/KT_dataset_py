#!pip install numpy --upgrade 

#!pip install catboost


import numpy as np

#np.set_printoptions(suppress=True)

import pandas as pd

from sklearn import metrics

from sklearn import preprocessing

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score

#PCA

from sklearn.decomposition import PCA,KernelPCA

# para modelar

import lightgbm as lgbm



#import io

#from google.colab import files

import os





print(os.listdir("../input"))

n_test= 49999

fichero = 'datos.csv'

tests = 'entrega_para_predecir.csv'

resultados_finales = 'resultados_finales_test.csv'

sample = 'resultados_finales_sampleSubmission.csv'

path_dir = 'pokemon-challenge-mlh/'

pokemon_csv = '../input/pokemon.csv'

battles_csv = '../input/battles.csv'

test_csv = '../input/test.csv'

# ----------------------------------------------------DATASET-----------------------------------------------------------

def get_data():



    #-------pokemon.csv

    df_pokemon = pd.read_csv(pokemon_csv)

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

    

    #sum de las stats

    #'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed'

    df_pokemon['total_stats'] = df_pokemon['HP'] + df_pokemon['Attack'] + df_pokemon['Defense'] + df_pokemon['Sp. Atk'] + df_pokemon['Sp. Def'] + df_pokemon['Speed']



    #-------battles.csv

    df_battles = pd.read_csv(battles_csv)

    # quitamos el numero de batalla

    df_battles = df_battles[['First_pokemon','Second_pokemon', 'Winner']]



    #winrates

    #df_pokemon = get_winrate(df_pokemon, df_battles)

    #print(df_pokemon.head())



    #-------test.csv

    df_test = pd.read_csv(test_csv)



    return df_pokemon, df_battles, df_test, le1, le2, lename
# autor: https://www.kaggle.com/vforvince1/visualizing-data-and-predicting-pokemon-fights

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

    p1_max = []

    p2_type1_list = []

    p2_type2_list = []

    p2_max = []



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

        p1_max.append(np.maximum(nested_type[0][0], nested_type[0][1]))

        p2_max.append(np.maximum(nested_type[1][0], nested_type[1][1]))



    data = data.assign(P1_type1=p1_type1_list, P1_type2=p1_type2_list,

                       P2_type1=p2_type1_list, P2_type2=p2_type2_list)

    #data = data.drop(['First_pokemon', 'Second_pokemon'], axis=1)



    return data

      

def diff_combates(df_lista):

  

  # DIFF ataques vs defensas cuerpo a cuerpo

  lista_ata_pok1 = df_lista[['Attack_id1']].values

  lista_ata_pok2 = df_lista[['Attack_id2']].values

  lista_def_pok1 = df_lista[['Defense_id1']].values

  lista_def_pok2 = df_lista[['Defense_id2']].values

  

  lista_diff_ataDef_pok1 = lista_ata_pok1 - lista_def_pok1

  lista_diff_ataDef_pok2 = lista_ata_pok2 - lista_def_pok1

  

  # asignamos

  df_lista['diff_ata_def_pok1'] = lista_diff_ataDef_pok1

  df_lista['diff_ata_def_pok2'] = lista_diff_ataDef_pok2

  

  # DIFF ataques especiales

  efec_pok1 = df_lista[['P1_type1', 'P1_type2']].values

  efec_pok2 = df_lista[['P2_type1', 'P2_type2']].values

  sumatorio_efectividad_pok1 = np.sum(efec_pok1, axis=1)

  sumatorio_efectividad_pok2 = np.sum(efec_pok2, axis=1)

  lista_ataESP_pok1 = df_lista[['Sp. Atk_id1']].values

  lista_ataESP_pok2 = df_lista[['Sp. Atk_id2']].values

  lista_defESP_pok1 = df_lista[['Sp. Def_id1']].values

  lista_defESP_pok2 = df_lista[['Sp. Def_id2']].values

  # los ataques los multiplico por la efectividad

  # multiplicaciones

  lista_ataESP_pok1_final = np.zeros((len(df_lista)))

  lista_ataESP_pok2_final = np.zeros((len(df_lista)))

  lista_diff_ataDefESP_pok1 = np.zeros((len(df_lista)))

  lista_diff_ataDefESP_pok2 = np.zeros((len(df_lista)))

  

  # he tenido que dividir las matrices porque colab se peta al multiplicarlas (fallo de RAM)

  for i in range(0, len(df_lista)):

    lista_ataESP_pok1_final[i] = lista_ataESP_pok1[i] * sumatorio_efectividad_pok1[i]

    lista_ataESP_pok2_final[i] = lista_ataESP_pok2[i] * sumatorio_efectividad_pok2[i]

    

    lista_diff_ataDefESP_pok1[i] = lista_ataESP_pok1_final[i] - lista_defESP_pok2[i]

    lista_diff_ataDefESP_pok2[i] = lista_ataESP_pok2_final[i] - lista_defESP_pok1[i]

    

    i+=1

     

  # restas

  #lista_diff_ataDefESP_pok1 = lista_ataESP_pok1_final - lista_defESP_pok2

  #lista_diff_ataDefESP_pok2 = lista_ataESP_pok2_final - lista_defESP_pok1

  

  # asignamos

  df_lista['diff_ata_def_ESP_pok1'] = lista_diff_ataDef_pok1

  df_lista['diff_ata_def_ESP_pok2'] = lista_diff_ataDef_pok2

  

  return df_lista
# ----------------------------------------------------------------------------------------------------------------------

def juntar_csvs(es_conjunto_de_test):

    df_pokemon, df_battles, df_test, le1, le2, lename = get_data()



    #vectorizacion

    pokemon_values = df_pokemon.values #(800, cols)

    if es_conjunto_de_test == False:

      battles_values = df_battles.values #(50000, 3) --> id_pok1, id_pok2, winner

      indice_pok1 = 0

      indice_pok2 = 1

    else:

      battles_values = df_test.values  # (10000, 3) --> battle_id, id_pok1, id_pok2

      indice_pok1 = 1

      indice_pok2 = 2

    

    ids_pokemon = pokemon_values[:,0]

    # obtenemos valores unicos y los indices inversos para luego reconstruir el array original

    ids_pok1, inv1 = np.unique(battles_values[:, indice_pok1], return_inverse=True)

    ids_pok2, inv2 = np.unique(battles_values[:, indice_pok2], return_inverse=True)

    

    if es_conjunto_de_test == False:

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

    # ([1|5]0000, x) cada uno

    pok1 = vals_pok1[inv1]

    pok2 = vals_pok2[inv2]

    #columnas = pok2.shape[1] * 2

    columnas = pok2.shape[1] + 13 + 6 + 6 #11 por atrs pok2 + 6 ratios + 6 diffs



    pok_final = np.ones((lon_values, columnas))

    print(pok_final.shape)

    pok_final[:, :11] = pok1[:, :11]#nombre1,tipo1_id1,tipo2_id1, 'HP_id1','Attack_id1','Defense_id1','Sp. Atk_id1','Sp. Def_id1','Speed_id1', 'Generation_id1', 'Legendary_id1'

    pok_final[:, 11:22] = pok2[:, :11]#nombre2,tipo2_id2,tipo2_id2,'HP_id2','Attack_id2','Defense_id2','Sp. Atk_id2','Sp. Def_id2','Speed_id2', 'Generation_id2', 'Legendary_id2'

    # aplicamos diff

    pok_final[:, 22:30] = pok1[:, 3:11] - pok2[:, 3:11]

    # el mas rapido

    #pok_final[:, -1] = np.where(pok1[:, -1] > pok2[:, -1], battles_values[:, 0], battles_values[:, 1])

    # ratios. Excluyo Generacion y Legendario

    pok1_ratios = pok1[:, 3:9]

    pok2_ratios = pok2[:, 3:9]

    pok_final[:, 30:36] = pok1_ratios/pok2_ratios

    

    # aqui juntamos el resto para crear el dataset con el que entrenar/testear

    # ids contrincante 1, ids contrincante 2 y el que golpea primero (añadido)

    valores = np.array((battles_values[:, indice_pok1], battles_values[:, indice_pok2], battles_values[:, indice_pok1])) #(3, [1|5]0000)

    valores = valores.T #([1|5]0000, 3)

    

    if es_conjunto_de_test == False:

      caracteristicas_y_resultados = np.ones((lon_values, columnas + 1)) # ([1|5]0000, x)

      caracteristicas_y_resultados[:,:-1] = pok_final

      caracteristicas_y_resultados[:,-1] = resultados_batallas

      

      lista = np.concatenate((valores, caracteristicas_y_resultados), axis=1)

      

    else: 

      # en el test no hay winner

      lista = np.concatenate((valores, pok_final), axis=1)

      

    lista = lista.astype(int)



    columnas = ['First_pokemon', 'Second_pokemon', 'id_primer_ataq',

                                            'nombre1', 'tipo1_id1', 'tipo2_id1',

                'HP_id1','Attack_id1','Defense_id1','Sp. Atk_id1','Sp. Def_id1','Speed_id1', 'Generation_id1', 'Legendary_id1',

                                            'nombre2', 'tipo1_id2', 'tipo2_id2',

                'HP_id2','Attack_id2','Defense_id2','Sp. Atk_id2','Sp. Def_id2','Speed_id2', 'Generation_id2', 'Legendary_id2',

                                            'diff_HP','diff_Attack','diff_Defense','diff_Sp. Atk','diff_Sp. Def','diff_Speed',

                                            'diff_Generation', 'diff_Legendary',

                                            'diff_Rapidez', 'diff_stats',

                                            'ratio_HP','ratio_Attack','ratio_Defense','ratio_Sp. Atk','ratio_Sp. Def','ratio_Speed']

    # caso conjunto de entrenamiento

    if es_conjunto_de_test == False:

      columnas.append('Winner')

    

    # guardo en DF

    df_lista = pd.DataFrame(lista, columns=columnas)



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

    

    # añadidos

    # ++ ratios y diffs agrupados

    df_lista['diff_HPDefense_SpDef'] = df_lista['diff_HP'] + df_lista['diff_Defense'] + df_lista['diff_Sp. Def']

    df_lista['ratio_HPDefense_SpDef'] = df_lista['ratio_HP'] + df_lista['ratio_Defense'] + df_lista['ratio_Sp. Def']

    #  ++ diff efectividad

    efec_pok1 = df_lista['P1_type1'].values + df_lista['P1_type2'].values

    efec_pok2 = df_lista['P2_type1'].values + df_lista['P2_type2'].values

    df_lista['diff_efectividad'] = np.subtract(efec_pok1,efec_pok2)

    # ata -ddef y ataESP- defESP

    #df_lista = diff_combates(df_lista)

    

    # reordenamos para colocar la columnas Winner al final

    if es_conjunto_de_test == False:

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



    # elimino carac que aportan menos

    df_lista = df_lista.drop(['Legendary_id1', 'Legendary_id2', 'Generation_id1', 'Generation_id2'], axis=1)

    df_lista = df_lista.drop(['nombre1', 'nombre2'], axis = 1)

    #df_lista = df_lista.drop(['diff_ata_def_pok1', 'diff_ata_def_pok2', 'diff_ata_def_ESP_pok1', 'diff_ata_def_ESP_pok2'], axis=1)

    #df_lista = df_lista.drop(['diff_HP','diff_Defense','diff_Sp. Def',

    #                         'ratio_HP','ratio_Defense', 'ratio_Sp. Def'], axis=1)

    

    print(df_lista.shape)



    # guardamos

    if es_conjunto_de_test == False:

      df_lista.to_csv(fichero, index=False)

    else:

      df_lista.to_csv(tests, index=False)



    return lista
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------MODELO EMPLEADOS--------------------------------------------------

#0.9791 --> 0.05lr

#n_estimators=800, learning_rate=0.05, subsample=0.75,max_depth=10

def GradientBoostingCl(train_x, train_y, test_x, test_y):



    clf = GradientBoostingClassifier(n_estimators=800, learning_rate=0.05, subsample=0.75,max_depth=9)

    clf.fit(train_x, train_y)



    y_pred=clf.predict(test_x)

    print(clf.feature_importances_)

    print("Accuracy random forest1:",metrics.accuracy_score(test_y, y_pred))

    



    return clf
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------- MODELO  LIGHTGBM -----------------------------------------------------

def lightgbm_model(train_x, train_y, test_x, test_y, columnas):

  



    # making lgbm datasets for train and valid

    d_train = lgbm.Dataset(train_x, label= train_y)

    d_valid = lgbm.Dataset(test_x, label= test_y)

    

    # training

    bst = lgbm.train(params, d_train, valid_sets= [d_valid], verbose_eval=100, feature_name= (columnas.tolist())[:-1])

    

    #Prediction --> comentar estas lineas cuando se vaya a subir a kaggle co n49999

    y_pred=bst.predict(test_x)

    

    #convert into binary values 

    for i in range(0,len(test_y)):

      if y_pred[i]>=.5:       # setting threshold to .5

        y_pred[i]=1

      else:  

        y_pred[i]=0

        

    #Accuracy

    accuracy=accuracy_score(y_pred,test_y)

    print(accuracy)

    

    lgbm.plot_importance(bst, max_num_features = 15)

    #lgbm.plot_tree(bst, figsize= (100,100))



    return bst

  

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------eleccion de mejores parametros-----------------------------------------------

def feature_search(X, y, columnas):

  

  # parametros a probar

  gridParams = {

      'learning_rate': [0.05, 0.07, 0.08, 0.1],

      'num_iterations': [900, 1000, 1100],

      'boosting_type' : ['goss'],

      'objective' : ['binary'],

      'subsample' : [0.7,0.75],

      'lambda_l1' : [0.005, 0.01],

      'lambda_l2' : [0.005, 0.01],

      'max_depth': [9, 10, 11, 12, -1] 

    }

  

  # entrenamos con los por defecto del modelo

  model = lgbm.LGBMClassifier(params)

  

  # definimos el grid

  grid = RandomizedSearchCV(estimator = model, 

    param_distributions = gridParams, 

    scoring='roc_auc',

    n_jobs=-1,

    iid=False, 

    verbose=2,

    cv=3)

  

  grid.fit(X, y)



  # Print the best parameters found

  print(grid.best_params_)

  print(grid.best_score_)

  

  # TODO reemplazar mejores parametros y entrenar modelo

  

  return grid

  

# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

# ----------------------------------------------------RESULTADOS--------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

def agrupados():



    df = pd.read_csv(fichero)

    

    lista = df.values

    

    print(df.columns)



    X = lista[:, :-1]

    y = lista[:, -1]



    train_x,train_y = X[:n_test], y[:n_test]

    test_x, test_y = X[n_test:], y[n_test:]



    # lightbeam

    clf = lightgbm_model(train_x, train_y, test_x, test_y, df.columns)

    #clf = feature_search(X, y, df.columns)





    return clf
# ------------------------------------------guardar datos finales-------------------------------------------------------

def resultado_final():



    df = pd.read_csv(tests)

    lista = df.values

    print(lista.shape)

    

    clf = agrupados()

    

    y_pred=clf.predict(lista)

    for i in range(0,len(lista)):

      if y_pred[i]>=.5:       # setting threshold to .5

        y_pred[i]=1

      else:  

        y_pred[i]=0

    y_pred = y_pred.astype(int)



    df_test = pd.read_csv(test_csv)

    df_test['Winner'] = y_pred

    df_test.to_csv(resultados_finales, index=False)



    df_sample = df_test[['battle_number', 'Winner']]

    df_sample.to_csv(sample, index=False)


# LIGHTGBM

#{'subsample': 0.7, 'objective': 'binary', 'num_iterations': 900, 'max_depth': 11, 'learning_rate': 0.07, 'lambda_l2': 0.01, 'lambda_l1': 0.005, 'boosting_type': 'goss'}

# parameters for LightGBMClassifier

params = {'num_iterations': 900,

          'objective' :'binary',

          'learning_rate' : 0.08,

          'boosting_type' : 'goss',

          'max_depth': 12,

          'metric': 'binary_logloss',

          #'xgboost_dart_mode': True,

          'lambda_l1': 0.01,

          'lambda_l2': 0.01,

          'subsample': 0.75 }



# ----------------------------------------------------------------------------------------------------------------------

# funciones que preparan los datos

juntar_csvs(False)

juntar_csvs(True)

#funciones que entrenan el modelo

#agrupados()

resultado_final()