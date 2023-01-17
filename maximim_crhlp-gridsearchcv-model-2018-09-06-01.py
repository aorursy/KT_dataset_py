import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import time

import os
import gc
import random
from contextlib import contextmanager




import warnings
warnings.filterwarnings("ignore")

_N_JOBS = 1

start = time.time()

@contextmanager
def timer(title):
    print('{} - Starting'.format(title))    
    tinit = time.time()
    yield
    print("{} - done in {:.0f}s\n".format(title, time.time() - tinit))    

def get_data(Debug = False):
    train_file = '../input/train.csv'
    test_file = '../input/test.csv'
#    train_file = 'C:/Users/maxim/Google Drive/Kaggle/CRHLP/train.csv'
#    test_file = 'C:/Users/maxim/Google Drive/Kaggle/CRHLP/test.csv'
    
    train_data =pd.read_csv(train_file)
    test_data =pd.read_csv(test_file)
    
    return train_data, test_data

def handling_missing_data(train_data, test_data):
    # USAR: si no hay tablets en la casa, entonces la cantidad total de tablet es 0! ó si v18q = 0 -> v18q1 = 0!
    train_data.loc[(train_data['v18q1'].isnull()), 'v18q1'] = train_data.loc[(train_data['v18q1'].isnull()), 'v18q']
    test_data.loc[(test_data['v18q1'].isnull()), 'v18q1'] = test_data.loc[(test_data['v18q1'].isnull()), 'v18q']
    
    # outlier in test set which rez_esc is 99.0
    test_data.loc[test_data['rez_esc'] > 5 , 'rez_esc'] = 5
    
    #Fill na
    def replace_v18q1(x):
        if x['v18q'] == 0:
            return x['v18q']
        else:
            return x['v18q1']
        
    train_data['v18q1'] = train_data.apply(lambda x : replace_v18q1(x),axis=1)
    test_data['v18q1'] = test_data.apply(lambda x : replace_v18q1(x),axis=1)
    # Esta verificado la consistencia entre 'v18q' y 'v18q1'
    
    #Fill na in v2a1
    def replace_v2a1(x):
        if x['tipovivi1'] == 1 or x['tipovivi4'] == 1 or x['tipovivi5'] == 1:
            return 0
        else:
            return x['v2a1']
        
    train_data['v2a1'] = train_data.apply(lambda x: replace_v2a1(x), axis=1)
    test_data['v2a1'] = test_data.apply(lambda x: replace_v2a1(x), axis=1)
        
    return train_data, test_data

def feature_engineering(train_data, test_data):
    cols = ['edjefe', 'edjefa']
    try:
        train_data[cols] = train_data[cols].replace({'no': 0, 'yes':1}).astype(float)
        test_data[cols] = test_data[cols].replace({'no': 0, 'yes':1}).astype(float)
    except:
        pass

    # It turns out orignial data lost one feature both for roof and electricity, so we manually add new feature
    train_data['roof_waste_material'] = np.nan
    test_data['roof_waste_material'] = np.nan
    train_data['electricity_other'] = np.nan
    test_data['electricity_other'] = np.nan

    def fill_roof_exception(x):
        if (x['techozinc'] == 0) and (x['techoentrepiso'] == 0) and (x['techocane'] == 0) and (x['techootro'] == 0):
            return 1
        else:
            return 0
    
    def fill_no_electricity(x):
        if (x['public'] == 0) and (x['planpri'] == 0) and (x['noelec'] == 0) and (x['coopele'] == 0):
            return 1
        else:
            return 0

    train_data['roof_waste_material'] = train_data.apply(lambda x : fill_roof_exception(x),axis=1)
    test_data['roof_waste_material'] = test_data.apply(lambda x : fill_roof_exception(x),axis=1)
    train_data['electricity_other'] = train_data.apply(lambda x : fill_no_electricity(x),axis=1)
    test_data['electricity_other'] = test_data.apply(lambda x : fill_no_electricity(x),axis=1)


    # Other features in train_data
    train_data['adult'] = train_data['hhsize'] - train_data['hogar_mayor'] - train_data['hogar_nin']
    train_data['no_elder'] = train_data['hhsize'] - train_data['hogar_mayor']
    train_data['dependency_count'] = train_data['hogar_nin'] + train_data['hogar_mayor']
    train_data['dependency'] = train_data['dependency_count'] / train_data['adult']
    train_data['dependency'] = train_data['dependency'].replace({np.inf: 0}) ## No es correcto -> no tienen dependencia o un número muy grande.

    train_data['child_percent'] = train_data['hogar_nin']/train_data['hhsize']
    train_data['elder_percent'] = train_data['hogar_mayor']/train_data['hhsize']
    train_data['adult_percent'] = train_data['hogar_adul']/train_data['hhsize']


    test_data['adult'] = test_data['hhsize'] - test_data['hogar_mayor'] - test_data['hogar_nin']
    test_data['no_elder'] = test_data['hhsize'] - test_data['hogar_mayor']
    test_data['dependency_count'] = test_data['hogar_nin'] + test_data['hogar_mayor']
    test_data['dependency'] = test_data['dependency_count'] / test_data['adult']

    test_data['child_percent'] = test_data['hogar_nin']/test_data['hhsize']
    test_data['elder_percent'] = test_data['hogar_mayor']/test_data['hhsize']
    test_data['adult_percent'] = test_data['hogar_adul']/test_data['hhsize']


    train_data['rent_per_adult'] = train_data['v2a1']/train_data['hogar_adul']
    train_data['rent_per_active_adult'] = train_data['v2a1']/train_data['adult']
    train_data['rent_per_person'] = train_data['v2a1']/train_data['hhsize']

    train_data['overcrowding_room_and_bedroom'] = (train_data['hacdor'] + train_data['hacapo'])/2

    train_data['no_appliances'] = train_data['refrig'] + train_data['computer'] + train_data['television']

    train_data['r4h1_percent_in_male'] = train_data['r4h1'] / train_data['r4h3']
    train_data['r4m1_percent_in_female'] = train_data['r4m1'] / train_data['r4m3']
    train_data['r4h1_percent_in_total'] = train_data['r4h1'] / train_data['hhsize']
    train_data['r4m1_percent_in_total'] = train_data['r4m1'] / train_data['hhsize']
    train_data['r4t1_percent_in_total'] = train_data['r4t1'] / train_data['hhsize']

    train_data['rent_per_room'] = train_data['v2a1']/train_data['rooms']
    train_data['bedroom_per_room'] = train_data['bedrooms']/train_data['rooms']
    train_data['elder_per_room'] = train_data['hogar_mayor']/train_data['rooms']
    train_data['adults_per_room'] = train_data['adult']/train_data['rooms']
    train_data['child_per_room'] = train_data['hogar_nin']/train_data['rooms']
    train_data['male_per_room'] = train_data['r4h3']/train_data['rooms']
    train_data['female_per_room'] = train_data['r4m3']/train_data['rooms']
    train_data['room_per_person_household'] = train_data['hhsize']/train_data['rooms']

    train_data['rent_per_bedroom'] = train_data['v2a1']/train_data['bedrooms']
    train_data['edler_per_bedroom'] = train_data['hogar_mayor']/train_data['bedrooms']
    train_data['adults_per_bedroom'] = train_data['adult']/train_data['bedrooms']
    train_data['child_per_bedroom'] = train_data['hogar_nin']/train_data['bedrooms']
    train_data['male_per_bedroom'] = train_data['r4h3']/train_data['bedrooms']
    train_data['female_per_bedroom'] = train_data['r4m3']/train_data['bedrooms']
    train_data['bedrooms_per_person_household'] = train_data['hhsize']/train_data['bedrooms']

    train_data['tablet_per_person_household'] = train_data['v18q1']/train_data['hhsize']
    train_data['phone_per_person_household'] = train_data['qmobilephone']/train_data['hhsize']

    train_data['age_12_19'] = train_data['hogar_nin'] - train_data['r4t1']

    train_data['escolari_age'] = train_data['escolari']/train_data['age']

    train_data['rez_esc_escolari'] = train_data['rez_esc']/train_data['escolari']
    train_data['rez_esc_r4t1'] = train_data['rez_esc']/train_data['r4t1']
    train_data['rez_esc_r4t2'] = train_data['rez_esc']/train_data['r4t2']
    train_data['rez_esc_r4t3'] = train_data['rez_esc']/train_data['r4t3']
    train_data['rez_esc_age'] = train_data['rez_esc']/train_data['age']

    # Other features in test_data
    test_data['adult'] = test_data['hhsize'] - test_data['hogar_mayor']
    test_data['dependency_count'] = test_data['hogar_nin'] + test_data['hogar_mayor']
    test_data['dependency'] = test_data['dependency_count'] / test_data['adult']
    test_data['dependency'] = test_data['dependency'].replace({np.inf: 0}) ## No es correcto -> no tienen dependencia o un número muy grande.

    test_data['child_percent'] = test_data['hogar_nin']/test_data['hhsize']
    test_data['elder_percent'] = test_data['hogar_mayor']/test_data['hhsize']
    test_data['adult_percent'] = test_data['hogar_adul']/test_data['hhsize']

    test_data['adult'] = test_data['hhsize'] - test_data['hogar_mayor'] - test_data['hogar_nin']
    test_data['dependency_count'] = test_data['hogar_nin'] + test_data['hogar_mayor']
    test_data['dependency'] = test_data['dependency_count'] / test_data['adult']
    test_data['child_percent'] = test_data['hogar_nin']/test_data['hhsize']
    test_data['elder_percent'] = test_data['hogar_mayor']/test_data['hhsize']
    test_data['adult_percent'] = test_data['hogar_adul']/test_data['hhsize']

    test_data['rent_per_adult'] = test_data['v2a1']/test_data['adult']
    test_data['rent_per_active_adult'] = test_data['v2a1']/test_data['hogar_adul']
    test_data['rent_per_person'] = test_data['v2a1']/test_data['hhsize']

    test_data['overcrowding_room_and_bedroom'] = (test_data['hacdor'] + test_data['hacapo'])/2

    test_data['no_appliances'] = test_data['refrig'] + test_data['computer'] + test_data['television']

    test_data['r4h1_percent_in_male'] = test_data['r4h1'] / test_data['r4h3']
    test_data['r4m1_percent_in_female'] = test_data['r4m1'] / test_data['r4m3']
    test_data['r4h1_percent_in_total'] = test_data['r4h1'] / test_data['hhsize']
    test_data['r4m1_percent_in_total'] = test_data['r4m1'] / test_data['hhsize']
    test_data['r4t1_percent_in_total'] = test_data['r4t1'] / test_data['hhsize']

    test_data['rent_per_room'] = test_data['v2a1']/test_data['rooms']
    test_data['bedroom_per_room'] = test_data['bedrooms']/test_data['rooms']
    test_data['elder_per_room'] = test_data['hogar_mayor']/test_data['rooms']
    test_data['adults_per_room'] = test_data['adult']/test_data['rooms']
    test_data['child_per_room'] = test_data['hogar_nin']/test_data['rooms']
    test_data['male_per_room'] = test_data['r4h3']/test_data['rooms']
    test_data['female_per_room'] = test_data['r4m3']/test_data['rooms']
    test_data['room_per_person_household'] = test_data['hhsize']/test_data['rooms']

    test_data['rent_per_bedroom'] = test_data['v2a1']/test_data['bedrooms']
    test_data['edler_per_bedroom'] = test_data['hogar_mayor']/test_data['bedrooms']
    test_data['adults_per_bedroom'] = test_data['adult']/test_data['bedrooms']
    test_data['child_per_bedroom'] = test_data['hogar_nin']/test_data['bedrooms']
    test_data['male_per_bedroom'] = test_data['r4h3']/test_data['bedrooms']
    test_data['female_per_bedroom'] = test_data['r4m3']/test_data['bedrooms']
    test_data['bedrooms_per_person_household'] = test_data['hhsize']/test_data['bedrooms']

    test_data['tablet_per_person_household'] = test_data['v18q1']/test_data['hhsize']
    test_data['phone_per_person_household'] = test_data['qmobilephone']/test_data['hhsize']

    test_data['age_12_19'] = test_data['hogar_nin'] - test_data['r4t1']


    test_data['escolari_age'] = test_data['escolari']/test_data['age']


    test_data['rez_esc_escolari'] = test_data['rez_esc']/test_data['escolari']
    test_data['rez_esc_r4t1'] = test_data['rez_esc']/test_data['r4t1']
    test_data['rez_esc_r4t2'] = test_data['rez_esc']/test_data['r4t2']
    test_data['rez_esc_r4t3'] = test_data['rez_esc']/test_data['r4t3']
    test_data['rez_esc_age'] = test_data['rez_esc']/test_data['age']

    return train_data, test_data

def more_feature_engineering(train_data, test_data):
    def fill_tipovivin(x, column_prefix, asc, n): # debe traer a n en el argumento.
        value = 0
        for i in range(1, n+1):
            if asc == True:
                value += x[column_prefix+str(i)] * (2 ** (i-1))
            else:
                value += x[column_prefix+str(i)] * (2 ** (n-i)) ### Falso!


        return value

    train_data['tipovivix'] = train_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'tipovivi', asc=False, n=5),axis=1)
    test_data['tipovivix'] = test_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'tipovivi', asc=False, n=5),axis=1)

    train_data['evivx'] = train_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'eviv', asc=True, n=3),axis=1)
    train_data['etechox'] = train_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'etecho', asc=True, n=3),axis=1)
    train_data['eparedx'] = train_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'epared', asc=True, n=3),axis=1)
    train_data['elimbasux'] = train_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'elimbasu', asc=False, n=6),axis=1)
    train_data['elimbasux'] = train_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'elimbasu', asc=False, n=6),axis=1)

    test_data['evivx'] = test_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'eviv', asc=True, n=3),axis=1)
    test_data['etechox'] = test_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'etecho', asc=True, n=3),axis=1)
    test_data['eparedx'] = test_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'epared', asc=True, n=3),axis=1)
    test_data['elimbasux'] = test_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'elimbasu', asc=False, n=6),axis=1)
    test_data['elimbasux'] = test_data.apply(lambda x : fill_tipovivin(x, column_prefix= 'elimbasu', asc=False, n=6),axis=1)

    return train_data, test_data

def even_more_feature_engineering(train_data, test_data):
    
    train_data['edjefx'] = train_data['edjefa'] + train_data['edjefe']
    test_data['edjefx'] = test_data['edjefa'] + test_data['edjefe']

    # Aproximación a los años que dedico al estudio ...
    def fill_instlevelx(x):
        value = 0
        for i in range(1, 9):
            value += 5 * x['instlevel'+str(i)] * int(i/2) / (2 ** (1-divmod(i,2)[1]))


        return value

    train_data['instlevelx'] = train_data.apply(lambda x : fill_instlevelx(x),axis=1)
    test_data['instlevelx'] = test_data.apply(lambda x : fill_instlevelx(x),axis=1)

    train_data['instlevelx_to_age'] = train_data['instlevelx'] / train_data['age']
    test_data['instlevelx_to_age'] = test_data['instlevelx'] / test_data['age']

    # Años que siguieron a la educación. Años laborales que le siguieron a la persona.
    def fill_laboralperiod(x):
        value = 0
        age = x['age']
        instlevelx = x['instlevelx']

        value = age - instlevelx - 5
        if instlevelx + 5 > 18:
            value += 0
        elif age > 18:
            value = age - 18 + 0.8 * (18-instlevelx - 5)
        else:
            value = 0.8 * value

        if value < 0:
            value = 0

        return value

    train_data['laboralperiod'] = train_data.apply(lambda x : fill_laboralperiod(x),axis=1)
    test_data['laboralperiod'] = test_data.apply(lambda x : fill_laboralperiod(x),axis=1)

    train_data['remainlaboralperiod'] = 65 - train_data['laboralperiod']
    test_data['remainlaboralperiod'] = 65 - test_data['laboralperiod']

    train_data['nobedrooms'] = train_data['rooms'] - train_data['bedrooms']
    test_data['nobedrooms'] = test_data['rooms'] - test_data['bedrooms']

    train_data['rent_per_nobedroom'] = train_data['v2a1']/train_data['nobedrooms']
    train_data['bedroom_per_nobedroom'] = train_data['bedrooms']/train_data['nobedrooms']
    train_data['elder_per_nobedroom'] = train_data['hogar_mayor']/train_data['nobedrooms']
    train_data['adults_per_nobedroom'] = train_data['adult']/train_data['nobedrooms']
    train_data['child_per_nobedroom'] = train_data['hogar_nin']/train_data['nobedrooms']
    train_data['male_per_nobedroom'] = train_data['r4h3']/train_data['nobedrooms']
    train_data['female_per_nobedroom'] = train_data['r4m3']/train_data['nobedrooms']
    train_data['nobedrooms_per_person_household'] = train_data['hhsize']/train_data['nobedrooms']

    test_data['rent_per_nobedroom'] = test_data['v2a1']/test_data['nobedrooms']
    test_data['bedroom_per_nobedroom'] = test_data['bedrooms']/test_data['nobedrooms']
    test_data['elder_per_nobedroom'] = test_data['hogar_mayor']/test_data['nobedrooms']
    test_data['adults_per_nobedroom'] = test_data['adult']/test_data['nobedrooms']
    test_data['child_per_nobedroom'] = test_data['hogar_nin']/test_data['nobedrooms']
    test_data['male_per_nobedroom'] = test_data['r4h3']/test_data['nobedrooms']
    test_data['female_per_nobedroom'] = test_data['r4m3']/test_data['nobedrooms']
    test_data['nobedrooms_per_person_household'] = test_data['hhsize']/test_data['nobedrooms']
    
    return train_data, test_data

def and_even_more_feature_engineering(train_data, test_data):
    
    # Año que le faltan para emanciparse
    def fill_remain_to_indep(x):
        value = 19 - x['age']

        if value < 0:
            value = 0

        return value

    train_data['remain_to_indep'] = train_data.apply(lambda x : fill_remain_to_indep(x),axis=1)
    test_data['remain_to_indep'] = test_data.apply(lambda x : fill_remain_to_indep(x),axis=1)

    # hogar_nin - r4t1
    
    return train_data, test_data

def stat_feature_engineering(train_data, test_data):

    stat_features = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    person_features = ['age','escolari','instlevelx', 'laboralperiod', 'remainlaboralperiod', 'remain_to_indep', 'instlevelx_to_age'] # Completar
    auxiliar_data = pd.DataFrame()

    ## Train transform
    #

    print("    ... data preparation for train_data ...")
    # Transfiero la información a un Dataframe auxiliar
    auxiliar_data['idhogar'] = train_data['idhogar']
    auxiliar_data.set_index('idhogar')

    for f in person_features:
        auxiliar_data[f] = train_data[f]


    # Agrupo la info por hogar.
    auxiliar_data_grouped = auxiliar_data.groupby('idhogar')


    #print(auxiliar_data.columns)
    #print(auxiliar_data.shape)


    #Inicializo las columnas de los nuevos atributos con 0 en auxiliar.
    for pf in person_features:
        for sf in stat_features:
            auxiliar_data['household_agg_'+pf+'_'+sf] = 0

    #print(auxiliar_data.columns)    

    # Calculo los valores de los atributos y los cargo en el dataframe auxiliar.
    print("    ... features calc for train_data ...")
    for pf in person_features:
        for name, group in auxiliar_data_grouped:
            group_result = group[pf].describe()
            for r in range(8):
                auxiliar_data.loc[(auxiliar_data['idhogar'] == name),  'household_agg_'+pf+'_'+stat_features[r]] = group_result[r]
        print('        ... %s  [OK]' % pf)


    # Transfiero los resultados a train_data
    print("    ... features tranfer to train_data ...")
    for pf in person_features: #Inicializo las columnas de los nuevos atributos con 0 en train_data.
        for sf in stat_features:
            train_data['household_agg_'+pf+'_'+sf] = 0

    columns = []
    for pf in person_features: 
        for sf in stat_features:
            columns.append('household_agg_'+pf+'_'+sf)


    train_data.reset_index()
    for i in train_data.index:
        y = auxiliar_data.loc[(auxiliar_data['idhogar']==train_data.get_value(i,'idhogar')), columns]
    #y['household_agg_'+pf+'_'+sf].
        for pf in person_features: 
            for sf in stat_features:
                try:
                    train_data.set_value(i,'household_agg_'+pf+'_'+sf, y.get_value(i,'household_agg_'+pf+'_'+sf))
                except:
                    pass


    ##### Test transform
    #

    print("    ... data preparation for test_data ...")
    # Transfiero la información a un Dataframe auxiliar
    auxiliar_data = pd.DataFrame()
    auxiliar_data['idhogar'] = test_data['idhogar']
    auxiliar_data.set_index('idhogar')

    for f in person_features:
        auxiliar_data[f] = test_data[f]


    # Agrupo la info por hogar.
    auxiliar_data_grouped = auxiliar_data.groupby('idhogar')

    #Inicializo las columnas de los nuevos atributos con 0 en auxiliar.
    for pf in person_features:
        for sf in stat_features:
            auxiliar_data['household_agg_'+pf+'_'+sf] = 0

    # Calculo los valores de los atributos y los cargo en el dataframe auxiliar.
    print("    ... features calc for test_data ...")
    for pf in person_features:
        for name, group in auxiliar_data_grouped:
            group_result = group[pf].describe()
            for r in range(8):
                auxiliar_data.loc[(auxiliar_data['idhogar'] == name),  'household_agg_'+pf+'_'+stat_features[r]] = group_result[r]
        print('        ... %s  [OK]' % pf)


    # Transfiero los resultados a train_data
    print("    ... features tranfer to test_data ...")
    for pf in person_features: #Inicializo las columnas de los nuevos atributos con 0 en train_data.
        for sf in stat_features:
            test_data['household_agg_'+pf+'_'+sf] = 0

    columns = []
    for pf in person_features: 
        for sf in stat_features:
            columns.append('household_agg_'+pf+'_'+sf)


    test_data.reset_index()
    for i in test_data.index:
        y = auxiliar_data.loc[(auxiliar_data['idhogar'] == test_data.get_value(i,'idhogar')), columns]
        for pf in person_features: 
            for sf in stat_features:
                try:
                    test_data.set_value(i,'household_agg_'+pf+'_'+sf, y.get_value(i,'household_agg_'+pf+'_'+sf))
                except:
                    pass

    return train_data, test_data

def ageing_feature_engineering(train_data, test_data):

    ageing_features = ['0_12','12_18','18_65', '65_80', '80_all'] # Fijo
    auxiliar_data = pd.DataFrame()

    ## Train transform
    #

    print("    ... data preparation for train_data ...")
    # Transfiero la información a un Dataframe auxiliar
    auxiliar_data['idhogar'] = train_data['idhogar']
    auxiliar_data.set_index('idhogar')



    # Agrupo la info por hogar.
    #auxiliar_data_grouped = auxiliar_data.groupby('idhogar')


    #Inicializo las columnas de los nuevos atributos con 0 en auxiliar.
    for af in ageing_features:
        auxiliar_data['household_agg_r16m_'+af] = 0
        auxiliar_data['household_agg_r16f_'+af] = 0

    # Calculo los valores de los atributos y los cargo en el dataframe auxiliar.
    print("    ... features calc for train_data ...")
    train_data.reset_index()
    for i in train_data.index:
        age = train_data.get_value(i,'age')
        idhogar = train_data.get_value(i,'idhogar')
        male = True
        if train_data.get_value(i,'male') == 0:
            male = False

        if male == True:
            if 0 <= age and age <= 12:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_0_12']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_0_12'] = current + 1
            elif 12 < age and age <= 18:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_12_18']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_12_18'] = current + 1
            elif 18 < age and age <= 65:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_18_65']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_18_65'] = current + 1
            elif 65 < age and age <= 80:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_65_80']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_65_80'] = current + 1
            elif 80 < age :
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_80_all']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_80_all'] = current + 1
        else:
            if 0 <= age and age <= 12:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_0_12']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_0_12'] = current + 1
            elif 12 < age and age <= 18:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_12_18']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_12_18'] = current + 1
            elif 18 < age and age <= 65:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_18_65']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_18_65'] = current + 1
            elif 65 < age and age <= 80:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_65_80']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_65_80'] = current + 1
            elif 80 < age :
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_80_all']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_80_all'] = current + 1


    # Transfiero los resultados a train_data
    print("    ... features tranfer to train_data ...")
    columns = []
    for af in ageing_features: 
        columns.append('household_agg_r16m_'+af)
        columns.append('household_agg_r16f_'+af)

    train_data.reset_index()
    for i in train_data.index:
        idhogar = train_data.get_value(i,'idhogar')
        y = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), columns]
        for af in ageing_features:
            train_data.set_value(i,'household_agg_r16m_'+af, y.get_value(i,'household_agg_r16m_'+af))        
            train_data.set_value(i,'household_agg_r16f_'+af, y.get_value(i,'household_agg_r16f_'+af))       

    for af in ageing_features: 
        train_data['household_agg_r16t_'+af] = train_data['household_agg_r16m_'+af] + train_data['household_agg_r16f_'+af]


    for af in ageing_features: 
        # r16m_xx_xx_percent_in_male 
        train_data['household_agg_r16m_'+af+'_percent_in_male'] = train_data['household_agg_r16m_'+af] / train_data['r4h3']
        # r16f_xx_xx_percent_in_male 
        train_data['household_agg_r16f_'+af+'_percent_in_female'] = train_data['household_agg_r16f_'+af] / train_data['r4m3']
        # r16m_xx_xx_percent_in_total 
        train_data['household_agg_r16m_'+af+'_percent_in_total'] = train_data['household_agg_r16m_'+af] / train_data['hhsize']
        # r16f_xx_xx_percent_in_total 
        train_data['household_agg_r16f_'+af+'_percent_in_total'] = train_data['household_agg_r16f_'+af] / train_data['hhsize']
        # r16t_xx_xx_percent_in_total 
        train_data['household_agg_r16t_'+af+'_percent_in_total'] = train_data['household_agg_r16t_'+af] / train_data['hhsize']

        # r16m_xx_xx_percent_in_r16t_18_65 
        train_data['household_agg_r16m_'+af+'_percent_in_r16t_18_65'] = train_data['household_agg_r16m_'+af] / train_data['household_agg_r16t_18_65']
        # r16f_xx_xx_percent_in_r16t_18_65 
        train_data['household_agg_r16f_'+af+'_percent_in_r16t_18_65'] = train_data['household_agg_r16f_'+af] / train_data['household_agg_r16t_18_65']
        # r16m_xx_xx_percent_in_r16m_18_65 
        train_data['household_agg_r16m_'+af+'_percent_in_r16t_18_65'] = train_data['household_agg_r16m_'+af] / train_data['household_agg_r16m_18_65']
        # r16f_xx_xx_percent_in_r16m_18_65 
        train_data['household_agg_r16f_'+af+'_percent_in_r16t_18_65'] = train_data['household_agg_r16f_'+af] / train_data['household_agg_r16m_18_65']
        # r16m_xx_xx_percent_in_r16f_18_65 
        train_data['household_agg_r16m_'+af+'_percent_in_r16t_18_65'] = train_data['household_agg_r16m_'+af] / train_data['household_agg_r16f_18_65']
        # r16f_xx_xx_percent_in_r16f_18_65 
        train_data['household_agg_r16f_'+af+'_percent_in_r16t_18_65'] = train_data['household_agg_r16f_'+af] / train_data['household_agg_r16f_18_65']


    ## Test transform
    #

    print("    ... data preparation for test_data ...")
    # Transfiero la información a un Dataframe auxiliar
    auxiliar_data = pd.DataFrame()
    auxiliar_data['idhogar'] = test_data['idhogar']
    auxiliar_data.set_index('idhogar')

    #Inicializo las columnas de los nuevos atributos con 0 en auxiliar.
    for af in ageing_features:
        auxiliar_data['household_agg_r16m_'+af] = 0
        auxiliar_data['household_agg_r16f_'+af] = 0

    # Calculo los valores de los atributos y los cargo en el dataframe auxiliar.
    print("    ... features calc for test_data ...")
    test_data.reset_index()
    for i in test_data.index:
        age = test_data.get_value(i,'age')
        idhogar = test_data.get_value(i,'idhogar')
        male = True
        if test_data.get_value(i,'male') == 0:
            male = False

        if male == True:
            if 0 <= age and age <= 12:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_0_12']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_0_12'] = current + 1
            elif 12 < age and age <= 18:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_12_18']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_12_18'] = current + 1
            elif 18 < age and age <= 65:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_18_65']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_18_65'] = current + 1
            elif 65 < age and age <= 80:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_65_80']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_65_80'] = current + 1
            elif 80 < age :
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_80_all']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16m_80_all'] = current + 1
        else:
            if 0 <= age and age <= 12:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_0_12']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_0_12'] = current + 1
            elif 12 < age and age <= 18:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_12_18']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_12_18'] = current + 1
            elif 18 < age and age <= 65:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_18_65']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_18_65'] = current + 1
            elif 65 < age and age <= 80:
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_65_80']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_65_80'] = current + 1
            elif 80 < age :
                current = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_80_all']
                auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), 'household_agg_r16f_80_all'] = current + 1


    # Transfiero los resultados a test_data
    print("    ... features tranfer to test_data ...")
    columns = []
    for af in ageing_features: 
        columns.append('household_agg_r16m_'+af)
        columns.append('household_agg_r16f_'+af)

    test_data.reset_index()
    for i in test_data.index:
        idhogar = test_data.get_value(i,'idhogar')
        y = auxiliar_data.loc[(auxiliar_data['idhogar']==idhogar), columns]
        for af in ageing_features:
            try:
                test_data.set_value(i,'household_agg_r16m_'+af, y.get_value(i,'household_agg_r16m_'+af))        
                test_data.set_value(i,'household_agg_r16f_'+af, y.get_value(i,'household_agg_r16f_'+af))
            except:
                print('Exception happends:  column household_agg_r16x_'+af+' idhogar:' +str(idhogar)) #pass


    for af in ageing_features: 
        test_data['household_agg_r16t_'+af] = test_data['household_agg_r16m_'+af] + test_data['household_agg_r16f_'+af]

    for af in ageing_features: 
        # r16m_xx_xx_percent_in_male 
        test_data['household_agg_r16m_'+af+'_percent_in_male'] = test_data['household_agg_r16m_'+af] / test_data['r4h3']
        # r16f_xx_xx_percent_in_male 
        test_data['household_agg_r16f_'+af+'_percent_in_female'] = test_data['household_agg_r16f_'+af] / test_data['r4m3']
        # r16m_xx_xx_percent_in_total 
        test_data['household_agg_r16m_'+af+'_percent_in_total'] = test_data['household_agg_r16m_'+af] / test_data['hhsize']
        # r16f_xx_xx_percent_in_total 
        test_data['household_agg_r16f_'+af+'_percent_in_total'] = test_data['household_agg_r16f_'+af] / test_data['hhsize']
        # r16t_xx_xx_percent_in_total 
        test_data['household_agg_r16t_'+af+'_percent_in_total'] = test_data['household_agg_r16t_'+af] / test_data['hhsize']

        # r16m_xx_xx_percent_in_r16t_18_65 
        test_data['household_agg_r16m_'+af+'_percent_in_r16t_18_65'] = test_data['household_agg_r16m_'+af] / test_data['household_agg_r16t_18_65']
        # r16f_xx_xx_percent_in_r16t_18_65 
        test_data['household_agg_r16f_'+af+'_percent_in_r16t_18_65'] = test_data['household_agg_r16f_'+af] / test_data['household_agg_r16t_18_65']
        # r16m_xx_xx_percent_in_r16m_18_65 
        test_data['household_agg_r16m_'+af+'_percent_in_r16t_18_65'] = test_data['household_agg_r16m_'+af] / test_data['household_agg_r16m_18_65']
        # r16f_xx_xx_percent_in_r16m_18_65 
        test_data['household_agg_r16f_'+af+'_percent_in_r16t_18_65'] = test_data['household_agg_r16f_'+af] / test_data['household_agg_r16m_18_65']
        # r16m_xx_xx_percent_in_r16f_18_65 
        test_data['household_agg_r16m_'+af+'_percent_in_r16t_18_65'] = test_data['household_agg_r16m_'+af] / test_data['household_agg_r16f_18_65']
        # r16f_xx_xx_percent_in_r16f_18_65 
        test_data['household_agg_r16f_'+af+'_percent_in_r16t_18_65'] = test_data['household_agg_r16f_'+af] / test_data['household_agg_r16f_18_65']

    return train_data, test_data

def save(train_data, test_data):
    train_data.to_csv('train_data'+ str(start) +'.csv')
    test_data.to_csv('test_data'+ str(start) +'.csv')

    train_data.to_csv('train_data'+ 'current' +'.csv')
    test_data.to_csv('test_data'+ 'current' +'.csv')
    
    return

def restore():
    train_data =pd.read_csv('train_data'+ 'current' +'.csv')
    test_data =pd.read_csv('test_data'+ 'current' +'.csv')
    
    return train_data, test_data

def feature_reduction(train_data, test_data):
    
    aux_data = train_data.copy()
    aux_data = aux_data.query('parentesco1==1').copy()

    result = pd.DataFrame(columns=['macro_F1_score', 'param', 'Runtime'])

    X_train, X_test, y_train, y_test = train_test_split(aux_data, aux_data['Target'], test_size=0.25, 
                                                            random_state=314, stratify=aux_data['Target'])        


    try:
        X_train.drop(columns=['Target', 'idhogar', 'Id'], inplace=True)
        X_train.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass

    try:
        X_test.drop(columns=['Target', 'idhogar', 'Id'], inplace=True)
        X_test.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass

    notebookstart= time.time()
    clf = lgb.LGBMClassifier(max_depth=11, learning_rate=0.1, objective='multiclass',
                                 random_state=None, silent=True, verbose=-1, metric='None', 
                                 n_jobs=3, n_estimators=6000, class_weight='balanced',
                                 colsample_bytree =  0.89, num_leaves = 32,
                                     min_data_in_leaf=29,
                                 subsample = 0.96)
    kfold = 5
    kf = StratifiedKFold(n_splits=kfold, shuffle=True)

    print('    ... training LGBMClassifier.')
    for train_index, test_index in kf.split(X_train, y_train):
        X_set, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_set, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        clf.fit(X_set, y_set, eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=400)

    y_test_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_test_pred, average='macro')

    print("    ... macro F1 score: {0:10.6f}, Runtime: {1:0.2f} min.\n".format(f1, (time.time() - notebookstart)/60))

    num_features = clf.feature_importances_.shape[0] 
    importance_split = clf.booster_.feature_importance(importance_type='split')
    importance_gain = clf.booster_.feature_importance(importance_type='gain')
    feature_names = clf.booster_.feature_name()

    features = pd.DataFrame(columns=['feature', 'importance_split', 'importance_gain'])
    for i in range(0,num_features):
        features.loc[i]=[feature_names[i], importance_split[i], importance_gain[i]]

#    print("    ... Feature Ranking:")
#    print(features.sort_values(by='importance_split', axis='index', ascending=False))

    null_features = features.loc[(features['importance_split'] == 0), 'feature']
    cant = X_train.shape[0] #null_features.shape[0]
    columnas = []
    for i in range(0,cant):
        try: 
            null_column = null_features.get_value(i,'feature')
            if null_column != 'parentesco1' and null_column != 'idhogar':
                columnas.append( null_column )
        except:
            pass

    train_data.drop(columns=columnas, inplace=True)
    test_data.drop(columns=columnas, inplace=True)

    return train_data, test_data

def gridsearch(train_data, test_data):

    param_grid = {
        'min_child_samples' : [29,30,31,32],
        'max_depth': [10,11,12]
    }

    param_grid['min_child_samples'] = range(29,35)
    param_grid['max_depth'] = range(9,15)

    gridsearchcv_start= time.time()
    X_set = train_data.copy()
    X_set = X_set.query('parentesco1==1').copy()

    y_set = X_set['Target']
    try:
        X_set.drop(columns=['Target', 'idhogar', 'Id'], inplace=True)
        X_set.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass


    estimator = lgb.LGBMClassifier(max_depth=11, learning_rate=0.09, objective='multiclass',
                             random_state=16, silent=True, verbose=-1, metric='None', 
                             n_jobs=16, n_estimators=4000, class_weight='balanced',
                             colsample_bytree =  0.89, min_child_samples = 33,
                                 min_data_in_leaf=31,
                             subsample = 0.96)
    gbm = GridSearchCV(estimator, param_grid, scoring='f1_macro', cv=5, verbose=6, n_jobs=3)
    gbm.fit(X_set, y_set)
    print("    ... Ending GridSearchCV cycle. Runtime: {0:0.2f} min.\n".format((time.time() - gridsearchcv_start)/60))
    print('    ... Best score found by grid search are:', gbm.best_score_ )
    print('    ... and Best parameters search are:', gbm.best_params_)
    print('    -----------------------------------------------')

    current_best_min_child_samples = gbm.best_params_['min_child_samples']
    current_best_max_depth = gbm.best_params_['max_depth']

    gbm.best_estimator_.booster_.save_model('boster-'+str(time.time())+'.txt')

    return current_best_min_child_samples, current_best_max_depth, gbm #train_data, test_data


def hyperopt(min_child_samples, max_depth, train_data, test_data):

    result = pd.DataFrame(columns=['macro_F1_score', 'num_leaves', 'max_depth', 'Runtime'])
    index = 0

    aux_data = train_data.query('parentesco1==1').copy()
    X_train = aux_data.copy()
    y_train = X_train['Target']

    X_train.drop(columns=['Target', 'idhogar', 'Id', 'Unnamed: 0'], inplace=True)

    clf = lgb.LGBMClassifier(max_depth=max_depth, learning_rate=0.095, objective='multiclass',
                                     random_state=None, silent=True, verbose=-1, metric='None', 
                                     n_jobs=4, n_estimators=10000, class_weight='balanced',
                                     colsample_bytree =  0.89, min_data_in_leaf=min_child_samples,
                                     subsample = 0.96)
    kfold = 5
    kf = StratifiedKFold(n_splits=kfold, shuffle=True)
    for train_index, test_index in kf.split(X_train, y_train):
        X_set, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_set, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
        clf.fit(X_set, y_set, eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=400)


    return result, clf

def submission_preparation(clf,train_data, test_data):

    X_set = test_data.copy()

    try:
        X_set.drop(columns=['idhogar', 'Id'], inplace=True)
        X_set.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass

    sample_submission = pd.DataFrame(columns=['Id', 'Target'])


    sample_submission['Id'] = test_data['Id']
    sample_submission['Target'] = clf.predict(X_set)

    sample_submission.to_csv('submission_'+ str(start) +'.csv')
    sample_submission.to_csv('submission.csv')
    
    return


def main(Debug = False):
    with timer("Load Data"):
        train_data, test_data = get_data(Debug=Debug)
    with timer('Handling Missing Data'):
        train_data, test_data = handling_missing_data(train_data, test_data)
    with timer('Feature Engineering'):
        train_data, test_data = feature_engineering(train_data, test_data)
    with timer('More Feature Engineering'):
        train_data, test_data = more_feature_engineering(train_data, test_data)
    with timer('Even More Feature Engineering'):
        train_data, test_data = even_more_feature_engineering(train_data, test_data)
    with timer('And Even More Feature Engineering'):
        train_data, test_data = and_even_more_feature_engineering(train_data, test_data)
    with timer('Stat Feature Engineering'):
        train_data, test_data = stat_feature_engineering(train_data, test_data)
    with timer('Ageing Feature Engineering'):
        train_data, test_data = ageing_feature_engineering(train_data, test_data)
    with timer('Back Up Point'):
        save(train_data, test_data)
    with timer('Restore Point'):
        train_data, test_data = restore()
    with timer('Features Importance'):
        train_data, test_data = feature_reduction(train_data, test_data)

#    with timer('GridSearchCV'):
#        current_best_min_child_samples, current_best_max_depth, gbm = gridsearch(train_data, test_data)
    current_best_min_child_samples, current_best_max_depth = 29, 13
    with timer('HyperOPt Liquid Force'):
        result, clf = hyperopt(current_best_min_child_samples, current_best_max_depth, train_data, test_data)

    with timer('Submission Preparation'):
        submission_preparation( clf, train_data, test_data)

        
    return

if __name__ == '__main__':
    main(Debug = False)
